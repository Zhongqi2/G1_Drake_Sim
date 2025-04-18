from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
    LoadModelDirectives,
    ProcessModelDirectives,
    InverseDynamicsController,
    TrajectorySource,
    StateInterpolatorWithDiscreteDerivative,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Simulator,
    LeafSystem,
    BasicVector,
    StartMeshcat,
    MultibodyPlant,
    PiecewisePolynomial,
)
from pydrake.multibody.tree import JointIndex, ModelInstanceIndex
import numpy as np
from support_functions import AddMultibodyTriad

# Shared Meshcat instance
meshcat = StartMeshcat()


class ArmStateExtractor(LeafSystem):
    """Extracts first N arm positions & velocities from full state."""

    def __init__(self, full_dim: int, arm_q: int):
        super().__init__()
        self._Np = arm_q
        self._half = full_dim // 2
        self.DeclareVectorInputPort("full_state", BasicVector(full_dim))
        self.DeclareVectorOutputPort(
            "arm_state", BasicVector(2 * arm_q), self.CalcArmState
        )

    def CalcArmState(self, context, output):
        full = self.get_input_port(0).Eval(context)
        q = full[: self._Np]
        v = full[self._half : self._half + self._Np]
        output.SetFromVector(np.concatenate([q, v]))


class KinovaEnv:
    """Kinova Gen3 with cart‑pole tool on wrist."""

    def __init__(self, time_step: float = 1e-3):
        self.builder = DiagramBuilder()
        self.time_step = time_step

        # ─ Simulation plant: Gen3 + cart‑pole
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=time_step
        )
        parser = Parser(self.plant)
        parser.package_map().Add("drake_project", "./")
        directives = LoadModelDirectives("models/kinova_description/robot_environment.dmd.yaml")
        ProcessModelDirectives(directives, self.plant, parser)
        self.plant.Finalize()

        # ─ Controller plant: only Gen3 arm
        self.controller_plant = MultibodyPlant(time_step=0.0)
        parser_ctrl = Parser(self.controller_plant)
        parser_ctrl.package_map().Add("drake_project", "./")
        # Load URDF via Parser.AddModelsFromUrl
        gp7_models = parser_ctrl.AddModelsFromUrl(
            "package://drake_project/models/kinova_description/GEN3_URDF_V12.urdf"
        )
        gp7_model = (
            gp7_models[0]
            if isinstance(gp7_models, (list, tuple))
            else gp7_models
        )
        # Weld base_link to world
        self.controller_plant.WeldFrames(
            self.controller_plant.world_frame(),
            self.controller_plant.GetFrameByName("base_link", gp7_model),
        )
        # Add actuators to all revolute joints of gp7_model
        for i in range(self.controller_plant.num_joints()):
            joint = self.controller_plant.get_joint(JointIndex(i))
            if joint.num_positions() == 0:
                continue
            if joint.model_instance() != gp7_model:
                continue
            self.controller_plant.AddJointActuator(
                f"{joint.name()}_actuator", joint
            )
        self.controller_plant.Finalize()

        # ─ Triad at base
        gp7_inst = self.plant.GetModelInstanceByName("gp7")
        AddMultibodyTriad(
            self.plant.GetFrameByName("base_link", gp7_inst),
            self.scene_graph,
            length=0.1,
            radius=0.005,
        )

        # ─ Joint limits
        self._compute_joint_limits()

        # ─ Visualizer
        MeshcatVisualizer.AddToBuilder(
            self.builder,
            self.scene_graph,
            meshcat,
            MeshcatVisualizerParams(),
        )

        # ─ Controller
        m = self.controller_plant.num_positions()
        kp, ki, kd = [np.full((m, 1), v) for v in (50.0, 1.0, 10.0)]
        controller = self.builder.AddSystem(
            InverseDynamicsController(
                self.controller_plant,
                kp,
                ki,
                kd,
                has_reference_acceleration=False,
            )
        )
        controller.set_name("robot_controller")

        # ─ Trajectory & interpolation
        traj = self._make_random_joint_trajectory()
        traj_source = self.builder.AddSystem(TrajectorySource(traj))
        interp = self.builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                m, time_step, suppress_initial_transient=True
            )
        )
        self.builder.Connect(
            traj_source.get_output_port(), interp.get_input_port()
        )
        self.builder.Connect(
            interp.get_output_port(), controller.get_input_port_desired_state()
        )

        # ─ State extraction
        full_dim = self.plant.num_multibody_states()
        extractor = self.builder.AddSystem(
            ArmStateExtractor(full_dim, m)
        )
        self.builder.Connect(
            self.plant.get_state_output_port(gp7_inst),
            extractor.get_input_port(0),
        )
        self.builder.Connect(
            extractor.get_output_port(0),
            controller.get_input_port_estimated_state(),
        )

        # ─ Torque routing
        self.builder.Connect(
            controller.get_output_port_control(),
            self.plant.get_actuation_input_port(gp7_inst),
        )

        self.diagram = self.builder.Build()

    def _compute_joint_limits(self):
        lows, highs = [], []
        for i in range(self.controller_plant.num_joints()):
            joint = self.controller_plant.get_joint(JointIndex(i))
            if joint.num_positions() == 0:
                continue
            lows.append(joint.position_lower_limits()[0])
            highs.append(joint.position_upper_limits()[0])
        self.joint_lower_limits = np.array(lows)
        self.joint_upper_limits = np.array(highs)

    def _make_random_joint_trajectory(self, horizon: float = 2.0):
        m = self.controller_plant.num_positions()
        q_init = np.zeros(m)
        q_final = (
            self.joint_lower_limits +
            (self.joint_upper_limits - self.joint_lower_limits) * np.random.rand(m)
        )
        return PiecewisePolynomial.FirstOrderHold(
            [0.0, horizon], np.vstack((q_init, q_final)).T
        )

    def run_simulation(self, duration: float = 2.0, realtime_rate: float = 1.0):
        ctx = self.diagram.CreateDefaultContext()
        sim = Simulator(self.diagram, ctx)
        sim.set_target_realtime_rate(realtime_rate)
        meshcat.StartRecording(set_visualizations_while_recording=True)
        sim.AdvanceTo(duration)
        meshcat.PublishRecording()


if __name__ == "__main__":
    env = KinovaEnv()
    env.run_simulation(3.0)

    import ipdb; ipdb.set_trace()
