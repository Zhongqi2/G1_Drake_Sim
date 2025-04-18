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
import numpy as np
import sys
sys.path.append("../..")
from support_functions import AddShape, AddMultibodyTriad

meshcat = StartMeshcat()


class ArmStateExtractor(LeafSystem):
    """Extracts first N arm positions + velocities out of a full state vector."""
    def __init__(self, full_dim: int, arm_q: int):
        super().__init__()
        self._Np = arm_q
        self._half = full_dim // 2
        self.DeclareVectorInputPort("full_state", BasicVector(full_dim))
        self.DeclareVectorOutputPort(
            "arm_state", BasicVector(2 * arm_q), self.CalcArmState)

    def CalcArmState(self, context, output):
        full = self.get_input_port(0).Eval(context)
        q = full[: self._Np]
        v = full[self._half : self._half + self._Np]
        output.SetFromVector(np.concatenate([q, v]))


class G1Env:
    def __init__(self, time_step: float = 0.001):
        self.builder = DiagramBuilder()
        self.time_step = time_step

        # ── Simulation plant: G1 arm + Cart‑Pole
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=time_step)
        parser = Parser(self.plant)
        parser.package_map().Add("drake_project", "./")
        directives_sim = LoadModelDirectives(
            "models/g1_description/g1_7dof_cartpole.yaml"
        )
        ProcessModelDirectives(directives_sim, self.plant, parser)
        self.plant.Finalize()

        # ── Controller plant: ONLY G1 arm
        self.controller_plant = MultibodyPlant(time_step=0.0)
        parser1 = Parser(self.controller_plant)
        parser1.package_map().Add("drake_project", "./")
        directives_ctrl = LoadModelDirectives(
            "models/g1_description/g1_7dof_rubber_hand.yaml"
        )
        ProcessModelDirectives(directives_ctrl, self.controller_plant, parser1)
        self.controller_plant.Finalize()

        # ── Visualization frames
        G1_inst = self.plant.GetModelInstanceByName("G1")
        AddMultibodyTriad(
            self.plant.GetFrameByName("pelvis"),
            self.scene_graph, length=0.1, radius=0.005)
        # AddMultibodyTriad(
        #     self.plant.GetFrameByName("right_wrist_yaw_link"), 
        #     self.scene_graph, length=0.1, radius=0.005)

        # AddMultibodyTriad(
        #     self.plant.GetFrameByName("cartpole_pole"),
        #     self.scene_graph, length=0.1, radius=0.005)

        # ── Joint limits for the 7‑DOF arm
        joint_names = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        self.joint_lower_limits = np.zeros(len(joint_names))
        self.joint_upper_limits = np.zeros(len(joint_names))
        for i, name in enumerate(joint_names):
            j = self.plant.GetJointByName(name, G1_inst)
            self.joint_lower_limits[i] = j.position_lower_limits()[0]
            self.joint_upper_limits[i] = j.position_upper_limits()[0]

        # ── MeshCat visualizer
        params = MeshcatVisualizerParams()
        MeshcatVisualizer.AddToBuilder(
            self.builder, self.scene_graph, meshcat, params)

        # ── InverseDynamicsController on 7‑DOF arm
        m = self.controller_plant.num_positions()  # should be 7
        kp = np.full((m, 1), 50.0)
        ki = np.full((m, 1),  1.0)
        kd = np.full((m, 1), 10.0)

        controller = self.builder.AddSystem(
            InverseDynamicsController(
                self.controller_plant,
                kp, ki, kd,
                False   # has_reference_acceleration
            )
        )
        controller.set_name("robot_controller")

        # ── Trajectory command & interpolator
        traj = self.MakeRobotCommandTrajectory()
        traj_source = self.builder.AddSystem(TrajectorySource(traj))
        interp = self.builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                m, time_step, suppress_initial_transient=True
            )
        )
        self.builder.Connect(traj_source.get_output_port(),
                             interp.get_input_port())
        self.builder.Connect(interp.get_output_port(),
                             controller.get_input_port_desired_state())

        # ── Extract arm-only state (14 D) from full (18 D) before controller
        full_dim = self.plant.num_multibody_states()  # 2*(7+2)=18
        extractor = self.builder.AddSystem(
            ArmStateExtractor(full_dim, m)
        )
        self.builder.Connect(
            self.plant.get_state_output_port(G1_inst),
            extractor.get_input_port(0)
        )
        self.builder.Connect(
            extractor.get_output_port(0),
            controller.get_input_port_estimated_state()
        )

        # ── Send controller torques back to sim plant
        self.builder.Connect(
            controller.get_output_port_control(),
            self.plant.get_actuation_input_port(G1_inst)
        )

        self.diagram = self.builder.Build()

    def MakeRobotCommandTrajectory(self):
        """Build a random 2‑second, 1st‑order‐hold trajectory in the 7‑DOF joint space."""
        m = self.controller_plant.num_positions()
        T = 2.0
        q_init = np.zeros(m)
        q_end  = (
            self.joint_lower_limits
            + (self.joint_upper_limits - self.joint_lower_limits)
              * np.random.rand(m)
        )
        A = np.vstack((q_init, q_end)).T
        return PiecewisePolynomial.FirstOrderHold([0, T], A)

    def run_simulation(self, duration: float = 2.0):
        ctx = self.diagram.CreateDefaultContext()
        sim = Simulator(self.diagram, ctx)
        sim.set_target_realtime_rate(1.0)
        meshcat.StartRecording(set_visualizations_while_recording=True)
        sim.AdvanceTo(duration)
        meshcat.PublishRecording()
