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
    ZeroOrderHold,
)
import numpy as np
import sys
sys.path.append("../..")
from support_functions import AddShape, AddMultibodyTriad

meshcat = StartMeshcat()


class G1Env:
    def __init__(self, time_step: float = 0.0001):
        self.builder = DiagramBuilder()
        self.time_step = time_step

        # ── Single plant for both simulation and control ──────────────────────
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=time_step)
        parser = Parser(self.plant)
        parser.package_map().Add("drake_project", "./")
        
        # Load full system (arm + cart-pole)
        directives = LoadModelDirectives(
            "models/g1_description/g1_7dof_cartpole.yaml"
        )
        ProcessModelDirectives(directives, self.plant, parser)
        self.plant.Finalize()
        self.G1_inst = self.plant.GetModelInstanceByName("G1")

        # ── Visualization setup ───────────────────────────────────────────────
        AddMultibodyTriad(
            self.plant.GetFrameByName("pelvis"),
            self.scene_graph, length=0.1, radius=0.005)

        # ── Direct torque application (no controller) ─────────────────────────
        self.DeclareActuationPort()

        # ── Joint limits storage ─────────────────────────────────────────────
        joint_names = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        self._store_joint_limits(joint_names)

        # ── MeshCat visualizer ───────────────────────────────────────────────
        MeshcatVisualizer.AddToBuilder(
            self.builder, self.scene_graph, meshcat,
            MeshcatVisualizerParams())

        self.diagram = self.builder.Build()

    def _store_joint_limits(self, joint_names):
        """Store joint limits from the plant."""
        self.joint_lower_limits = []
        self.joint_upper_limits = []
        for name in joint_names:
            joint = self.plant.GetJointByName(name, self.G1_inst)
            self.joint_lower_limits.append(joint.position_lower_limits()[0])
            self.joint_upper_limits.append(joint.position_upper_limits()[0])
        self.joint_lower_limits = np.array(self.joint_lower_limits)
        self.joint_upper_limits = np.array(self.joint_upper_limits)

    def DeclareActuationPort(self):
        """Set up direct torque input port."""
        self.torque_input = self.builder.AddSystem(
            ZeroOrderHold(self.time_step, self.plant.num_actuators())
        )
        self.builder.Connect(
            self.torque_input.get_output_port(),
            self.plant.get_actuation_input_port(self.G1_inst)
        )
        self.torque_port = self.torque_input.get_input_port()

    def get_state_port(self):
        """Get the full state output port."""
        return self.plant.get_state_output_port(self.G1_inst)

    def get_torque_port(self):
        """Get the torque input port."""
        return self.torque_port

    def run_simulation(self, duration: float = 2.0):
        """Run simulation with direct torque input."""
        ctx = self.diagram.CreateDefaultContext()
        sim = Simulator(self.diagram, ctx)
        sim.set_target_realtime_rate(1.0)
        meshcat.StartRecording()
        sim.AdvanceTo(duration)
        meshcat.PublishRecording()
