from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
    LoadModelDirectives,
    ProcessModelDirectives,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Simulator,
    LeafSystem,
    BasicVector,
    StartMeshcat,
    MultibodyPlant,
)
import numpy as np
import sys
sys.path.append("../..")
from support_functions import AddShape, AddMultibodyTriad
from g1_cartpole_mpc import KoopmanMPCController
meshcat = StartMeshcat()

class G1Env:
    def __init__(self, time_step: float = 0.001):
        self.builder = DiagramBuilder()
        self.time_step = time_step

        # ── Simulation plant: G1 arm + Cart‑Pole ───────────────────────────────
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=time_step)
        parser = Parser(self.plant)
        parser.package_map().Add("drake_project", "./")
        directives_sim = LoadModelDirectives(
            "models/g1_description/g1_7dof_cartpole.yaml"
        )
        ProcessModelDirectives(directives_sim, self.plant, parser)
        self.plant.Finalize()
        G1_inst = self.plant.GetModelInstanceByName("G1")

        # ── Visualization frames ──────────────────────────────────────────────
        AddMultibodyTriad(
            self.plant.GetFrameByName("pelvis"),
            self.scene_graph, length=0.1, radius=0.005)

        # ── MeshCat visualizer ────────────────────────────────────────────────
        params = MeshcatVisualizerParams()
        MeshcatVisualizer.AddToBuilder(
            self.builder, self.scene_graph, meshcat, params)

        # ── Koopman MPC Controller ────────────────────────────────────────────
        self.mpc = self.builder.AddSystem(
            KoopmanMPCController(
                plant=self.plant,
                koopman_ckpt=(
                    "../log/best_models/G1/"
                    "best_model_norm_G1CartPole_512_1_1_1.pth"
                ),
                dataset_pt=(
                    "../data/datasets/"
                    "dataset_G1CartPole_norm_train_60000_val_20000_test_20000_steps_50.pt"
                ),
                urdf_yaml="models/g1_description/g1_7dof_cartpole.yaml"
            )
        )

        # ── System connections ────────────────────────────────────────────────
        # Connect plant state to MPC input
        self.builder.Connect(
            self.plant.get_state_output_port(G1_inst),
            self.mpc.get_input_port(0))
        
        # Connect MPC output to plant actuation
        self.builder.Connect(
            self.mpc.get_output_port(0),
            self.plant.get_actuation_input_port(G1_inst))

        # ── Joint limits storage ──────────────────────────────────────────────
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

        self.diagram = self.builder.Build()

    def run_simulation(self, duration: float = 10.0):
        ctx = self.diagram.CreateDefaultContext()
        sim = Simulator(self.diagram, ctx)
        sim.set_target_realtime_rate(1.0)
        meshcat.StartRecording(set_visualizations_while_recording=True)
        sim.AdvanceTo(duration)
        meshcat.PublishRecording()

# The KoopmanMPCController implementation from previous answer should be here
# Make sure to include it in the same file or import it

if __name__ == "__main__":
    env = G1Env()
    env.run_simulation(duration=10.0)