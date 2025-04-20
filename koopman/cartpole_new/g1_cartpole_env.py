from pydrake.all import *
import numpy as np
import sys
sys.path.append("../..")
from support_functions import AddMultibodyTriad
from tqdm import tqdm

meshcat = StartMeshcat()

class G1Env():
    def __init__(self, env_name="g1_cartpole", time_step=0.001):
        self.builder = DiagramBuilder()
        self.time_step = time_step
        self.num_positions = 8 if env_name == "g1_cartpole" else 7
        file_path = "models/g1_description/g1_7dof_cartpole.yaml" if env_name == "g1_cartpole" else "models/g1_description/g1_7dof.yaml"
 
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step)
        parser = Parser(self.plant)
        parser.package_map().Add("drake_project", "./")
        directives = LoadModelDirectives(file_path)
        ProcessModelDirectives(directives, self.plant, parser)

        G1 = self.plant.GetModelInstanceByName("G1")
        base_frame = self.plant.GetFrameByName("pelvis", G1)
        AddMultibodyTriad(base_frame, self.scene_graph)
        
        self.plant.Finalize()

        self.controller_plant = MultibodyPlant(0)
        parser = Parser(self.controller_plant)
        parser.package_map().Add("drake_project", "./")
        ProcessModelDirectives(directives, self.controller_plant, parser)
        self.controller_plant.Finalize()
        
        self.joint_limits = [
            (self.controller_plant.GetJointByName(f"right_{joint}").position_lower_limits()[0],
             self.controller_plant.GetJointByName(f"right_{joint}").position_upper_limits()[0])
            for joint in ["shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
                          "elbow_joint", "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint"]
        ]
        
        MeshcatVisualizer.AddToBuilder(self.builder, self.scene_graph, meshcat)
        self.diagram = self.builder.Build()

        self.diagram_context = self.diagram.CreateDefaultContext()
        self.simulator = Simulator(self.diagram, self.diagram_context)
        self.simulator.set_publish_every_time_step(False)
        self.simulator.set_target_realtime_rate(1.0)
        
    def step(self, x, u):
        context = self.controller_plant.CreateDefaultContext()

        q = x[:self.num_positions]
        v = x[self.num_positions:]
        self.controller_plant.SetPositions(context, q)
        self.controller_plant.SetVelocities(context, v)
        
        M = self.controller_plant.CalcMassMatrix(context)
        Cv = self.controller_plant.CalcBiasTerm(context)
        tau_g = self.controller_plant.CalcGravityGeneralizedForces(context)
        
        tau = np.zeros(self.num_positions)
        tau[:7] = u
        
        v_dot = np.linalg.solve(M, tau - Cv - tau_g)
        
        x_dot = np.concatenate([v, v_dot])
        x_next = x + x_dot * self.time_step
        
        return x_next
    
    def run_simulation(self):
        simulator = Simulator(self.diagram)
        simulator.AdvanceTo(10)

    def publish_visualization(self, x):
        plant_context = self.diagram.GetMutableSubsystemContext(self.plant, self.diagram_context)
        q = x[:self.num_positions]
        v = x[self.num_positions:]
        self.plant.SetPositions(plant_context, q)
        self.plant.SetVelocities(plant_context, v)

        # Publish to MeshCat
        self.diagram.ForcedPublish(self.diagram_context)