from pydrake.all import *
import numpy as np
import sys
sys.path.append("../..")
from support_functions import AddMultibodyTriad

meshcat = StartMeshcat()

class G1Env():
    def __init__(self, time_step=0.001):
        self.builder = DiagramBuilder()
        self.time_step = time_step
        self.num_positions = 8
        
        # Create plant and scene graph
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step)
        parser = Parser(self.plant)
        parser.package_map().Add("drake_project", "./")
        directives = LoadModelDirectives("models/g1_description/g1_7dof_rubber_hand.yaml")
        #directives = LoadModelDirectives("models/g1_description/g1_7dof.yaml")
        ProcessModelDirectives(directives, self.plant, parser)
        
        # Add visualization frames
        G1 = self.plant.GetModelInstanceByName("G1")
        base_frame = self.plant.GetFrameByName("pelvis", G1)
        #hand_frame = self.plant.GetFrameByName("cart", G1)
        AddMultibodyTriad(base_frame, self.scene_graph)
        #AddMultibodyTriad(hand_frame, self.scene_graph)
        
        self.plant.Finalize()
        
        # Create controller plant (for dynamics calculations)
        self.controller_plant = MultibodyPlant(0)
        parser = Parser(self.controller_plant)
        parser.package_map().Add("drake_project", "./")
        ProcessModelDirectives(directives, self.controller_plant, parser)
        self.controller_plant.Finalize()
        
        # Joint limits for arm joints only
        self.joint_limits = [
            (self.controller_plant.GetJointByName(f"right_{joint}").position_lower_limits()[0],
             self.controller_plant.GetJointByName(f"right_{joint}").position_upper_limits()[0])
            for joint in ["shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
                          "elbow_joint", "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint"]
        ]
        
        # Set up visualization
        MeshcatVisualizer.AddToBuilder(self.builder, self.scene_graph, meshcat)
        self.diagram = self.builder.Build()
        
    def step(self, x, u):
        # plant_context = self.controller_plant.CreateDefaultContext()
        # self.controller_plant.SetPositions(plant_context, x[:self.num_positions])
        # self.controller_plant.SetVelocities(plant_context, x[self.num_positions:])
        # M = self.controller_plant.CalcMassMatrix(plant_context)
        # Cv = self.controller_plant.CalcBiasTerm(plant_context)
        # tauG = self.controller_plant.CalcGravityGeneralizedForces(plant_context)
        # q_ddot = np.linalg.inv(M) @ (u - tauG - Cv)  # Compute acceleration q̈
        # x_dot = np.concatenate((x[self.num_positions:],q_ddot))
        # x_next = x + x_dot * self.time_step
        
        # return x_next
        context = self.controller_plant.CreateDefaultContext()
        
        # Set state (positions and velocities)
        q = x[:self.num_positions]
        v = x[self.num_positions:]
        self.controller_plant.SetPositions(context, q)
        self.controller_plant.SetVelocities(context, v)
        
        # Compute dynamics
        M = self.controller_plant.CalcMassMatrix(context)
        Cv = self.controller_plant.CalcBiasTerm(context)
        tau_g = self.controller_plant.CalcGravityGeneralizedForces(context)
        
        # Apply control (zero torque for pole joint)
        tau = np.zeros(self.num_positions)
        tau[:7] = u
        
        # Compute acceleration
        v_dot = np.linalg.solve(M, tau - Cv - tau_g)
        
        # Euler integration
        x_dot = np.concatenate([v, v_dot])
        x_next = x + x_dot * self.time_step
        
        return x_next
    
    def run_simulation(self):
        simulator = Simulator(self.diagram)
        simulator.AdvanceTo(10)