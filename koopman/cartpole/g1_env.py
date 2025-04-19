from pydrake.all import *
import numpy as np
import sys
sys.path.append("../..")
from support_functions import AddMultibodyTriad
import tqdm

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

class DataCollector:
    def __init__(self, env_name="g1_cartpole", dt=0.001):
        self.env = G1Env(env_name, dt)

        self.state_dim = 16 if env_name == "g1_carpole" else 14
        self.control_dim = 7
        self.reset_joint_state = np.zeros(self.state_dim)
        
        self.effort_lower_limits = self.env.controller_plant.GetEffortLowerLimits()
        self.effort_upper_limits = self.env.controller_plant.GetEffortUpperLimits()

    
    def collect_koopman_data(self,traj_num,steps):
        train_data = np.empty((steps+1,traj_num,self.state_dim+self.control_dim))
        for traj_i in tqdm(range(traj_num)):
            x = self.reset_joint_state
            u = np.random.uniform(self.effort_lower_limits, self.effort_upper_limits)
            train_data[0,traj_i,:]=np.concatenate([u.reshape(-1),x.reshape(-1)],axis=0).reshape(-1)
            for i in range(1,steps+1):
                x_next = self.env.step(x,u)
                u = np.random.uniform(self.effort_lower_limits, self.effort_upper_limits)
                train_data[i,traj_i,:]=np.concatenate([u.reshape(-1),x_next.reshape(-1)],axis=0).reshape(-1)
        return train_data

if __name__ == "__main__":
    env_name = "g1_cartpole"
    traj_num = 100000
    steps = 50
    dt = 0.01

    collector = DataCollector(env_name=env_name, dt=dt)
    dataset = collector.collect_koopman_data(traj_num=traj_num, steps=steps)
    np.save(f"../data/datasets/{env_name}_data/{env_name}_{steps}_step_{traj_num}_traj_{dt}_dtctrl.npy", dataset)
    print(f"Data collection complete. Shape: {dataset.shape}")