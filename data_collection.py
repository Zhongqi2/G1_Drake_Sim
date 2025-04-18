from pydrake.all import *
import numpy as np
import ipdb
from support_functions import AddShape, AddMultibodyTriad
import matplotlib.pyplot as plt
from g1_env import G1Env
from inference import recover_A_and_B,load_koopman_model,recover_single_control
import torch

class data_collecter():
    def __init__(self,env_name) -> None:
        self.env_name = env_name
        self.env = G1Env()
        self.Nstates = 14
        self.udim = 7
        self.reset_joint_state = np.zeros(self.Nstates)
        self.effort_lower_limits = self.env.controller_plant.GetEffortLowerLimits()
        self.effort_upper_limits = self.env.controller_plant.GetEffortUpperLimits()
        
    def collect_koopman_data(self,traj_num,steps):
        train_data = np.empty((steps+1,traj_num,self.Nstates+self.udim))
        for traj_i in range(traj_num):
            x = self.reset_joint_state
            u = np.random.uniform(self.effort_lower_limits, self.effort_upper_limits)
            train_data[0,traj_i,:]=np.concatenate([u.reshape(-1),x.reshape(-1)],axis=0).reshape(-1)
            for i in range(1,steps+1):
                x_next = self.env.step(x,u)
                u = np.random.uniform(self.effort_lower_limits, self.effort_upper_limits)
                train_data[i,traj_i,:]=np.concatenate([u.reshape(-1),x_next.reshape(-1)],axis=0).reshape(-1)
        return train_data
    
    def gravity_compensation_test(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = load_koopman_model("model.pth",device)
        A,B = recover_A_and_B(net,device)
        x = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0,0,0,0,0,0,0])
        for i in range(10):
            # u = np.zeros(7)
            # x_next = self.env.step(x,u)
            # plant_context = self.env.controller_plant.CreateDefaultContext()
            # plant_context.SetContinuousState(x)
            # tauG = self.env.controller_plant.CalcGravityGeneralizedForces(plant_context)
            u = recover_single_control(x,x,net,device).cpu().numpy()
            x_next = self.env.step(x,u)
            print(x_next)
            x_next[7:] = np.zeros(7)
            x = x_next
        ipdb.set_trace()
        temp = (np.eye(78)- A) @ x
        
        
        
        ipdb.set_trace()
    
if __name__ == "__main__":
    num_samples = 100000
    steps = 50
    collector = data_collecter('g1')  
    # train_data = G1_data_collecter.collect_koopman_data(Ktrain_samples,Ksteps)
    # G1_data_collecter.gravity_compensation_test()
    
    data = collector.collect_koopman_data(num_samples, steps)
    np.save("G1_data.npy", data)
    print("Saved data shape:", data.shape)
    #ipdb.set_trace()