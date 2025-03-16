from pydrake.all import *
import numpy as np
import ipdb
from support_functions import AddShape, AddMultibodyTriad
import matplotlib.pyplot as plt
from g1_env import G1Env

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
                x_d = self.env.step(x,u)
                u = np.random.uniform(self.effort_lower_limits, self.effort_upper_limits)
                train_data[i,traj_i,:]=np.concatenate([u.reshape(-1),x_d.reshape(-1)],axis=0).reshape(-1)

        return train_data

if __name__ == "__main__":
    Ktrain_samples = 1000
    Ksteps = 10
    G1_data_collecter = data_collecter('g1')  
    train_data = G1_data_collecter.collect_koopman_data(Ktrain_samples,Ksteps)
    ipdb.set_trace()