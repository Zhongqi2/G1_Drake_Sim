import numpy as np
from g1_env import G1Env
from tqdm import tqdm

class DataCollector:
    def __init__(self, env_name="g1", dt=0.001):
        self.env = G1Env(env_name, dt)

        self.state_dim = 14 if env_name == "g1" else 16
        self.control_dim = 7
        self.reset_joint_state = np.zeros(self.state_dim)
        
        # Get torque limits for arm joints
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