import numpy as np
from g1_env import G1Env

class DataCollector:
    def __init__(self):
        self.env = G1Env()
        self.state_dim = 16  # 8 positions + 8 velocities
        #self.state_dim = 14
        self.control_dim = 7
        self.reset_joint_state = np.zeros(self.state_dim)
        
        # Get torque limits for arm joints
        self.effort_lower_limits = self.env.controller_plant.GetEffortLowerLimits()
        self.effort_upper_limits = self.env.controller_plant.GetEffortUpperLimits()

        # self.torque_low = np.array([-100]*7)  # Adjust based on actual robot limits
        # self.torque_high = np.array([100]*7)
        
    # def collect_data(self, num_episodes=1000, steps_per_episode=50):
    #     data = np.zeros((num_episodes, steps_per_episode, self.state_dim + self.control_dim))
        
    #     for ep in range(num_episodes):
    #         # Random initial state (arm joints + pole angle/velocity)
    #         state = np.concatenate([
    #             [np.random.uniform(low, high) for low, high in self.env.joint_limits],
    #             #np.random.uniform(-0.1, 0.1, 1),  # Pole angle
    #             np.zeros(7),  # Arm velocities
    #             #np.random.uniform(-0.5, 0.5, 1)   # Pole velocity
    #         ])
            
    #         for step in range(steps_per_episode):
    #             # Random control input
    #             control = np.random.uniform(self.torque_low, self.torque_high)
                
    #             # Store transition
    #             data[ep, step, :self.control_dim] = control
    #             data[ep, step, self.control_dim:] = state
                
    #             # Step environment
    #             state = self.env.step(state, control)
        
    #     return data
    
    def collect_koopman_data(self,traj_num,steps):
        train_data = np.empty((steps+1,traj_num,self.state_dim+self.control_dim))
        for traj_i in range(traj_num):
            x = self.reset_joint_state
            u = np.random.uniform(self.effort_lower_limits, self.effort_upper_limits)
            train_data[0,traj_i,:]=np.concatenate([u.reshape(-1),x.reshape(-1)],axis=0).reshape(-1)
            for i in range(1,steps+1):
                x_next = self.env.step(x,u)
                u = np.random.uniform(self.effort_lower_limits, self.effort_upper_limits)
                train_data[i,traj_i,:]=np.concatenate([u.reshape(-1),x_next.reshape(-1)],axis=0).reshape(-1)
        return train_data

if __name__ == "__main__":
    collector = DataCollector()
    dataset = collector.collect_koopman_data(traj_num=100000, steps=50)
    np.save("../data/datasets/g1_cartpole_data/g1_cartpole_50_step_100000_traj_0.0001_dtctrl.npy", dataset)
    print(f"Data collection complete. Shape: {dataset.shape}")