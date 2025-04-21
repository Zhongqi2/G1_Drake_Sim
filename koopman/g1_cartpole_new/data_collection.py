import numpy as np
from g1_cartpole_env import G1Env
from tqdm import tqdm

class DataCollector:
    def __init__(self, env_name="g1", dt=0.001, collision_free=True):
        self.env = G1Env(env_name, dt, collision_free)

        self.state_dim = 14 if env_name == "g1" else 16
        self.control_dim = 7
        self.reset_joint_state = np.zeros(self.state_dim)
        self.reset_joint_state[7] = 3.14

        self.pos_lower = np.array(self.env.joint_lower_limits)
        self.pos_upper = np.array(self.env.joint_upper_limits)
        self.vel_lower = self.env.controller_plant.GetVelocityLowerLimits()
        self.vel_upper = self.env.controller_plant.GetVelocityUpperLimits()
        self.effort_lower = self.env.effort_lower_limits
        self.effort_upper = self.env.effort_upper_limits

        # print the limits
        print("Position Lower Limits:", self.pos_lower)
        print("Position Upper Limits:", self.pos_upper)
        print("Velocity Lower Limits:", self.vel_lower)
        print("Velocity Upper Limits:", self.vel_upper)
        print("Effort Lower Limits:", self.effort_lower)
        print("Effort Upper Limits:", self.effort_upper)
    
    def collect_data(self, traj_num, steps):
        data = np.empty((steps, traj_num, self.state_dim + self.control_dim))

        for traj_idx in tqdm(range(traj_num)):
            success = False

            while not success:
                traj_data = np.empty((steps, 1, self.state_dim + self.control_dim))

                x = np.zeros(self.state_dim)
                x[:8] = np.random.uniform(self.pos_lower, self.pos_upper)
                x[8:16] = np.random.uniform(self.vel_lower, self.vel_upper)
                u = np.random.uniform(self.effort_lower, self.effort_upper)

                for step_idx in range(steps):
                    traj_data[step_idx, 0, :self.control_dim] = u
                    traj_data[step_idx, 0, self.control_dim:] = x

                    u = np.random.uniform(self.effort_lower, self.effort_upper)
                    x = self.env.step(x, u)

                    if (np.any(x[:8] < self.pos_lower) or 
                        np.any(x[:8] > self.pos_upper) or 
                        np.any(x[8:16] < self.vel_lower) or 
                        np.any(x[8:16] > self.vel_upper)):
                        break

                    if step_idx == steps - 1:
                        traj_data[step_idx, 0, :self.control_dim] = u
                        traj_data[step_idx, 0, self.control_dim:] = x
                        success = True
                        data[:, traj_idx, :] = traj_data[:, 0, :]

        return data


if __name__ == "__main__":
    env_name = "g1_cartpole"
    train_rollout_steps = 20
    train_traj_num = 1000
    dt = 0.001

    collector = DataCollector(env_name=env_name, dt=dt, collision_free=True)
    dataset = collector.collect_data(traj_num=train_traj_num, steps=train_rollout_steps)
    
    np.save(f"../data/datasets/{env_name}_data/{env_name}_{train_rollout_steps}_step_{train_traj_num}_traj_{dt}_dtctrl.npy", dataset)
    print(f"Data collection complete. Shape: {dataset.shape}")