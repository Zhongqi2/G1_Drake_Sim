import numpy as np
import torch
from g1_cartpole_env import G1Env
from datetime import datetime
from pydrake.all import Simulator
import gc
import argparse
import ipdb
class DataCollector:
    def __init__(self):
        # 1) Build environment and create ONE simulator instance
        self.env = G1Env()
        self.dt = self.env.time_step
        self.plant = self.env.plant
        
        # System dimensions
        self.Nstates = self.plant.num_multibody_states()  # 18
        self.nq = self.plant.num_positions()              # 9
        self.udim = self.plant.num_actuators()            # 7
        
        # Actuation matrix and limits
        self.B = self.plant.MakeActuationMatrix()
        self.effort_limits = (
            self.plant.GetEffortLowerLimits(),
            self.plant.GetEffortUpperLimits()
        )
        
        # Create ONE simulator instance to reuse
        self.sim = Simulator(self.env.diagram)

    def collect_koopman_data(self, num_trajectories, steps_per_trajectory):
        data = np.zeros((num_trajectories, steps_per_trajectory+1, 
                        self.udim + self.Nstates))
        
        # Main simulation loop
        for traj_idx in range(num_trajectories):
            if traj_idx % 1000 == 0:
                print(f"Trajectory {traj_idx}/{num_trajectories}")
            
            # 1) Create FRESH context for this trajectory
            root_context = self.env.diagram.CreateDefaultContext()
            plant_context = self.env.diagram.GetMutableSubsystemContext(
                self.plant, root_context)
            
            # 2) Initialize random state
            q_init = self._random_initial_position()

            self.plant.SetPositions(plant_context, q_init)
            self.plant.SetVelocities(plant_context, np.zeros(self.nq))
            
            # 3) Configure simulator with the new context
            self.sim.reset_context(root_context)
            self.sim.Initialize()
            
            # 4) Store initial state
            x = np.concatenate([q_init, np.zeros(self.nq)])
            data[traj_idx, 0, :] = np.hstack([np.zeros(self.udim), x])
            
            current_time = 0.0  # Track simulation time
            # self.env.meshcat.StartRecording(set_visualizations_while_recording=True)
            # Step through trajectory
            for step in range(1, steps_per_trajectory+1):
                # Random torque
                u = np.random.uniform(*self.effort_limits)
                self.plant.SetPositions(plant_context, x[:8])
                self.plant.SetVelocities(plant_context, x[8:])
                M = self.plant.CalcMassMatrix(plant_context)
                Cv = self.plant.CalcBiasTerm(plant_context)
                tauG = self.plant.CalcGravityGeneralizedForces(plant_context)
                B = self.plant.MakeActuationMatrix()
                q_ddot = np.linalg.inv(M) @ (B @ u + tauG - Cv)  # Compute acceleration q̈
                x_dot = np.concatenate((x[8:],q_ddot))
                x = x + x_dot * self.dt
                data[traj_idx, step, :] = np.hstack([u, x])

                # # Apply torque to the SAME context
                torque_context = self.env.diagram.GetMutableSubsystemContext(
                    self.env.torque_input, root_context)
                # self.env.torque_port.FixValue(torque_context, u)
                
                # # Advance simulation by ONE STEP
                # self.sim.AdvanceTo(current_time + self.dt)
                # current_time += self.dt
                
                # # Get updated state
                # x = np.concatenate([
                #     self.plant.GetPositions(plant_context),
                #     self.plant.GetVelocities(plant_context)
                # ])
                # data[traj_idx, step, :] = np.hstack([u, x])
                
            # self.env.meshcat.PublishRecording()
            # ipdb.set_trace()
            # 5) Force cleanup of trajectory-specific resources
            del root_context, plant_context, torque_context
            gc.collect()
        
        return data.transpose(1, 0, 2)

    def _random_initial_position(self):
        """Safer initialization with proper joint limits"""
        q = np.zeros(self.nq)
        # Arm joints (first 7)
        q[:7] = np.random.uniform(
            self.env.joint_lower_limits[:7],  # Ensure matching dimensions
            self.env.joint_upper_limits[:7]
        )
        # Cart-pole (positions 7-8)
        # q[7] = 0.0  # Cart position
        q[7] = np.pi + np.random.normal(0, 0.05)  # Pole angle
        return q

    def gravity_compensation_test(self, model_path="model.pth"):
        """Updated gravity compensation test (if needed)"""
        # Implementation would need to match your Koopman network architecture
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect G1 CartPole data')
    parser.add_argument('--num_trajectories', type=int, default=800,
                       help='Number of trajectories per batch')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file name')
    args = parser.parse_args()

    collector = DataCollector()
    print(f"Starting batch collection for {args.num_trajectories} trajectories...")
    data = collector.collect_koopman_data(args.num_trajectories, 50)
    
    # Save with incremental numbering
    np.save(args.output, data)
    print(f"Saved batch data to {args.output}")