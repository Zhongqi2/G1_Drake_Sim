# data_collection_cart_pole.py

import numpy as np
import torch
from g1_env_cart_pole import G1Env
from inference import recover_A_and_B, load_koopman_model, recover_single_control
from datetime import datetime

class DataCollector:
    """
    Collect (u, x) sequences from the full system (G1 arm + cart-pole)
    using random torque exploration on the 7-DOF arm.
    """

    def __init__(self):
        # 1) Build the sim+controller environment
        self.env = G1Env()
        self.dt = self.env.time_step

        # 2) Dimensions
        self.Nstates = self.env.plant.num_multibody_states()  # 16 = 8 q + 8 v
        self.nq      = self.env.plant.num_positions()         # 8
        self.udim    = self.env.controller_plant.num_actuators()  # 7

        # 3) Actuation mapping B: generalized_forces = B @ u
        self.B = self.env.plant.MakeActuationMatrix()         # shape (8,7)

        # 4) Reset & limits
        self.reset_joint_state     = np.zeros(self.Nstates)
        self.effort_lower_limits   = self.env.controller_plant.GetEffortLowerLimits()
        self.effort_upper_limits   = self.env.controller_plant.GetEffortUpperLimits()

    def step(self, x, u):
        """
        One forward Euler step of the full dynamics:
            x = [q; v], u in R^7
        """
        plant = self.env.plant
        ctx   = plant.CreateDefaultContext()
        # set sim‐plant state
        plant.SetPositions(ctx,   x[:self.nq])
        plant.SetVelocities(ctx,  x[self.nq:])

        # map actuator torques into generalized forces
        tau_act = self.B.dot(u)            # → R^8

        M   = plant.CalcMassMatrix(ctx)    # 8×8
        Cv  = plant.CalcBiasTerm(ctx)      #  8
        tauG = plant.CalcGravityGeneralizedForces(ctx)  # 8

        qdd = np.linalg.solve(M, tau_act - Cv - tauG)   # → R^8
        x_dot = np.concatenate([x[self.nq:], qdd])     # 16

        return x + self.dt * x_dot

    def collect_koopman_data(self, traj_num, steps):
        """
        Returns: data.shape = (steps+1, traj_num, udim + Nstates)
        Each timestep t row is [u_t, x_t].
        """
        data = np.zeros((steps+1, traj_num, self.udim + self.Nstates))
        for j in range(traj_num):
            x = self.reset_joint_state.copy()
            u = np.random.uniform(self.effort_lower_limits,
                                  self.effort_upper_limits)
            data[0, j, :] = np.hstack([u, x])
            for t in range(1, steps+1):
                x = self.step(x, u)
                u = np.random.uniform(self.effort_lower_limits,
                                      self.effort_upper_limits)
                data[t, j, :] = np.hstack([u, x])
        return data

    def gravity_compensation_test(self, model_path="model.pth"):
        """
        Example: recover gravity‐compensating torques via Koopman model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net    = load_koopman_model(model_path, device)
        A, B   = recover_A_and_B(net, device)

        x = np.concatenate([0.3 * np.ones(self.nq),
                            np.zeros(self.nq)])
        for _ in range(10):
            u      = recover_single_control(
                        torch.from_numpy(x).float(),
                        torch.from_numpy(x).float(),
                        net, device
                     ).cpu().numpy()
            x_next = self.step(x, u)
            print("x_next:", x_next)
            x = np.concatenate([x_next[:self.nq], np.zeros(self.nq)])

if __name__ == "__main__":
    num_samples = 100000
    steps = 50

    collector = DataCollector()
    data = collector.collect_koopman_data(num_samples, steps)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(f"G1CartPole_data_{timestamp}.npy", data)
    print("Saved data shape:", data.shape)