#!/usr/bin/env python3
"""
Balance a cart‑pole on a G1 arm using:
 1) MPC on the linearized pendulum (with CVXPY + OSQP + LQR fallback),
 2) Deep Koopman latent dynamics → arm torques,
 3) Meshcat + RK3 fixed‑step integrator for simulation.
"""

import sys
import numpy as np
import scipy.linalg
import torch
import cvxpy as cp

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LinearQuadraticRegulator,
    LoadModelDirectives,
    Parser,
    ProcessModelDirectives,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Simulator,
    StartMeshcat,
    RungeKutta3Integrator,
)
from pydrake.systems.analysis import ResetIntegratorFromFlags

# project imports
sys.path.append("koopman/utility")
from network import KoopmanNet
from dataset import KoopmanDatasetCollector

# ─── tunables ───────────────────────────────────────────────────────────
EPS          = 1e-8      # zero‐std → EPS
NAN_GUARD    = True      # clamp NaN/Inf → 0
MAX_ACCEL    = 100.0     # rad/s² limit on control accel
TORQUE_LIMIT = 50.0      # Nm per joint limit
MP_HORIZON   = 20        # MPC horizon in steps
# ────────────────────────────────────────────────────────────────────────

class G1CartPoleMPCController:
    def __init__(self, urdf_yaml, dataset_pt, control_dt=1e-3):
        self.dt = float(control_dt)

        # 1) Load + sanitize normalization stats
        data = torch.load(dataset_pt, map_location="cpu", weights_only=False)
        self.state_mean = np.asarray(data["train_state_mean"])
        self.state_std  = np.asarray(data["train_state_std"])
        self.ctrl_mean  = np.asarray(data["train_control_mean"])
        self.ctrl_std   = np.asarray(data["train_control_std"])
        self.state_std[self.state_std < EPS] = EPS
        self.ctrl_std[self.ctrl_std   < EPS] = EPS

        # 2) Build continuous MultibodyPlant + Meshcat
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
        parser = Parser(plant)
        parser.package_map().Add("drake_project", "./")
        directives = LoadModelDirectives(urdf_yaml)
        ProcessModelDirectives(directives, plant, parser)
        plant.Finalize()

        self.meshcat = StartMeshcat()
        MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, self.meshcat, MeshcatVisualizerParams()
        )

        self.plant   = plant
        self.diagram = builder.Build()

        # 3) Precompute discrete pendulum linear model + LQR fallback
        L, g = 0.05, 9.81
        A_c = np.array([[0, 1], [g/L, 0]])
        B_c = np.array([[0], [-1/L]])
        M = np.block([[A_c, B_c], [np.zeros((1,3))]])
        Md = scipy.linalg.expm(M * self.dt)
        self.Ap = Md[:2, :2]
        self.Bp = Md[:2, 2:]

        Q = np.diag([1000.0, 1e-3])
        R = np.eye(1) * 1e-2
        # LQR fallback gain
        self.K_lqr, _ = LinearQuadraticRegulator(self.Ap, self.Bp, Q, R)
        # MPC cost matrices
        self.N_mpc = MP_HORIZON
        self.Q_mpc = Q
        self.R_mpc = R

    def solve_mpc(self, x0):
        """
        Solve finite-horizon MPC; if the QP fails, fallback to LQR acceleration.
        """
        N, A, B = self.N_mpc, self.Ap, self.Bp
        Q, R     = self.Q_mpc, self.R_mpc

        # decision vars
        x = cp.Variable((2, N+1))
        u = cp.Variable((1, N))

        cost = 0
        cons = [x[:,0] == x0]
        for k in range(N):
            cost  += cp.quad_form(x[:,k], Q) + cp.quad_form(u[:,k], R)
            cons  += [x[:,k+1] == A @ x[:,k] + B @ u[:,k]]
            cons  += [cp.abs(u[:,k]) <= MAX_ACCEL]
        cost += cp.quad_form(x[:,N], Q)

        prob = cp.Problem(cp.Minimize(cost), cons)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if prob.status not in ("optimal","optimal_inaccurate"):
                raise cp.SolverError("infeasible")
            a0 = float(u.value[0,0])
        except Exception:
            # fallback to LQR
            a0 = float((-self.K_lqr @ x0).item())
        # final clamp
        return float(np.clip(a0, -MAX_ACCEL, MAX_ACCEL))

    def run(self, koopman_ckpt, t_final=10.0, realtime_rate=1.0):
        # Load KoopmanNet
        ckpt = torch.load(koopman_ckpt, map_location="cpu")
        layers = ckpt.get("encode_layers") or ckpt["layer"]
        env_name = "G1"
        dc = KoopmanDatasetCollector(env_name)

        koopman = KoopmanNet(layers, dc.state_dim + layers[-1], dc.u_dim)
        koopman.load_state_dict(ckpt.get("state_dict", ckpt["model"]))
        koopman.eval()
        A_k = koopman.lA.weight.detach().numpy()
        B_k = koopman.lB.weight.detach().numpy()
        Bpinv = np.linalg.pinv(B_k)

        # Setup simulator with RK3 fixed‑step integrator
        sim = Simulator(self.diagram)
        ResetIntegratorFromFlags(
            simulator=sim,
            scheme="runge_kutta3",
            max_step_size=self.dt
        )
        sim.set_target_realtime_rate(realtime_rate)
        sim.Initialize()

        ctx       = sim.get_context()
        plant_ctx = self.diagram.GetMutableSubsystemContext(self.plant, ctx)
        self.meshcat.StartRecording()

        while ctx.get_time() < t_final:
            x_full = self.plant.get_state_output_port().Eval(plant_ctx)
            x_arm  = x_full[:14]
            th, thd = float(x_full[14]), float(x_full[15])

            # 1) MPC or LQR fallback → desired accel
            a_des = self.solve_mpc(np.array([th, thd]))

            # 2) encode arm state → latent z
            x_norm = (x_arm - self.state_mean) / self.state_std
            with torch.no_grad():
                z = koopman.encode(torch.from_numpy(x_norm).float()).numpy()
            if NAN_GUARD:
                z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

            # 3) latent→normalized torques
            w    = np.zeros_like(z); w[0] = a_des
            delta= w - A_k.dot(z)
            u_nm = Bpinv.dot(delta)

            # 4) denormalize + clamp
            u_cmd = u_nm * self.ctrl_std + self.ctrl_mean
            if NAN_GUARD:
                u_cmd = np.nan_to_num(u_cmd, nan=0.0, posinf=0.0, neginf=0.0)
            u_cmd = np.clip(u_cmd, -TORQUE_LIMIT, TORQUE_LIMIT)

            self.plant.get_actuation_input_port().FixValue(plant_ctx, u_cmd)
            sim.AdvanceTo(ctx.get_time() + self.dt)

        self.meshcat.PublishRecording()
        print("✅ MPC+Koopman simulation complete — open Meshcat to replay")


if __name__ == "__main__":
    controller = G1CartPoleMPCController(
        urdf_yaml=(
            "models/g1_description/g1_7dof_cartpole.yaml"
        ),
        dataset_pt=(
            "koopman/data/datasets/"
            "dataset_G1_norm_train_60000_val_20000_test_20000_steps_50.pt"
        ),
        control_dt=1e-3,
    )
    controller.run(
        koopman_ckpt=(
            "koopman/log/best_models/G1/"
            "best_model_norm_G1_448_1_1_1.pth"
        ),
        t_final=10.0,
        realtime_rate=1.0,
    )
