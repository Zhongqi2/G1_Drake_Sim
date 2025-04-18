#!/usr/bin/env python3
"""
Balance a cart‑pole bolted to the right wrist of a Unitree G1 arm
via  Koopman latent‑space control  +  LQR on the pole.
Uses an explicit Runge‑Kutta integrator to avoid any
discrete‑QP factorization errors.
"""

from __future__ import annotations
import sys
import numpy as np
import scipy.linalg
import torch

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LinearQuadraticRegulator,
    LoadModelDirectives,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    ProcessModelDirectives,
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
EPS          = 1e-8    # zero‐std → EPS
NAN_GUARD    = True    # clamp NaN/Inf → 0
MAX_ACCEL    = 100.0   # rad/s² limit on LQR output
TORQUE_LIMIT = 50.0    # Nm per joint limit
# ────────────────────────────────────────────────────────────────────────


class G1CartPoleController:
    def __init__(
        self,
        urdf_yaml: str,
        dataset_pt: str,
        control_dt: float = 1e-3,
    ):
        self.dt = float(control_dt)

        # 1) Load + sanitize normalization stats
        data = torch.load(dataset_pt, map_location="cpu", weights_only=False)
        self.state_mean = np.asarray(data["train_state_mean"])
        self.state_std  = np.asarray(data["train_state_std"])
        self.ctrl_mean  = np.asarray(data["train_control_mean"])
        self.ctrl_std   = np.asarray(data["train_control_std"])
        self.state_std[self.state_std < EPS] = EPS
        self.ctrl_std[self.ctrl_std   < EPS] = EPS

        # 2) Build continuous plant + Meshcat
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

        # 3) Precompute small‑angle LQR for the pole
        L = 0.05; g = 9.81
        A_c = np.array([[0, 1], [g/L, 0]])
        B_c = np.array([[0], [-1/L]])
        M = np.block([[A_c, B_c], [np.zeros((1,3))]])
        Md = scipy.linalg.expm(M * self.dt)
        A_d = Md[:2, :2]
        B_d = Md[:2, 2:]
        Q = np.diag([1000.0, 1e-3])
        R = np.eye(1) * 1e-2
        self.K, _ = LinearQuadraticRegulator(A_d, B_d, Q, R)

    def run(
        self,
        koopman_ckpt: str,
        *,
        t_final: float = 10.0,
        realtime_rate: float = 1.0,
    ):
        # A) Load Koopman net + extract A,B
        ckpt = torch.load(koopman_ckpt, map_location="cpu")
        layers = ckpt.get("encode_layers") or ckpt["layer"]
        env_name = "G1CartPole"
        dc = KoopmanDatasetCollector(env_name)

        koopman = KoopmanNet(layers, dc.state_dim + layers[-1], dc.u_dim)
        koopman.load_state_dict(ckpt.get("state_dict", ckpt["model"]))
        koopman.eval()

        A      = koopman.lA.weight.detach().numpy()
        B      = koopman.lB.weight.detach().numpy()
        B_pinv = np.linalg.pinv(B)

        # B) Setup simulator with RK3 integrator
        simulator = Simulator(self.diagram)
        # Replace the integrator with RK3 and max‑step = self.dt
        ResetIntegratorFromFlags(
            simulator=simulator,
            scheme="runge_kutta3",
            max_step_size=self.dt,
        )
        simulator.set_target_realtime_rate(realtime_rate)
        simulator.Initialize()


        ctx       = simulator.get_context()
        plant_ctx = self.diagram.GetMutableSubsystemContext(self.plant, ctx)

        self.meshcat.StartRecording()

        # C) Closed‑loop
        while ctx.get_time() < t_final:
            x = self.plant.get_state_output_port().Eval(plant_ctx)
            x_arm = x[:14]
            theta, theta_dot = float(x[14]), float(x[15])

            # 1) LQR → desired pole accel
            pole_state = np.array([theta, theta_dot])
            a_des = -self.K.dot(pole_state).item()
            if not np.isfinite(a_des): a_des = 0.0
            a_des = float(np.clip(a_des, -MAX_ACCEL, MAX_ACCEL))

            # 2) Encode normalized arm state
            x_norm = (x_arm - self.state_mean) / self.state_std
            with torch.no_grad():
                z = koopman.encode(torch.from_numpy(x_norm).float()).numpy()
            if NAN_GUARD:
                z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

            # 3) Solve for normalized torques
            w_des = np.zeros_like(z); w_des[0] = a_des
            delta = w_des - A.dot(z)
            u_norm = B_pinv.dot(delta)

            # 4) Denormalize + clamp
            u_cmd = u_norm * self.ctrl_std + self.ctrl_mean
            if NAN_GUARD:
                u_cmd = np.nan_to_num(u_cmd, nan=0.0, posinf=0.0, neginf=0.0)
            u_cmd = np.clip(u_cmd, -TORQUE_LIMIT, TORQUE_LIMIT)

            # 5) Apply and step exactly dt
            self.plant.get_actuation_input_port().FixValue(plant_ctx, u_cmd)
            simulator.AdvanceTo(ctx.get_time() + self.dt)

        self.meshcat.PublishRecording()
        print("✅ Simulation complete — open Meshcat to replay")


if __name__ == "__main__":
    controller = G1CartPoleController(
        urdf_yaml  = "models/g1_description/g1_7dof_cartpole.yaml",
        dataset_pt = (
            "koopman/data/datasets/"
            "dataset_G1CartPole_norm_train_60000_val_20000_test_20000_steps_50.pt"
        ),
        control_dt = 1e-3,  # 1 kHz control
    )
    controller.run(
        koopman_ckpt  = (
            "koopman/log/best_models/G1/"
            "best_model_norm_G1CartPole_448_1_1_1.pth"
        ),
        t_final       = 10.0,
        realtime_rate = 10.0,
    )
