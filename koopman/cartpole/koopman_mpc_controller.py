import os
import sys
import numpy as np
import torch

from pydrake.all import (
    DiagramBuilder,
    Simulator,
    LeafSystem,
    BasicVector,
    AddMultibodyPlantSceneGraph,
    Parser,
    LoadModelDirectives,
    ProcessModelDirectives,
    MeshcatVisualizer,
    StartMeshcat,
    MultibodyPlant,
)
from pydrake.solvers import MathematicalProgram, Solve

# helper to draw triads
sys.path.append("../..")
sys.path.append("../utility")
from support_functions import AddMultibodyTriad

# start Meshcat once
meshcat = StartMeshcat()

def load_koopman(checkpoint_path, state_dim, control_dim):
    """Load KoopmanNet and grab A,B,C,D matrices."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    layers = ckpt.get("encode_layers", ckpt["layer"])
    feature_dim = layers[-1]
    Nkoop = state_dim + feature_dim

    from network import KoopmanNet
    net = KoopmanNet(layers, Nkoop, control_dim)
    net.load_state_dict(ckpt.get("state_dict", ckpt["model"]))
    net.eval()

    A = net.lA.weight.detach().cpu().numpy()
    B = net.lB.weight.detach().cpu().numpy()
    C = np.zeros((state_dim, Nkoop))
    C[:, :state_dim] = np.eye(state_dim)
    D = np.zeros((state_dim, control_dim))
    return net, A, B, C, D

class KoopmanMPCSystem(LeafSystem):
    """Lift state → Koopman space, solve QP to get u₀ towards x_goal."""
    def __init__(self, net, A, B, C, Q, R, Qf,
                 u_lower, u_upper, horizon, x_goal):
        super().__init__()
        self.net      = net
        self.A, self.B= A, B
        self.C        = C
        self.Q, self.R, self.Qf = Q, R, Qf
        self.u_lower, self.u_upper = u_lower, u_upper
        self.horizon  = horizon
        self.x_goal   = x_goal

        self.nkoop    = A.shape[0]
        self.nu       = B.shape[1]
        self.nx       = C.shape[0]

        # input = current state x ∈ ℝⁿˣ
        self.DeclareVectorInputPort("x", BasicVector(self.nx))
        # output = torque u ∈ ℝⁿᵘ
        self.DeclareVectorOutputPort("u", BasicVector(self.nu), self.CalcControl)

    def CalcControl(self, context, output):
        x0 = self.get_input_port(0).Eval(context)
        # lift: z0 = [x0; ψ(x0)]
        z0 = self.net.encode(torch.from_numpy(x0[None,:]).float()) \
                    .detach().cpu().numpy().ravel()

        prog = MathematicalProgram()
        Z = prog.NewContinuousVariables(self.nkoop, self.horizon+1, "Z")
        U = prog.NewContinuousVariables(self.nu,    self.horizon,   "U")

        # initial cond
        prog.AddLinearEqualityConstraint(np.eye(self.nkoop), z0, Z[:,0])

        # linear Koopman dynamics
        Mdyn = np.hstack([self.A, self.B, -np.eye(self.nkoop)])
        for k in range(self.horizon):
            prog.AddLinearEqualityConstraint(
                Mdyn, np.zeros(self.nkoop),
                np.hstack([Z[:,k], U[:,k], Z[:,k+1]])
            )

        # running cost: track x_goal
        for k in range(self.horizon):
            x_pred = self.C @ Z[:,k]
            err    = x_pred - self.x_goal
            prog.AddQuadraticCost(err.T @ self.Q  @ err
                                 + U[:,k].T @ self.R @ U[:,k])

        # terminal cost
        xN = self.C @ Z[:,self.horizon]
        errN = xN - self.x_goal
        prog.AddQuadraticCost(errN.T @ self.Qf @ errN)

        # input bounds
        for k in range(self.horizon):
            prog.AddBoundingBoxConstraint(self.u_lower, self.u_upper, U[:,k])

        sol = Solve(prog)
        u0 = sol.GetSolution(U[:,0])
        output.SetFromVector(u0)

def main():
    # user‐tweakable parameters
    dt          = 0.01
    horizon     = 50
    checkpoint  = "../log/best_models/G1/best_model_unnorm_G1CartPole_512_1_1_1.pth"
    model_yaml  = "models/g1_description/g1_7dof_cartpole.yaml"

    # cost on (q,v) and effort
    state_dim   = 16
    control_dim = 7
    Q   = np.diag([100.0]*state_dim)
    R   = np.eye(control_dim)*0.1
    Qf  = np.diag([500.0]*state_dim)

    # desired state: robot joints zero; pendulum upright => angle = π
    x_goal = np.zeros(state_dim)
    pend_index = 7     # 0–6 are arm joints; index 7 is the pole angle
    x_goal[pend_index] = np.pi

    # 1) plant + scene_graph
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)

    parser = Parser(plant)
    parser.package_map().Add("drake_project", "./")
    directives = LoadModelDirectives(model_yaml)
    ProcessModelDirectives(directives, plant, parser)
    plant.Finalize()

    # 2) extract effort limits from a zero‐lag plant
    ctrl_plant = MultibodyPlant(0.0)
    p2 = Parser(ctrl_plant)
    p2.package_map().Add("drake_project", "./")
    ProcessModelDirectives(directives, ctrl_plant, p2)
    ctrl_plant.Finalize()
    effort_lower = ctrl_plant.GetEffortLowerLimits()
    effort_upper = ctrl_plant.GetEffortUpperLimits()

    # 3) visuals
    G1_inst = plant.GetModelInstanceByName("G1")
    pelvis  = plant.GetFrameByName("pelvis", G1_inst)
    AddMultibodyTriad(pelvis, scene_graph)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    # 4) Koopman load
    net, A, B, C, _ = load_koopman(checkpoint, state_dim, control_dim)

    # 5) add MPC system
    mpc = builder.AddSystem(
        KoopmanMPCSystem(
            net, A, B, C,
            Q, R, Qf,
            effort_lower, effort_upper,
            horizon,
            x_goal
        )
    )
    mpc.set_name("koopman_mpc")

    # 6) connect: plant state → MPC → plant actuation
    builder.Connect(plant.get_state_output_port(), mpc.get_input_port(0))
    builder.Connect(mpc.get_output_port(0),       plant.get_actuation_input_port())

    # 7) build & run
    diagram = builder.Build()
    sim     = Simulator(diagram)
    sim.set_target_realtime_rate(1.0)
    sim.AdvanceTo(10.0)

    print(f"Done. View in Meshcat: http://localhost:{meshcat.web_port}")

if __name__ == "__main__":
    main()
