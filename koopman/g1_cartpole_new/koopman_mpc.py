import cvxpy as cp
import numpy as np
import torch
from tqdm import tqdm
import imageio
import sys
sys.path.append("../utility")
from network import KoopmanNet
from g1_cartpole_env import G1Env

def load_koopman(checkpoint_path, state_dim=16, control_dim=7):
    ckpt = torch.load(checkpoint_path)
    layers = ckpt.get("encode_layers", ckpt["layer"])
    feature_dim = layers[-1]
    Nkoop = state_dim + feature_dim

    net = KoopmanNet(layers, Nkoop, control_dim)
    net.load_state_dict(ckpt.get("state_dict", ckpt["model"]))
    net.eval()

    A = net.lA.weight.detach().numpy()
    B = net.lB.weight.detach().numpy()

    return net, A, B

def koopman_mpc(A, B, z0, z_ref_seq, H, Q, R, joint_lower_limits, joint_upper_limits, effort_lower_limits, effort_upper_limits):
    n = A.shape[0]
    m = B.shape[1]
    
    z = cp.Variable((H + 1, n))
    u = cp.Variable((H, m))
    cost = 0
    constraints = [z[0][:16] == z0[:16]]

    for t in range(H):
        cost += cp.quad_form(z[t] - z_ref_seq[t], Q)
        cost += cp.quad_form(u[t], R)
        cost += cp.quad_form(z[H] - z_ref_seq[-1], Q)
        constraints += [
            z[t+1] == A @ z[t] + B @ u[t],
            joint_lower_limits <= u[t][:7],
            u[t][:7] <= joint_upper_limits,
            effort_lower_limits <= u[t],
            u[t] <= effort_upper_limits,
        ]
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(
        solver=cp.OSQP,
        warm_start=True,
        eps_abs=1e-4,
        eps_rel=1e-4,
        max_iter=10000
    )
    return u.value

def main():
    checkpoint_path = "../log/best_models/G1/best_model_G1CartPole.pth"
    net, A, B = load_koopman(checkpoint_path)

    num_iterations = 1000
    dt = 0.001
    H = 20

    orig_dim = 16
    feat_dim = A.shape[0] - orig_dim
    q_weights = np.concatenate([
        np.ones(orig_dim)*1,
        np.ones(feat_dim)*0.01
    ])
    Q = np.diag(q_weights)
    Q[7,7] = 100

    col_norms = np.linalg.norm(B, axis=0)
    R_diag = (col_norms.max() / col_norms)
    R = np.diag(R_diag)

    

    env = G1Env(time_step=dt)
    
    initial_x = torch.zeros(16)
    initial_x[7] = np.pi-0.1
    target_x = torch.zeros(16)
    target_x[7] = np.pi

    x = initial_x.clone()

    for i in tqdm(range(num_iterations)):
        x = x.float()
        z0 = net.encode(x).detach().numpy()
        z_ref_seq = np.tile(net.encode(target_x).detach().numpy(), (H, 1))

        u_seq = koopman_mpc(A, B, z0, z_ref_seq, H, Q, R, 
                            env.joint_lower_limits, env.joint_upper_limits,
                            env.effort_lower_limits, env.effort_upper_limits)
        u0 = u_seq[0]

        x = env.step(x, u0)
        env.publish_visualization(x)

        theta     = x[7].item()
        theta_dot = x[15].item()

        # check “balanced upright”
        # angle_ok = abs(theta - np.pi) < 0.01
        # rate_ok  = abs(theta_dot)    < 0.5

        # if angle_ok and rate_ok:
        #     print(f"Balanced upright at step {i} (θ error = {theta-np.pi:.3f},  ω = {theta_dot:.3f})")
        #     break

        print(f"[Step {i}] Control: {u0}, State: {x}")

if __name__ == "__main__":
    main()