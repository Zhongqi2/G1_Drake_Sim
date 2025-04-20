import cvxpy as cp
import numpy as np
import torch
from tqdm import tqdm
import sys
sys.path.append("../utility")
from network import KoopmanNet
from g1_env import G1Env

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

def koopman_mpc(A, B, z0, z_ref_seq, H, Q, R):
    n = A.shape[0]
    m = B.shape[1]
    
    z = cp.Variable((H + 1, n))
    u = cp.Variable((H, m))
    cost = 0
    constraints = [z[0] == z0]

    for t in range(H):
        cost += cp.quad_form(z[t] - z_ref_seq[t], Q)
        cost += cp.quad_form(u[t], R)
        constraints += [
            z[t+1] == A @ z[t] + B @ u[t],
        ]
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)
    return u.value

def main():
    checkpoint_path = "../log/best_models/G1/best_model_unnorm_G1CartPole_128_1_1_1.pth"
    net, A, B = load_koopman(checkpoint_path)

    num_iterations = 1000
    dt = 0.01
    H = 10
    q_scale = 1
    r_scale = 1
    Q = np.eye(A.shape[0]) * q_scale
    R = np.eye(B.shape[1]) * r_scale

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

        u_seq = koopman_mpc(A, B, z0, z_ref_seq, H, Q, R)
        u0 = u_seq[0]

        x = env.step(x, u0)

        print(f"[Step {i}] Control: {u0}, State: {x}")

        env.publish_visualization(x)


if __name__ == "__main__":
    main()