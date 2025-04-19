import torch
import numpy as np
import torch.nn as nn
import random
import copy
import itertools
import wandb
from torch.utils.data import DataLoader
import os
import sys
sys.path.append('../utility')
from dataset import KoopmanDatasetCollector, KoopmanDataset
from network import KoopmanNet

def get_layers(input_dim, layer_depth):
    return [input_dim * (2 ** i) for i in range(layer_depth + 2)]

def koopman_rollout_loss(data, net, mse_loss, u_dim, gamma, device, control_loss_weight=0.0):
    steps, traj_num, N = data.shape
    state_dim = N - u_dim

    X_current = net.encode(data[0, :, u_dim:])
    initial_encoding = X_current
    beta = 1.0
    beta_sum = 0.0
    state_loss = torch.zeros(1, dtype=torch.float32).to(device)
    control_loss = torch.zeros(1, dtype=torch.float32).to(device)

    for i in range(steps-1):
        X_current = net.forward(X_current, data[i, :, :u_dim])
        beta_sum += beta
        state_loss += beta * mse_loss(X_current[:, :state_dim], data[i+1, :, u_dim:])
        if control_loss_weight > 0:
            X_i = net.encode(data[i, :, u_dim:])
            X_ip1 = net.encode(data[i+1, :, u_dim:])
            A = net.lA.weight
            B = net.lB.weight
            residual = X_ip1 - (X_i @ A.t())
            B_pinv = torch.linalg.pinv(B)
            u_rec = residual @ B_pinv.t()
            control_loss += beta * mse_loss(u_rec, data[i, :, :u_dim])
        beta *= gamma

    state_loss = state_loss / beta_sum

    if control_loss_weight > 0:
        control_loss = control_loss / beta_sum
        total_loss = state_loss + control_loss_weight * control_loss
        return total_loss, state_loss, control_loss, initial_encoding
    else:
        return state_loss, initial_encoding


def cov_loss(z):
    z_mean = torch.mean(z, dim=0, keepdim=True)
    z_centered = z - z_mean
    cov_matrix = (z_centered.t() @ z_centered) / (z_centered.size(0) - 1)
    diag_cov = torch.diag(torch.diag(cov_matrix))
    off_diag = cov_matrix - diag_cov
    return torch.norm(off_diag, p='fro')**2

def train(project_name, env_name, train_samples=60000, val_samples=20000, test_samples=20000, steps=15,
          train_steps=20000, hidden_layers=2, cov_reg=0, gamma=0.99, seed=42, batch_size=64, control_loss_weight=0,
          initial_lr=1e-3, lr_step=1000, lr_gamma=0.95, val_step=1000, max_norm=1, cov_reg_weight=1, normalize=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    norm_str = "norm" if normalize else "unnorm"

    if not os.path.exists(f"../log/best_models/{project_name}/"):
        os.makedirs(f"../log/best_models/{project_name}/")

    print("Loading dataset...")

    data_collector = KoopmanDatasetCollector(env_name, train_samples, val_samples, test_samples, steps, normalize=normalize)
    train_data, val_data, _ = data_collector.get_data()

    train_data = torch.from_numpy(train_data).float()
    val_data = torch.from_numpy(val_data).float()

    u_dim = data_collector.u_dim
    state_dim = data_collector.state_dim

    print("u_dim:", u_dim)
    print("state_dim:", state_dim)

    print("Train data shape:", train_data.shape)
    print("Validation data shape:", val_data.shape)

    layers = get_layers(state_dim, hidden_layers)
    Nkoopman = state_dim + layers[-1]

    print("Encoder layers:", layers)

    net = KoopmanNet(layers, Nkoopman, u_dim)
    net.to(device)
    mse_loss = nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    wandb.init(project=project_name, 
               name=f"{env_name}_edim{layers[-1]}_closs{'on' if cov_reg else 'off'}_seed{seed}",
               config={
                    "env_name": env_name,
                    "train_steps": train_steps,
                    "encode_dim": layers[-1],
                    "hidden_layers": hidden_layers,
                    "c_loss": cov_reg,
                    "gamma": gamma,
                    "train_samples": train_data.shape[1],
                    "val_samples": val_data.shape[1],
                    "steps": train_data.shape[0],
                    "seed": seed,
                    "initial_lr": initial_lr,
                    "lr_step": lr_step,
                    "lr_gamma": lr_gamma,
                    "batch_size": batch_size,
                    "max_norm": max_norm,
                    "cov_reg_weight": cov_reg_weight,
               })

    best_state_loss = 1e10
    step = 0
    val_losses = []

    train_dataset = KoopmanDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_data = val_data.to(device)

    while step < train_steps:
        for batch in train_loader:
            if step >= train_steps:
                break

            X = batch.permute(1, 0, 2).to(device)
            if step % 100 == 0 and control_loss_weight > 0:
                total_loss, state_loss, ctrl_loss, initial_encoding = koopman_rollout_loss(
                    X, net, mse_loss, u_dim, gamma, device, control_loss_weight=control_loss_weight
                )
            else:
                state_loss, initial_encoding = koopman_rollout_loss(
                    X, net, mse_loss, u_dim, gamma, device, control_loss_weight=0.0
                )
                total_loss = state_loss



            Closs = cov_loss(initial_encoding[:, state_dim:])

            if cov_reg:
                factor = initial_encoding[:, state_dim:].shape[1] * (initial_encoding[:, state_dim:].shape[1] - 1)
                loss = total_loss + cov_reg_weight * Closs / factor
            else:
                loss = total_loss


            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), max_norm)

            optimizer.step()
            scheduler.step()

            wandb.log({
                "Train/StateLoss": state_loss.item(),
                "Train/CovLoss": Closs.item(),
                "step": step
            })

            if step % val_step == 0:
                with torch.no_grad():
                    if control_loss_weight > 0:
                        total_loss_val, state_loss_val, ctrl_loss_val, initial_encoding = koopman_rollout_loss(
                            val_data, net, mse_loss, u_dim, gamma, device, control_loss_weight=control_loss_weight
                        )
                    else:
                        state_loss_val, initial_encoding = koopman_rollout_loss(
                            val_data, net, mse_loss, u_dim, gamma, device, control_loss_weight=0.0
                        )
                        ctrl_loss_val = torch.zeros(1)

                    Closs_val = cov_loss(initial_encoding[:, state_dim:])
                    val_losses.append(state_loss_val.item())
                    if state_loss_val < best_state_loss:
                        best_state_loss = copy.copy(state_loss_val)
                        best_state_dict = copy.copy(net.state_dict())
                        saved_dict = {'model': best_state_dict, 'layer': layers}
                        torch.save(saved_dict, f"../log/best_models/{project_name}/best_model_{norm_str}_{env_name}_{layers[-1]}_{cov_reg}_{control_loss_weight}_{seed}.pth")

                    wandb.log({
                        "Val/StateLoss": state_loss_val.item(),
                        "Val/CtrlLoss": ctrl_loss_val.item(),
                        "Val/CovLoss": Closs_val.item(),
                        "Val/best_StateLoss": best_state_loss.item(),
                        "step": step,
                    })
                    print("Step:{} Validation State Loss:{}".format(step, state_loss_val.item()))


            step += 1

    if len(val_losses) >= 10:
        convergence_loss = np.mean(val_losses[-10:])
    else:
        convergence_loss = np.mean(val_losses) if len(val_losses) > 0 else None

    print("END - Best State loss: {}  Convergence loss: {}".format(best_state_loss, convergence_loss))
    wandb.log({"best_state_loss": best_state_loss, "convergence_state_loss": convergence_loss})
    wandb.finish()

def main():
    train(project_name=f'G1',
            env_name='G1',
            train_samples=80000,
            val_samples=10000,
            test_samples=10000,
            steps=50,
            train_steps=50000,
            hidden_layers=4,
            cov_reg=1,
            gamma=0.8,
            seed=1,
            batch_size=64,
            val_step=1000,
            initial_lr=1e-3,
            lr_step=1000,
            lr_gamma=0.9,
            max_norm=0.01,
            cov_reg_weight=1,
            control_loss_weight=1,
            normalize=True,
            )

if __name__ == "__main__":
    main()