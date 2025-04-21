import torch
import ipdb
from network import KoopmanNet
from dataset import KoopmanDatasetCollector

def load_koopman_model(pth_path, state_dim, u_dim, device="cpu"):
    checkpoint = torch.load(pth_path, map_location=device)
    print("Checkpoint keys:", checkpoint.keys())
    layers = checkpoint["layer"]

    NKoopman = layers[-1] + state_dim

    u_dim = u_dim

    # Build the model
    net = KoopmanNet(layers, NKoopman, u_dim)
    net.load_state_dict(checkpoint["model"])
    net.eval()
    net.to(device)

    return net

def recover_A_and_B(net, device="cpu"):
    with torch.no_grad():
        A = net.lA.weight.detach().cpu().numpy()
        B = net.lB.weight.detach().cpu().numpy()
    return A, B

def recover_single_control(current_state,
                           target_state,
                           net,
                           device="cpu"):
    if not isinstance(current_state, torch.Tensor):
        current_state = torch.tensor(current_state, dtype=torch.float, device=device)
    if not isinstance(target_state, torch.Tensor):
        target_state = torch.tensor(target_state, dtype=torch.float, device=device)

    with torch.no_grad():
        X_k = net.encode(current_state)
        X_kplus1 = net.encode(target_state)

        A = net.lA.weight
        B = net.lB.weight

        residual = X_kplus1 - A.mv(X_k)
        B_pinv = torch.linalg.pinv(B)
        u = B_pinv.mv(residual)

    return u

def recover_controls_for_trajectory(states,
                                    net,
                                    device="cpu"):
    if not torch.is_tensor(states):
        states_tensor = torch.as_tensor(states, dtype=torch.float, device=device)
    else:
        states_tensor = states.to(device)

    with torch.no_grad():
        X_all = net.encode(states_tensor)
        X_k = X_all[:-1]
        X_kplus1 = X_all[1:]

        A = net.lA.weight
        B = net.lB.weight

        residual = X_kplus1 - (X_k @ A.t())
        B_pinv = torch.linalg.pinv(B)
        controls = residual @ B_pinv.t()

    return controls

def predict_next_state(current_state, control, net, device="cpu"):
    if not isinstance(current_state, torch.Tensor):
        current_state = torch.tensor(current_state, dtype=torch.float, device=device)
    if not isinstance(control, torch.Tensor):
        control = torch.tensor(control, dtype=torch.float, device=device)

    with torch.no_grad():
        X_k = net.encode(current_state)
        X_kplus1 = net.forward(X_k, control)
        Nstate = current_state.shape[-1]
        next_state = X_kplus1[:Nstate]

    return next_state

def predict_next_states(states, controls, net, device="cpu"):
    if not torch.is_tensor(states):
        states_tensor = torch.as_tensor(states, dtype=torch.float, device=device)
    else:
        states_tensor = states.to(device)
    if not torch.is_tensor(controls):
        controls_tensor = torch.as_tensor(controls, dtype=torch.float, device=device)
    else:
        controls_tensor = controls.to(device)

    with torch.no_grad():
        X_k = net.encode(states_tensor)
        X_kplus1 = net.forward(X_k, controls_tensor)
        Nstate = states_tensor.size(-1)
        next_states = X_kplus1[:, :Nstate]

    return next_states

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net = load_koopman_model("model.pth",device)
  A,B = recover_A_and_B(net,device)
  ipdb.set_trace()
