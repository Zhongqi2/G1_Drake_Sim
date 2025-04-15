import torch
import ipdb
from network import KoopmanNet
from dataset import KoopmanDatasetCollector

def load_koopman_model(pth_path, device="cpu"):
    """
    Loads the saved Koopman model (state dict + layer info) from a .pth file.
    Returns a reconstructed `Network` instance on the specified device.
    """
    checkpoint = torch.load(pth_path, map_location=device)
    print("Checkpoint keys:", checkpoint.keys())
    layers = checkpoint["layer"]  # [in_dim, ..., encode_dim]

    # Calculate dimension of the Koopman space: 
    data_collector = KoopmanDatasetCollector("Kinova")
    NKoopman = layers[-1] + data_collector.state_dim

    u_dim = data_collector.u_dim

    # Build the model
    net =KoopmanNet(layers, NKoopman, u_dim)
    net.load_state_dict(checkpoint["model"])
    net.eval()
    net.to(device)

    return net

def recover_A_and_B(net, device="cpu"):
    with torch.no_grad():
      # A and B are the weight matrices from net.lA and net.lB
      A = net.lA.weight.detach().cpu().numpy()  # shape [Nkoopman, Nkoopman]
      B = net.lB.weight.detach().cpu().numpy()  # shape [Nkoopman, u_dim]
    return A,B


def recover_single_control(current_state,
                           target_state,
                           net,
                           device="cpu"):
    """
    Given a Koopman `net`, and the pair of states (current_state, target_state),
    returns a single control u that should move `current_state` to `target_state` in one step,
    according to the Koopman model.
    
    Inputs:
      - current_state: np.array or torch.Tensor, shape [Nstate]
      - target_state:  np.array or torch.Tensor, shape [Nstate]
      - net:           the loaded Koopman Network (with lA, lB, and encode)
      - device:        "cpu" or "cuda"

    Returns:
      - u: torch.Tensor of shape [u_dim], the control input
    """
    # Convert states to torch Tensors if they're not already
    if not isinstance(current_state, torch.Tensor):
        current_state = torch.tensor(current_state, dtype=torch.float, device=device)
    if not isinstance(target_state, torch.Tensor):
        target_state = torch.tensor(target_state, dtype=torch.float, device=device)

    # Encode into Koopman space
    with torch.no_grad():
        X_k = net.encode(current_state)     # shape [Nkoopman]
        X_kplus1 = net.encode(target_state) # shape [Nkoopman]

        # A and B are the weight matrices from net.lA and net.lB
        A = net.lA.weight  # shape [Nkoopman, Nkoopman]
        B = net.lB.weight  # shape [Nkoopman, u_dim]

        # residual = X_{k+1} - A @ X_k
        residual = X_kplus1 - A.mv(X_k)

        # Solve for u via pseudo-inverse of B
        B_pinv = torch.linalg.pinv(B)       # shape [u_dim, Nkoopman]
        u = B_pinv.mv(residual)            # shape [u_dim]

    return u

def recover_controls_for_trajectory(states,
                                    net,
                                    device="cpu"):
    """
    If you have a sequence of states [x_0, x_1, ..., x_T], this function
    returns the list of control inputs [u_0, u_1, ..., u_{T-1}] that drive
    x_k -> x_{k+1} at each step, according to the Koopman model.

    Inputs:
      - states: list (or np.array) of shape [T+1, Nstate]
      - net:    the loaded Koopman Network
      - device: "cpu" or "cuda"

    Returns:
      - controls: list of shape [T, u_dim]
    """
    if not torch.is_tensor(states):
        states_tensor = torch.as_tensor(states, dtype=torch.float, device=device)
    else:
        states_tensor = states.to(device)
    
    X_all = net.encode(states_tensor)

    X_k = X_all[:-1]    # shape: [T, Nkoopman]
    X_kplus1 = X_all[1:]  # shape: [T, Nkoopman]

    A = net.lA.weight  # shape: [Nkoopman, Nkoopman]
    B = net.lB.weight  # shape: [Nkoopman, u_dim]

    residual = X_kplus1 - (X_k @ A.t())  # shape: [T, Nkoopman]
    
    B_pinv = torch.linalg.pinv(B)  # shape: [u_dim, Nkoopman]
    
    controls = residual @ B_pinv.t()  # shape: [T, u_dim]
    
    return controls

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net = load_koopman_model("model.pth",device)
  A,B = recover_A_and_B(net,device)
  ipdb.set_trace()
