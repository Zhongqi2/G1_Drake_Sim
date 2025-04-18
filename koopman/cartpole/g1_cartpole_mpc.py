from pydrake.all import LeafSystem, BasicVector, DiscreteDerivative
import numpy as np
import torch
import osqp
from scipy import sparse
import sys
sys.path.append("../utility")
from network import KoopmanNet
from dataset import KoopmanDatasetCollector

class KoopmanMPCController(LeafSystem):
    def __init__(self, plant, koopman_ckpt, dataset_pt, urdf_yaml):
        super().__init__()
        self.plant = plant
        self.dt = plant.time_step()
        
        # Load Koopman model and normalization parameters
        self._load_koopman_model(koopman_ckpt, dataset_pt)
        
        # MPC parameters
        self.N = 20  # Prediction horizon
        self.Q = sparse.diags([1.0]*self.z_dim)  # State cost
        self.R = sparse.diags([0.1]*self.u_dim)  # Control cost
        self.QN = self.Q  # Terminal cost
        
        # Setup input/output ports
        self.DeclareVectorInputPort("full_state", BasicVector(plant.num_multibody_states()))
        self.DeclareVectorOutputPort("control", BasicVector(self.u_dim), 
                                    self.CalcControlOutput)
        
        # Precompute QP matrices
        self._setup_qp()

    def _load_koopman_model(self, ckpt_path, data_path):
        # Load normalization parameters
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        self.state_mean = np.asarray(data["train_state_mean"])
        self.state_std = np.asarray(data["train_state_std"])
        self.ctrl_mean = np.asarray(data["train_control_mean"])
        self.ctrl_std = np.asarray(data["train_control_std"])
        
        # Load Koopman network
        ckpt = torch.load(ckpt_path, map_location="cpu")
        env_name = "G1CartPole"
        dc = KoopmanDatasetCollector(env_name)
        layers = ckpt.get("encode_layers", ckpt["layer"])
        self.koopman = KoopmanNet(layers, dc.state_dim + layers[-1], dc.u_dim)
        self.koopman.load_state_dict(ckpt.get("state_dict", ckpt["model"]))
        self.koopman.eval()
        
        # Extract linear matrices
        self.A = self.koopman.lA.weight.detach().numpy()
        self.B = self.koopman.lB.weight.detach().numpy()
        self.z_dim = self.A.shape[0]
        self.u_dim = self.B.shape[1]

    def _setup_qp(self):
        # Build prediction matrices with correct horizon alignment
        P = np.zeros((self.N*self.z_dim, self.z_dim))
        H = np.zeros((self.N*self.z_dim, self.N*self.u_dim))
        
        for i in range(self.N):
            # State prediction matrix
            P[i*self.z_dim:(i+1)*self.z_dim] = np.linalg.matrix_power(self.A, i+1)
            
            # Control influence matrix
            for j in range(i+1):
                H[i*self.z_dim:(i+1)*self.z_dim, 
                 j*self.u_dim:(j+1)*self.u_dim] = np.linalg.matrix_power(self.A, i-j) @ self.B

        # Cost matrices with proper terminal cost alignment
        self.Q_bar = sparse.block_diag(
            [self.Q]*(self.N-1) +  # Running cost for N-1 steps
            [self.QN]               # Terminal cost for final step
        )
        R_bar = sparse.block_diag([self.R]*self.N)  # Control cost for N steps

        # QP matrices
        self.H_qp = H.T @ self.Q_bar @ H + R_bar
        self.P = P
        self.H = H

        # OSQP setup with actuator limits
        u_min = (self.plant.GetEffortLowerLimits() - self.ctrl_mean) / self.ctrl_std
        u_max = (self.plant.GetEffortUpperLimits() - self.ctrl_mean) / self.ctrl_std

        self.u_min = np.tile(u_min, self.N)
        self.u_max = np.tile(u_max, self.N)

        # Initialize OSQP problem
        self.prob = osqp.OSQP()
        A_qp = sparse.eye(self.N * self.u_dim)  # Identity matrix for simple box constraints
        self.prob.setup(
            P=sparse.csc_matrix(self.H_qp),
            q=None,
            A=A_qp,
            l=self.u_min,
            u=self.u_max,
            verbose=False,
            polish=True
        )


    def _normalize_state(self, x):
        return (x - self.state_mean) / self.state_std

    def _denormalize_control(self, u):
        return u * self.ctrl_std + self.ctrl_mean

    def CalcControlOutput(self, context, output):
        # Get current state
        x = self.get_input_port(0).Eval(context)
        
        # Normalize and lift to Koopman space
        x_norm = self._normalize_state(x)
        with torch.no_grad():
            z = self.koopman.encode(torch.from_numpy(x_norm).float()).numpy()
        
        # Build QP problem
        q = (self.H.T @ self.Q_bar @ self.P @ z).flatten()
        
        # Solve QP
        self.prob.update(q=q)
        res = self.prob.solve()
        u_opt = res.x[:self.u_dim]  # First control input
        
        # Denormalize and send command
        output.SetFromVector(self._denormalize_control(u_opt))