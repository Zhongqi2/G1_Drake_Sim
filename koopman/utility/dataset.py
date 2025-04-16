import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset

class KinovaDataCollecter():
    def __init__(self):
        self.state_dim = 14
        self.u_dim = 7
        self.data_pathes = ['output_20250402_172619.txt',
                            'output_20250402_182836.txt',
                            'output_20250402_195709.txt',
                            'output_20250402_205831.txt',
                            'output_20250403_104412.txt']
    
    def _low_pass_filter(self, data, window_size=3):
        kernel = np.ones(window_size) / window_size
        filtered = np.zeros_like(data)

        for i in range(data.shape[1]):
            filtered[:, i] = np.convolve(data[:, i], kernel, mode='same')
        return filtered

    def get_data(self, data_paths, steps=10):
        def clean_and_load_data(path, expected_cols=21):
            cleaned_lines = []
            with open(path, 'r') as f:
                for i, line in enumerate(f):
                    split_line = line.strip().split()
                    if len(split_line) == expected_cols:
                        cleaned_lines.append(line)
                    else:
                        print(f"Skipped row {i+1} due to column mismatch: {len(split_line)} columns")

            with open(path, 'w') as f:
                f.writelines(cleaned_lines)

            return np.loadtxt(path)


        def process_data(file_path):
            arr = clean_and_load_data(f'../data/datasets/kinova_data/{file_path}')

            if self.u_dim is not None and self.u_dim > 0:
                arr[:, :self.u_dim] = self._low_pass_filter(arr[:, :self.u_dim], window_size=1000)
                
            # arr = arr[::10, :]
            
            total_data = arr.shape[0]
            trimmed_len = (total_data // steps) * steps
            trimmed = arr[:trimmed_len]
            traj_count = trimmed_len // steps
            return trimmed.reshape(traj_count, steps, arr.shape[1]).transpose(1, 0, 2)
        
        lst = []
        for path in data_paths:
            lst.append(process_data(path))
        return np.concatenate(lst, axis=1)

    def collect_koopman_data(self, traj_num, steps):
        data = self.get_data(self.data_pathes, steps+1)
        print(f"Data shape: {data.shape}")
        return data[:, :traj_num, :]

class KoopmanDatasetCollector():
    def __init__(self, env_name, train_samples=60000, val_samples=20000, test_samples=20000, Ksteps=50, normalize=False, shuffle=False):
        self.normalize = normalize

        norm_str = "norm" if self.normalize else "unnorm"
        data_path = f"../data/datasets/dataset_{env_name}_{norm_str}_Ktrain_{train_samples}_Kval_{val_samples}_Ktest_{test_samples}_Ksteps_{Ksteps}.pt"
        
        self.u_dim = None
        self.state_dim = None

        if env_name == "Kinova":
            collector = KinovaDataCollecter()
            self.state_dim = collector.state_dim
            self.u_dim = collector.u_dim
        else:
            raise ValueError("Unknown environment name.")
        
        if not os.path.exists(data_path):
            data = collector.collect_koopman_data(train_samples+val_samples+test_samples, Ksteps)
            if shuffle:
                permutation = np.random.permutation(data.shape[1])
                shuffled = data[:, permutation, :]

                train_data = shuffled[:, :train_samples, :]
                val_data = shuffled[:, train_samples:train_samples+val_samples, :]
                test_data = shuffled[:, train_samples+val_samples:train_samples+val_samples+test_samples, :]
            else:
                train_data = data[:, :train_samples, :]
                val_data = data[:, train_samples:train_samples+val_samples, :]
                test_data = data[:, train_samples+val_samples:train_samples+val_samples+test_samples, :]
            
            if self.normalize:
                if self.u_dim is None:
                    train_mean = np.mean(train_data, axis=(0,1))
                    train_std = np.std(train_data, axis=(0,1))
                    train_data = (train_data - train_mean) / train_std
                    val_data = (val_data - train_mean) / train_std
                    test_data = (test_data - train_mean) / train_std
                else:
                    action_train_mean = np.mean(train_data[..., :self.u_dim], axis=(0,1))
                    action_train_std = np.std(train_data[..., :self.u_dim], axis=(0,1))
                    state_train_mean = np.mean(train_data[..., self.u_dim:], axis=(0,1))
                    state_train_std = np.std(train_data[..., self.u_dim:], axis=(0,1))

                    action_train_std = np.maximum(action_train_std, 1e-8)
                    state_train_std = np.maximum(state_train_std, 1e-8)

                    train_data[..., :self.u_dim] = (train_data[..., :self.u_dim] - action_train_mean) / action_train_std
                    train_data[..., self.u_dim:] = (train_data[..., self.u_dim:] - state_train_mean) / state_train_std
                    val_data[..., :self.u_dim] = (val_data[..., :self.u_dim] - action_train_mean) / action_train_std
                    val_data[..., self.u_dim:] = (val_data[..., self.u_dim:] - state_train_mean) / state_train_std
                    test_data[..., :self.u_dim] = (test_data[..., :self.u_dim] - action_train_mean) / action_train_std
                    test_data[..., self.u_dim:] = (test_data[..., self.u_dim:] - state_train_mean) / state_train_std
            
            torch.save({"Ktrain_data": train_data, "Kval_data": val_data, "Ktest_data": test_data}, data_path)

        self.train_data = torch.load(data_path, weights_only=False)["Ktrain_data"]
        self.val_data = torch.load(data_path, weights_only=False)["Kval_data"]
        self.test_data = torch.load(data_path, weights_only=False)["Ktest_data"]

    
    def get_data(self):
        return self.train_data, self.val_data, self.test_data

class KoopmanDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx, :]
