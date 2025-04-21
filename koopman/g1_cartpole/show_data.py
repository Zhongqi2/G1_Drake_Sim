from g1_cartpole_env import G1Env
import ipdb
import numpy as np
from pydrake.all import Simulator
import matplotlib.pyplot as plt
if __name__ == "__main__":
    data = np.load('test.txt.npy')
    data = data.squeeze() 
    torque = data[:, :7]
    position = data[:, 7:15]
    velocity = data[:, 15:]
    # Time axis
    timesteps = range(data.shape[0])
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Torque plot
    for i in range(torque.shape[1]):
        axs[0].plot(timesteps, torque[:, i], label=f'Torque {i+1}')
    axs[0].set_ylabel('Torque')
    axs[0].legend()

    # Position plot
    for i in range(position.shape[1]):
        axs[1].plot(timesteps, position[:, i], label=f'Position {i+1}')
    axs[1].set_ylabel('Position')
    axs[1].legend()

    # Velocity plot
    for i in range(velocity.shape[1]):
        axs[2].plot(timesteps, velocity[:, i], label=f'Velocity {i+1}')
    axs[2].set_ylabel('Velocity')
    axs[2].set_xlabel('Timestep')
    axs[2].legend()

    plt.tight_layout()
    plt.show()
    ipdb.set_trace()