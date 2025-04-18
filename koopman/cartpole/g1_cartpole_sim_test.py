from g1_cartpole_env import G1Env
import ipdb
import numpy as np
from pydrake.all import Simulator

if __name__ == "__main__":
    env = G1Env()

    # Create context and fix a zero torque input
    context = env.diagram.CreateDefaultContext()
    torque_context = env.diagram.GetMutableSubsystemContext(env.torque_input, context)
    zero_u = np.zeros(env.plant.num_actuators())
    env.torque_port.FixValue(torque_context, zero_u)

    # Simulate for 10 seconds
    sim = Simulator(env.diagram, context)
    sim.set_target_realtime_rate(1.0)
    sim.AdvanceTo(10.0)

    ipdb.set_trace()