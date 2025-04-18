from g1_cartpole_env import G1Env
import ipdb

if __name__ == "__main__":
    env = G1Env()
    env.run_simulation(2)
    ipdb.set_trace()