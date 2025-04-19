from pydrake.all import *
import numpy as np
import ipdb
from g1_env import G1Env

if __name__ == "__main__":
    G1_env = G1Env()       
    G1_env.run_simulation()
    ipdb.set_trace()