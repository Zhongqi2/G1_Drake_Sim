from pydrake.all import *
import os
sys.path.append("../")
meshcat = StartMeshcat()

# Create a model visualizer and add the robot arm.
visualizer = ModelVisualizer(meshcat=meshcat)

# show g1 
visualizer.package_map().Add("drake_project", "./")
visualizer.parser().AddModels("models/g1_description/g1_7dof_cartpole.dmd.yaml")

test_mode = True if "TEST_SRCDIR" in os.environ else False
visualizer.Run(loop_once=test_mode)