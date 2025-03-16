from pydrake.all import *
import numpy as np
import ipdb
import graphviz
from support_functions import AddShape, AddMultibodyTriad
import matplotlib.pyplot as plt

meshcat = StartMeshcat()

class G1Env():

    def __init__(self, time_step=0.001):
        self.builder = DiagramBuilder()
        self.time_step = time_step
        # Make the whole plant for scene
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=time_step)
        parser = Parser(self.plant)
        parser.package_map().Add("drake_project", "./")    
        directives = LoadModelDirectives("models/g1_description/g1_7dof_rubber_hand.yaml")
        models = ProcessModelDirectives(directives, self.plant, parser)

        # Make the plant for the g1 arm controller to use.
        self.controller_plant = MultibodyPlant(time_step = 0)
        parser1 = Parser(self.controller_plant)
        parser1.package_map().Add("drake_project", "./")    
        directives = LoadModelDirectives("models/g1_description/g1_7dof_rubber_hand.yaml")
        models = ProcessModelDirectives(directives, self.controller_plant, parser1)
        self.controller_plant.Finalize()

        # defind target frame
        G1 = self.plant.GetModelInstanceByName("G1")
        robot_base_frame = self.plant.GetFrameByName("pelvis")
        robot_right_hand_frame = self.plant.GetFrameByName("right_rubber_hand")
        AddMultibodyTriad(robot_base_frame, self.scene_graph, length=0.1, radius=0.005)
        AddMultibodyTriad(robot_right_hand_frame, self.scene_graph, length=0.1, radius=0.005)

        # build plant 
        self.plant.Finalize()
        
        # List of joint names in order
        joint_names = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint"
        ]

        # Retrieve the model instance
        model_instance = self.plant.GetModelInstanceByName("G1")

        # Initialize joint limit arrays
        self.joint_lower_limits = np.zeros(len(joint_names))
        self.joint_upper_limits = np.zeros(len(joint_names))

        # Retrieve and store joint limits
        for i, joint_name in enumerate(joint_names):
            joint = self.plant.GetJointByName(joint_name, model_instance)
            self.joint_lower_limits[i] = joint.position_lower_limits()[0]
            self.joint_upper_limits[i] = joint.position_upper_limits()[0]
            
        # Show scene visualization
        params = MeshcatVisualizerParams()
        visualizer = MeshcatVisualizer.AddToBuilder(self.builder, self.scene_graph, meshcat, params)

        # Add G1 controller
        self.robot_dof = self.controller_plant.num_positions()  
        kp = [50] * self.robot_dof
        ki = [1] * self.robot_dof
        kd = [10] * self.robot_dof
        G1_controller = self.builder.AddSystem(InverseDynamicsController(self.controller_plant, kp, ki, kd, False))
        G1_controller.set_name("robot_controller")

        # Add trajectory generator
        robot_traj_command = self.MakeRobotCommandTrajectory(self.controller_plant)
        robot_traj = self.builder.AddSystem(TrajectorySource(robot_traj_command))
        robot_traj.set_name("robot_traj")
    
        # Add discrete derivative to command velocities.
        desired_state_from_position = self.builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                self.controller_plant.num_positions(),
                time_step,
                suppress_initial_transient=True,
            )
        )
            
        self.builder.Connect(robot_traj.get_output_port(), desired_state_from_position.get_input_port())
        self.builder.Connect(desired_state_from_position.get_output_port(), G1_controller.get_input_port_desired_state())
        self.builder.Connect(self.plant.get_state_output_port(G1), G1_controller.get_input_port_estimated_state())
        self.builder.Connect(G1_controller.get_output_port_control(), self.plant.get_actuation_input_port(G1))

        self.diagram = self.builder.Build()
        
    def draw_control_diagram(self):
        graph = graphviz.Source(self.diagram.GetGraphvizString())
        graph.render(filename='svg', format='png', cleanup=True)
    
    def step(self, x, u_input):
        plant_context = self.controller_plant.CreateDefaultContext()
        # plant_context.SetContinuousState(x)
        # self.controller_plant.get_input_port(3).FixValue(plant_context, u_input)
        # x_dot = self.controller_plant.EvalTimeDerivatives(plant_context).CopyToVector()
        
        self.controller_plant.SetPositions(plant_context, x[:7])
        self.controller_plant.SetVelocities(plant_context, x[7:])
        M = self.controller_plant.CalcMassMatrixViaInverseDynamics(plant_context)
        Cv = self.controller_plant.CalcBiasTerm(plant_context)
        tauG = self.controller_plant.CalcGravityGeneralizedForces(plant_context)
        q_ddot = np.linalg.inv(M) @ (u_input - Cv - tauG)  # Compute acceleration q̈
        x_dot = np.concatenate((x[7:],q_ddot))
        x_next = x + x_dot * self.time_step
        
        return x_next
    
    def MakeRobotCommandTrajectory(self, plant):
        T = 2.0
        context = plant.CreateDefaultContext()

        # Get joint positions
        q_init = np.array([0,0,0,0,0,0,0])
        q_end = self.joint_lower_limits + (self.joint_upper_limits - self.joint_lower_limits) * np.random.rand(1, self.robot_dof)[0]
        A0 = np.vstack((q_init, q_end)).T  
        traj_wsg_command = PiecewisePolynomial.FirstOrderHold(
            [0, T],              
            A0,
        )

        return traj_wsg_command

    def run_simulation(self):
        # Run simulation
        diagram_context = self.diagram.CreateDefaultContext()
        simulator = Simulator(self.diagram, diagram_context)
        simulator.set_target_realtime_rate(1)
        meshcat.StartRecording(set_visualizations_while_recording=True)
        simulator.AdvanceTo(2)
        meshcat.PublishRecording()

        
