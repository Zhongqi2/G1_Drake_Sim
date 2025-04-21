# #!/usr/bin/env python3

# import numpy as np
# import time
# import pickle

# from pydrake.all import (
#     DiagramBuilder,
#     AddMultibodyPlantSceneGraph,
#     Parser,
#     Solve,
#     Meshcat,
#     MeshcatVisualizer,
# )
# from pydrake.multibody.meshcat import JointSliders
# from pydrake.planning import KinematicTrajectoryOptimization
# from pydrake.multibody.inverse_kinematics import MinimumDistanceLowerBoundConstraint


# def generate_random_joint_target(lower_limits, upper_limits):
#     return np.random.uniform(lower_limits, upper_limits)


# def build_robot_visualizer(model_url):
#     builder = DiagramBuilder()
#     plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
#     parser = Parser(plant)
#     parser.package_map().Add("project", "./")
#     parser.AddModelsFromUrl(model_url)
#     plant.Finalize()

#     meshcat = Meshcat()
#     MeshcatVisualizer.AddToBuilder(builder, scene_graph.get_query_output_port(), meshcat)
#     builder.AddSystem(JointSliders(meshcat, plant))

#     diagram = builder.Build()
#     context = diagram.CreateDefaultContext()
#     return plant, diagram, context


# def build_trajopt_model(model_url):
#     builder = DiagramBuilder()
#     plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
#     parser = Parser(plant)
#     parser.package_map().Add("project", "./")
#     parser.AddModelsFromUrl(model_url)
#     plant.Finalize()

#     diagram = builder.Build()
#     return plant, diagram


# def solve_trajectory(plant, diagram, start_q, goal_q):
#     trajopt = KinematicTrajectoryOptimization(plant.num_positions(), 10)
#     trajopt.AddDurationCost(1.0)
#     trajopt.AddPathLengthCost(1.0)
#     trajopt.AddPositionBounds(
#         plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
#     )
#     trajopt.AddVelocityBounds(
#         plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits()
#     )
#     trajopt.AddDurationConstraint(0.5, 5.0)
#     trajopt.AddPathPositionConstraint(start_q, start_q, 0.0)
#     trajopt.AddPathPositionConstraint(goal_q, goal_q, 1.0)
#     trajopt.AddPathVelocityConstraint(np.zeros(plant.num_positions()), np.zeros(plant.num_positions()), 0.0)
#     trajopt.AddPathVelocityConstraint(np.zeros(plant.num_positions()), np.zeros(plant.num_positions()), 1.0)

#     result = Solve(trajopt.prog())
#     if not result.is_success():
#         print("Initial optimization failed")
#         return None

#     context = diagram.CreateDefaultContext()
#     plant_context = diagram.GetMutableSubsystemContext(plant, context)
#     collision_constraint = MinimumDistanceLowerBoundConstraint(
#         plant=plant,
#         bound=0.0015,
#         plant_context=plant_context
#     )

#     for s in np.linspace(0, 1, 30):
#         trajopt.AddPathPositionConstraint(collision_constraint, s)

#     trajopt.SetInitialGuess(trajopt.ReconstructTrajectory(result))
#     final_result = Solve(trajopt.prog())
#     if not final_result.is_success():
#         print("Final optimization failed")
#         return None

#     qtraj = trajopt.ReconstructTrajectory(final_result)
#     t0, tf = qtraj.start_time(), qtraj.end_time()
#     dt = 0.001
#     times = np.linspace(t0, tf, int(round((tf - t0)/dt)) + 1)
#     trajectory = np.vstack([qtraj.value(t).reshape(1, -1) for t in times])

#     return trajectory


# def main():
#     model_url = "package://project/script/environment_description/robot_environment_cartpole.dmd.yaml"
#     plant_vis, diagram_vis, context_vis = build_robot_visualizer(model_url)
#     plant_opt, diagram_opt = build_trajopt_model(model_url)

#     lower_limits = plant_vis.GetPositionLowerLimits()
#     upper_limits = plant_vis.GetPositionUpperLimits()

#     start_q = np.zeros(plant_vis.num_positions())
#     collected_data = []
#     max_iterations = 300

#     for i in range(max_iterations + 1):
#         goal_q = generate_random_joint_target(lower_limits, upper_limits)
#         traj = solve_trajectory(plant_opt, diagram_opt, start_q, goal_q)
#         if traj is None:
#             continue

#         print(f"iteration: {i}")
#         collected_data.append({
#             "start": start_q.copy(),
#             "goal": goal_q.copy(),
#             "trajectory": traj.copy()
#         })

#         plant_context_vis = diagram_vis.GetMutableSubsystemContext(plant_vis, context_vis)
#         for j in range(0, traj.shape[0], 2):
#             plant_vis.SetPositions(plant_context_vis, traj[j])
#             diagram_vis.ForcedPublish(context_vis)


#         start_q = goal_q.copy()
#         time.sleep(1.0)

#     print(f"done, total successful iterations: {len(collected_data)}")
#     with open("kinova_gen3_random_data.pkl", "wb") as f:
#         pickle.dump(collected_data, f)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3

import numpy as np
import time
import pickle

from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    Solve,
    Meshcat,
    MeshcatVisualizer,
)
from pydrake.multibody.meshcat import JointSliders
from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.multibody.inverse_kinematics import MinimumDistanceLowerBoundConstraint


def generate_random_joint_target(lower_limits, upper_limits):
    """Sample a random joint configuration within finite [lower_limits, upper_limits]."""
    return np.random.uniform(lower_limits, upper_limits)


def build_robot_visualizer(model_url):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    parser.package_map().Add("project", "./")
    parser.AddModelsFromUrl(model_url)
    plant.Finalize()

    meshcat = Meshcat()
    MeshcatVisualizer.AddToBuilder(
        builder, scene_graph.get_query_output_port(), meshcat
    )
    builder.AddSystem(JointSliders(meshcat, plant))

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    return plant, diagram, context


def build_trajopt_model(model_url):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    parser.package_map().Add("project", "./")
    parser.AddModelsFromUrl(model_url)
    plant.Finalize()

    diagram = builder.Build()
    return plant, diagram


def solve_trajectory(
    plant,
    diagram,
    start_q,
    goal_q,
    pos_lower,
    pos_upper,
    vel_lower,
    vel_upper,
):
    # num_positions() is the dimensionality of q
    trajopt = KinematicTrajectoryOptimization(plant.num_positions(), 10)

    # 1) use *your* clipped bounds here:
    trajopt.AddPositionBounds(pos_lower, pos_upper)
    trajopt.AddVelocityBounds(vel_lower, vel_upper)

    # rest of your costs & constraints untouched
    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(1.0)
    trajopt.AddDurationConstraint(0.5, 5.0)
    trajopt.AddPathPositionConstraint(start_q, start_q, 0.0)
    trajopt.AddPathPositionConstraint(goal_q, goal_q, 1.0)
    trajopt.AddPathVelocityConstraint(
        np.zeros(plant.num_positions()),
        np.zeros(plant.num_positions()),
        0.0,
    )
    trajopt.AddPathVelocityConstraint(
        np.zeros(plant.num_positions()),
        np.zeros(plant.num_positions()),
        1.0,
    )

    # initial solve
    result = Solve(trajopt.prog())
    if not result.is_success():
        print("Initial optimization failed")
        return None

    # add your minimum‑distance (collision) constraints
    context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, context)
    collision_constraint = MinimumDistanceLowerBoundConstraint(
        plant=plant, bound=0.0015, plant_context=plant_context
    )
    for s in np.linspace(0, 1, 30):
        trajopt.AddPathPositionConstraint(collision_constraint, float(s))

    # re‑solve with the new constraints
    trajopt.SetInitialGuess(trajopt.ReconstructTrajectory(result))
    final_result = Solve(trajopt.prog())
    if not final_result.is_success():
        print("Final optimization failed")
        return None

    # extract the time‑parameterized trajectory
    qtraj = trajopt.ReconstructTrajectory(final_result)
    t0, tf = qtraj.start_time(), qtraj.end_time()
    dt = 0.001
    times = np.linspace(t0, tf, int(round((tf - t0) / dt)) + 1)
    trajectory = np.vstack([qtraj.value(t).reshape(1, -1) for t in times])

    return trajectory


def main():
    model_url = (
        "package://project/"
        "script/environment_description/robot_environment_cartpole.dmd.yaml"
    )
    plant_vis, diagram_vis, context_vis = build_robot_visualizer(model_url)
    plant_opt, diagram_opt = build_trajopt_model(model_url)

    # ——— clip position limits ———
    raw_pos_lo = plant_vis.GetPositionLowerLimits().copy()
    raw_pos_hi = plant_vis.GetPositionUpperLimits().copy()
    # replace any infinite bound with ±π
    pos_lo = raw_pos_lo
    pos_hi = raw_pos_hi
    pos_lo[~np.isfinite(raw_pos_lo)] = -np.pi
    pos_hi[~np.isfinite(raw_pos_hi)] = np.pi

    # ——— clip velocity limits ———
    raw_vel_lo = plant_vis.GetVelocityLowerLimits().copy()
    raw_vel_hi = plant_vis.GetVelocityUpperLimits().copy()
    # pick a default max from the *finite* velocities in your model
    finite_vi = raw_vel_hi[np.isfinite(raw_vel_hi)]
    default_max_vel = float(np.max(finite_vi)) if finite_vi.size else 1.0
    vel_lo = raw_vel_lo
    vel_hi = raw_vel_hi
    vel_lo[~np.isfinite(raw_vel_lo)] = -default_max_vel
    vel_hi[~np.isfinite(raw_vel_hi)] = default_max_vel

    # start at zero config
    start_q = np.zeros(plant_vis.num_positions())
    collected_data = []
    max_iterations = 300

    for i in range(max_iterations + 1):
        goal_q = generate_random_joint_target(pos_lo, pos_hi)
        traj = solve_trajectory(
            plant_opt,
            diagram_opt,
            start_q,
            goal_q,
            pos_lo,
            pos_hi,
            vel_lo,
            vel_hi,
        )
        if traj is None:
            continue

        print(f"iteration: {i}")
        collected_data.append(
            {"start": start_q.copy(), "goal": goal_q.copy(), "trajectory": traj.copy()}
        )

        # Publish frames at half-step to Meshcat
        plant_ctx = diagram_vis.GetMutableSubsystemContext(
            plant_vis, context_vis
        )
        for j in range(0, traj.shape[0], 2):
            plant_vis.SetPositions(plant_ctx, traj[j])
            diagram_vis.ForcedPublish(context_vis)

        start_q = goal_q.copy()
        time.sleep(1.0)

    print(f"done, total successful iterations: {len(collected_data)}")
    with open("kinova_gen3_random_data.pkl", "wb") as f:
        pickle.dump(collected_data, f)


if __name__ == "__main__":
    main()

