"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Asset and Environment Information
---------------------------------
Demonstrates introspection capabilities of the gym api at the asset and environment levels
- Once an asset is loaded its properties can be queried
- Assets in environments can be queried and their current states be retrieved
"""

import numpy as np

import os
from isaacgym import gymapi
from isaacgym import gymutil


def print_asset_info(asset, name):
    print("======== Asset info %s: ========" % (name))
    num_bodies = gym.get_asset_rigid_body_count(asset)
    num_joints = gym.get_asset_joint_count(asset)
    num_dofs = gym.get_asset_dof_count(asset)
    print("Got %d bodies, %d joints, and %d DOFs" % (num_bodies, num_joints, num_dofs))

    # Iterate through bodies
    print("Bodies:")
    for i in range(num_bodies):
        name = gym.get_asset_rigid_body_name(asset, i)
        print(" %2d: '%s'" % (i, name))

    # Iterate through joints
    print("Joints:")
    for i in range(num_joints):
        name = gym.get_asset_joint_name(asset, i)
        type = gym.get_asset_joint_type(asset, i)
        type_name = gym.get_joint_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))

    # iterate through degrees of freedom (DOFs)
    print("DOFs:")
    for i in range(num_dofs):
        name = gym.get_asset_dof_name(asset, i)
        type = gym.get_asset_dof_type(asset, i)
        type_name = gym.get_dof_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))


def print_actor_info(gym, env, actor_handle):

    name = gym.get_actor_name(env, actor_handle)

    body_names = gym.get_actor_rigid_body_names(env, actor_handle)
    body_dict = gym.get_actor_rigid_body_dict(env, actor_handle)

    joint_names = gym.get_actor_joint_names(env, actor_handle)
    joint_dict = gym.get_actor_joint_dict(env, actor_handle)

    dof_names = gym.get_actor_dof_names(env, actor_handle)
    dof_dict = gym.get_actor_dof_dict(env, actor_handle)

    print()
    print("===== Actor: %s =======================================" % name)

    print("\nBodies")
    print(body_names)
    print(body_dict)

    print("\nJoints")
    print(joint_names)
    print(joint_dict)

    print("\n Degrees Of Freedom (DOFs)")
    print(dof_names)
    print(dof_dict)
    print()

    # Get body state information
    body_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_ALL)

    # Print some state slices
    print("Poses from Body State:")
    print(body_states["pose"])  # print just the poses

    print("\nVelocities from Body State:")
    print(body_states["vel"])  # print just the velocities
    print()

    # iterate through bodies and print name and position
    body_positions = body_states["pose"]["p"]
    for i in range(len(body_names)):
        print("Body '%s' has position" % body_names[i], body_positions[i])

    print("\nDOF states:")

    # get DOF states
    dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)

    # print some state slices
    # Print all states for each degree of freedom
    print(dof_states)
    print()

    # iterate through DOFs and print name and position
    dof_positions = dof_states["pos"]
    for i in range(len(dof_names)):
        print("DOF '%s' has position" % dof_names[i], dof_positions[i])


# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Asset and Environment Information")

# create simulation context
sim_params = gymapi.SimParams()

sim_params.use_gpu_pipeline = False
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.up_axis = gymapi.UpAxis(1)
sim_params.dt = 0.00033333333
print(sim_params.gravity)
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(
    args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params
)

if sim is None:
    print("*** Failed to create sim")
    quit()
# add ground plane
plane_params = gymapi.PlaneParams()
print("----------------plane_params:", plane_params.distance)
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# Print out the working directory
# helpful in determining the relative location that assets will be loaded from
print("Working directory: %s" % os.getcwd())

# Path where assets are searched, relative to the current working directory
asset_root = "../assets"

# List of assets that will be loaded, both URDF and MJCF files are supported
# asset_files = ["urdf/aliengo/urdf/aliengo.urdf"]
asset_files = ["urdf/pai_12dof/urdf/pai_12dof.urdf"]
asset_names = ["pai_12dof"]
loaded_assets = []

# Load the assets and ensure that we are successful
for asset in asset_files:
    print("Loading asset '%s' from '%s'" % (asset, asset_root))
    asset_options = gymapi.AssetOptions()
    asset_options.flip_visual_attachments = False
    asset_options.disable_gravity = False
    current_asset = gym.load_asset(sim, asset_root, asset, asset_options)

    if current_asset is None:
        print("*** Failed to load asset '%s'" % (asset, asset_root))
        quit()
    loaded_assets.append(current_asset)

for i in range(len(loaded_assets)):
    print()
    print_asset_info(loaded_assets[i], asset_names[i])

# Setup environment spacing
spacing = 2.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)

# Create one environment
env = gym.create_env(sim, lower, upper, 1)

# Add actors to environment
pose = gymapi.Transform()
for i in range(len(loaded_assets)):
    pose.p = gymapi.Vec3(0.0, 0.0, 0.3825)
    print(f"x: {pose.p.x}, y: {pose.p.y}, z: {pose.p.z}")

    # pose.r = gymapi.Quat(0, 0.0, 0.0, 1)
    gym.create_actor(env, loaded_assets[i], pose, asset_names[i], -1, -1)

print("=== Environment info: ================================================")

actor_count = gym.get_actor_count(env)
print("%d actors total" % actor_count)

# Iterate through all actors for the environment
for i in range(actor_count):
    actor_handle = gym.get_actor_handle(env, i)
    print_actor_info(gym, env, actor_handle)
    dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)
    print(dof_states)    
    # for i in range(12):
    #     dof_states[i][0] = ar[i]
    shape = (12,)
    dtype = np.dtype([('pos', '<f4'), ('vel', '<f4')])
    structured_array = np.ndarray(shape, dtype=dtype)
    structured_array['pos'] = [0.0, 0.0, -0.3, -0.6, -0.3, 0.0, 0.0, 0.0, 0.3, 0.6, 0.3, 0.0]
    dof_states['pos'] = structured_array['pos']
    print(dof_states)    
    gym.set_actor_dof_states(
        env,
        actor_handle,
        dof_states,
        gymapi.STATE_POS,
    )

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(2, 0, 0.5), gymapi.Vec3(-1, 0, 0))
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))
# print(initial_state)
import time

while not gym.query_viewer_has_closed(viewer):
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
    _state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))
    # print(_state[0][0][2],_state[0][0][2])
    print("=============================")
    # print(_state)
    print("-")
    print(_state[0][0][0][2],_state[6][0][0][2]) #足底默认高0.0368 baselink 高0.3965
                                                 #足底默认高0.0368 baselink 高0.3825
    
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

# Cleanup the simulator
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
