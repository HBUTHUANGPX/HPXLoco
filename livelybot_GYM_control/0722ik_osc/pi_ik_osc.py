from __future__ import print_function
import numpy as np
import math
import os

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import time
from mat_function import *

import pinocchio
debug = True
if debug:
    _dt = 1.0/100000.0
else:
    _dt = 1.0/100.0

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


torch.set_printoptions(precision=4, sci_mode=False, linewidth=500, threshold=20000000)
gym = gymapi.acquire_gym()  # initialize gym
custom_parameters = [
    {
        "name": "--controller",
        "type": str,
        "default": "osc",
        "help": "Controller to use for Franka. Options are {ik, osc}",
    },
    {
        "name": "--num_envs",
        "type": int,
        "default": 4,
        "help": "Number of environments to create",
    },
]
args = gymutil.parse_arguments(
    description="Asset and Environment Information",
    custom_parameters=custom_parameters,
)  # parse arguments

# Grab controller
controller = args.controller
assert controller in {
    "ik",
    "osc",
}, f"Invalid controller specified -- options are (ik, osc). Got: {controller}"


sim_params = gymapi.SimParams()  # create simulation context
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.up_axis = gymapi.UP_AXIS_Z
# sim_params.dt = 1.0 / 100.0
sim_params.dt = _dt
# sim_params.use_gpu_pipeline = args.use_gpu_pipeline
print("sim_params.use_gpu_pipeline:", sim_params.use_gpu_pipeline)
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(
    args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params
)
if sim is None:
    raise Exception("Failed to create sim")

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

plane_params = gymapi.PlaneParams()  # add ground plane
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)


# Print out the working directory
# helpful in determining the relative location that assets will be loaded from
print("Working directory: %s" % os.getcwd())

asset_root = "/home/hpx/HPXLoco/livelybot_GYM_control/assets"
asset_files = "urdf/pai_12dof/urdf/pai_12dof.urdf"
asset_names = "pai_12dof"
loaded_assets = []

print("Loading asset '%s' from '%s'" % (asset_files, asset_root))
asset_options = gymapi.AssetOptions()
asset_options.flip_visual_attachments = False
asset_options.disable_gravity = False
# asset_options.fix_base_link = True

current_asset = gym.load_asset(sim, asset_root, asset_files, asset_options)

if current_asset is None:
    print("*** Failed to load asset '%s'" % (asset_files, asset_root))
    quit()

# print_asset_info(current_asset, asset_names)

# configure dofs
dof_props = gym.get_asset_dof_properties(current_asset)
if controller == "ik":
    dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"][:].fill(400.0)
    dof_props["damping"][:].fill(40.0)
else:  # osc
    dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
    dof_props["stiffness"][:7].fill(0.0)
    dof_props["damping"][:7].fill(0.0)

# Setup environment spacing
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 2.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

envs = []
actor_handles = []
pose = gymapi.Transform()
shape = (12,)
default_dof_state = np.ndarray(shape, dtype=gymapi.DofState.dtype)
default_dof_pos = [
    0.0,
    0.0,
    -0.3,
    -0.6,
    -0.3,
    0.0,
    0.0,
    0.0,
    0.3,
    0.6,
    0.3,
    0.0,
]
default_dof_pos_tensor = to_torch(default_dof_pos, device="cpu")
default_dof_state["pos"] = default_dof_pos
feet_r_idxs = []
feet_l_idxs = []
base_idxs = []
joint_transforms = []
pai_handles = []
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)
    # Add actors to environment
    pose.p = gymapi.Vec3(0.0, 0.0, 0.3855)
    pai_handle = gym.create_actor(env, current_asset, pose, asset_names, -1, -1)
    pai_handles.append(pai_handle)
    actor_count = gym.get_actor_count(env)
    gym.set_actor_dof_properties(env, pai_handle, dof_props)
    gym.set_actor_dof_states(
        env,
        pai_handle,
        default_dof_state,
        gymapi.STATE_POS,
    )
    feet_l_idx = gym.find_actor_rigid_body_index(
        env, pai_handle, "l_ankle_roll_link", gymapi.DOMAIN_SIM
    )
    # print(feet_l_idx)
    feet_l_idxs.append(feet_l_idx)

    feet_r_idx = gym.find_actor_rigid_body_index(
        env, pai_handle, "r_ankle_roll_link", gymapi.DOMAIN_SIM
    )
    # print(feet_r_idx)
    feet_r_idxs.append(feet_r_idx)

    base_idx = gym.find_actor_rigid_body_index(
        env, pai_handle, "base_link", gymapi.DOMAIN_SIM
    )
    # print(feet_r_idx)
    base_idxs.append(base_idx)


gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(2, 0, 0.5), gymapi.Vec3(-1, 0, 0))

initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
# print(dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 12, 1)
dof_vel = dof_states[:, 1].view(num_envs, 12, 1)

link_dict = gym.get_asset_rigid_body_dict(current_asset)
print("link_dict:",link_dict)
body_names = gym.get_asset_rigid_body_names(current_asset)
body_index = [link_dict[s] for s in body_names if "base" in s]
# print("body_index: ", body_index)
feet_index = [link_dict[s] for s in body_names if "ankle_roll" in s]
# print("feet_index: ", feet_index)


# get jacobian tensor
_jacobian = gym.acquire_jacobian_tensor(sim, "pai_12dof")
jacobian = gymtorch.wrap_tensor(_jacobian)
print(jacobian.size())


# get mass matrix tensor
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "pai_12dof")
mm = gymtorch.wrap_tensor(_massmatrix)
# print("mm.size(): ", mm.size())
mm = mm[:, :12, :12]
# print("mm.size(): ", mm.size())
# print("mm: ", mm[0])

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)
effort_action = torch.zeros_like(pos_action)
# print(pos_action.size())


down_hight = 0.05
feet_r_baselink_tf_mat_up = get_tf_mat(rb_states, base_idxs, feet_r_idxs)
feet_l_baselink_tf_mat_up = get_tf_mat(rb_states, base_idxs, feet_l_idxs)
feet_r_baselink_tf_mat_down = feet_r_baselink_tf_mat_up.clone()
feet_l_baselink_tf_mat_down = feet_l_baselink_tf_mat_up.clone()
feet_r_baselink_tf_mat_down[:, 2, 3] = feet_r_baselink_tf_mat_up[:, 2, 3] + down_hight
feet_l_baselink_tf_mat_down[:, 2, 3] = feet_l_baselink_tf_mat_up[:, 2, 3] + down_hight
# print(feet_r_baselink_tf_mat_down[0])

baselink_feet_r_tf_mat_up = invert_tf_mat(feet_r_baselink_tf_mat_up)
baselink_feet_l_tf_mat_up = invert_tf_mat(feet_l_baselink_tf_mat_up)
baselink_feet_r_tf_mat_down = invert_tf_mat(feet_r_baselink_tf_mat_down)
baselink_feet_l_tf_mat_down = invert_tf_mat(feet_l_baselink_tf_mat_up)

# print(feet_r_baselink_tf_mat_up[0])
# print(baselink_feet_r_tf_mat_up[0])
# print(baselink_feet_r_tf_mat_down[0])

up_vel = down_hight / 2.0
dt_move = up_vel * sim_params.dt
# print(dt_move)
# compute_dpose()
cont = 0
dir = -1

damping = 0.05

j_L = jacobian[:, feet_index[0], :, 6:12]
j_R = jacobian[:, feet_index[1], :, 12:]
while not gym.query_viewer_has_closed(viewer):

    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
            cont = 0
            dir = -1
    # _state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))
    # print("=============================")
    # print(_state[0][0][0][2],_state[6][0][0][2]) #足底默认高0.0368 baselink 高0.3965
    # 足底默认高0.0368 baselink 高0.3825

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)
    feet_r_baselink_tf_mat_now = get_tf_mat(rb_states, base_idxs, feet_r_idxs)
    feet_l_baselink_tf_mat_now = get_tf_mat(rb_states, base_idxs, feet_l_idxs)
    baselink_feet_r_tf_mat_now = invert_tf_mat(feet_r_baselink_tf_mat_now)
    baselink_feet_l_tf_mat_now = invert_tf_mat(feet_l_baselink_tf_mat_now)

    if dir == 1:  # down
        feet_r_baselink_tf_mat_target = feet_r_baselink_tf_mat_down
        feet_l_baselink_tf_mat_target = feet_l_baselink_tf_mat_down
        baselink_feet_l_tf_mat_target = baselink_feet_l_tf_mat_down
        baselink_feet_r_tf_mat_target = baselink_feet_r_tf_mat_down
    elif dir == -1:  # up
        feet_r_baselink_tf_mat_target = feet_r_baselink_tf_mat_up
        feet_l_baselink_tf_mat_target = feet_l_baselink_tf_mat_up
        baselink_feet_r_tf_mat_target = baselink_feet_r_tf_mat_up
        baselink_feet_l_tf_mat_target = baselink_feet_l_tf_mat_up

    if controller == "ik":
        dpose_r = compute_dpose(
            feet_r_baselink_tf_mat_target, feet_r_baselink_tf_mat_now
        )
        dpose_l = compute_dpose(
            feet_l_baselink_tf_mat_target, feet_l_baselink_tf_mat_now
        )
        u_r = control_ik(dpose_r, damping, j_R, num_envs)
        u_l = control_ik(dpose_l, damping, j_L, num_envs)
    else:
        dpose_l = compute_dpose(
            baselink_feet_l_tf_mat_target, baselink_feet_l_tf_mat_now
        )
        dpose_r = compute_dpose(
            baselink_feet_r_tf_mat_target, baselink_feet_r_tf_mat_now
        )
        vel_feet_l_in_base = torch.matmul(j_L, dof_vel[:, :6]).squeeze(-1)
        vel_base_in_feet_l = transform_velocity_to_feet_frame(
            vel_feet_l_in_base, feet_r_baselink_tf_mat_now[:, :3, :3]
        )
        # print(vel_feet_l_in_base.size(), vel_base_in_feet_l.size())
        print(vel_feet_l_in_base[0])
        print(vel_base_in_feet_l[0])
        u_l = control_osc(
            dpose_l,
            torch.flip(default_dof_pos_tensor[:6], [0]),
            invert_mass_matrix(mm[:, :6, :6], baselink_feet_l_tf_mat_now[:, :3, :3]),
            invert_jacobian(j_L, baselink_feet_l_tf_mat_now[:, :3, :3]),
            torch.flip(dof_pos[:, :6], [1]),
            torch.flip(dof_vel[:, :6], [1]),
            vel_base_in_feet_l
        )
        vel_feet_r_in_base = torch.matmul(j_R, dof_vel[:, :6]).squeeze(-1)
        vel_base_in_feet_r = transform_velocity_to_feet_frame(
            vel_feet_r_in_base, feet_r_baselink_tf_mat_now[:, :3, :3]
        )
        u_r = control_osc(
            dpose_r,
            torch.flip(default_dof_pos_tensor[6:], [0]),
            invert_mass_matrix(
                mm[:, 6:12, 6:12], baselink_feet_r_tf_mat_now[:, :3, :3]
            ),
            invert_jacobian(j_R, baselink_feet_r_tf_mat_now[:, :3, :3]),
            torch.flip(dof_pos[:, 6:], [1]),
            torch.flip(dof_vel[:, 6:], [1]),
            vel_base_in_feet_r
        )
        ...
    # print(rb_states[base_idxs, 7:][0])
    # print(rb_states[feet_l_idxs, 7:][0])
    # print(u_r[0], u_l[0])
    # print(dof_pos.squeeze(-1)[:, 6:], dof_pos.squeeze(-1)[:, :6])
    if controller == "ik":
        pos_action[:, :6] = dof_pos.squeeze(-1)[:, :6] + u_l
        pos_action[:, 6:] = dof_pos.squeeze(-1)[:, 6:] + u_r
    else:  # osc
        effort_action[:, :6] = u_l
        effort_action[:, 6:] = u_r

    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))
    elapsed_time = gym.get_sim_time(sim)
    # print(f"Elapsed simulation time: {elapsed_time} seconds, ", dir)
    if debug:
        time.sleep(0.01)
    # time.sleep(0.01)
    # print("jacobian: ",jacobian[0])
    # print(j_L[0])
    # print(j_R[0])
    # for i in jacobian[0]:
    #     a = i[:,6:]
    #     print(a)
    # break
    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)
    cont += 1
    if cont > int(2 / sim_params.dt):
        cont = 0
        dir *= -1
        print("turn direction: ", dir)

# Cleanup the simulator
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
