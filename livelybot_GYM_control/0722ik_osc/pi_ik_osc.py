from __future__ import print_function
import numpy as np
import math
import os

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import time

import pinocchio


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


def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping**2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    # j_eef @ j_eef_T 计算雅可比矩阵与其转置的乘积，形状为 (num_envs, 6, 6)。
    # j_eef @ j_eef_T + lmbda 在雅可比矩阵乘积上加上阻尼矩阵，形状为 (num_envs, 6, 6)。
    # torch.inverse(j_eef @ j_eef_T + lmbda) 计算上述矩阵的逆矩阵，形状为 (num_envs, 6, 6)。
    # j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) 计算雅可比矩阵转置与逆矩阵的乘积，形状为 (num_envs, 7, 6)。
    # j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose 计算上述结果与 dpose 的乘积，得到关节角度变化 u，形状为 (num_envs, 7, 1)。
    # u.view(num_envs, 7) 将结果调整为形状 (num_envs, 7)，表示每个环境中的关节角度变化。
    return u


def quaternion_to_rotation_matrix(q):
    q = q / torch.norm(q)
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    R = torch.tensor(
        [
            [
                1 - 2 * qy**2 - 2 * qz**2,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx**2 - 2 * qz**2,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx**2 - 2 * qy**2,
            ],
        ],
        device=q.device,
    )
    return R


def create_transformation_matrix(rotation_matrix, position):
    transformation_matrix = torch.eye(4, device=rotation_matrix.device)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position
    return transformation_matrix


torch.set_printoptions(precision=4, sci_mode=False, linewidth=500, threshold=20000000)
gym = gymapi.acquire_gym()  # initialize gym
custom_parameters = [
    {
        "name": "--controller",
        "type": str,
        "default": "ik",
        "help": "Controller to use for Franka. Options are {ik, osc}",
    },
    {
        "name": "--num_envs",
        "type": int,
        "default": 16,
        "help": "Number of environments to create",
    },
]
args = gymutil.parse_arguments(
    description="Asset and Environment Information",
    custom_parameters=custom_parameters,
)  # parse arguments

device = args.sim_device if args.use_gpu_pipeline else "cpu"

sim_params = gymapi.SimParams()  # create simulation context
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.dt = 1.0 / 60000000.0
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
structured_array = np.ndarray(shape, dtype=gymapi.DofState.dtype)
structured_array["pos"] = [
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
    pose.p = gymapi.Vec3(0.0, 0.0, 0.3825)
    pai_handle = gym.create_actor(env, current_asset, pose, asset_names, -1, -1)
    pai_handles.append(pai_handle)
    actor_count = gym.get_actor_count(env)
    # Iterate through all actors for the environment
    actor_handle = gym.get_actor_handle(env, actor_count)
    actor_handles.append(actor_handle)
    gym.set_actor_dof_states(
        env,
        actor_handle,
        structured_array,
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


print(joint_transforms)
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
print(link_dict)
body_names = gym.get_asset_rigid_body_names(current_asset)
body_index = [link_dict[s] for s in body_names if "base" in s]
# print("body_index: ", body_index)
feet_index = [link_dict[s] for s in body_names if "ankle_roll" in s]
# print("feet_index: ", feet_index)


# get jacobian tensor
_jacobian = gym.acquire_jacobian_tensor(sim, "pai_12dof")
jacobian = gymtorch.wrap_tensor(_jacobian)
print(jacobian.size())
j_L = jacobian[:, feet_index[0], :, 6:12]
j_R = jacobian[:, feet_index[1], :, 12:]

# get mass matrix tensor
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "pai_12dof")
mm = gymtorch.wrap_tensor(_massmatrix)
# print(mm)
mm = mm[:, :12, :12]  # only need elements corresponding to the franka arm

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)
effort_action = torch.zeros_like(pos_action)
# print(pos_action.size())
while not gym.query_viewer_has_closed(viewer):
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
    _state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))
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

    # print(feet_r_idxs, feet_l_idxs)
    feet_pos_r_at_world = rb_states[feet_r_idxs, :3]
    feet_pos_l_at_world = rb_states[feet_l_idxs, :3]
    base_pos_at_world = rb_states[base_idxs, :3]

    feet_pos_r_at_base = rb_states[feet_r_idxs, :3] - rb_states[base_idxs, :3]
    feet_pos_l_at_base = rb_states[feet_l_idxs, :3] - rb_states[base_idxs, :3]

    # print(feet_pos_l_at_base)

    feet_rot_r_at_world = rb_states[feet_r_idxs, 3:7]
    feet_rot_l_at_world = rb_states[feet_l_idxs, 3:7]
    base_rot_at_world = rb_states[base_idxs, 3:7]

    feet_rot_r_at_base = rb_states[feet_r_idxs, 3:7] - rb_states[base_idxs, 3:7]
    feet_rot_l_at_base = rb_states[feet_l_idxs, 3:7] - rb_states[base_idxs, 3:7]
    # print(base_rot_at_world, feet_rot_r_at_world)

    for i in range(1):
    # for i in range(num_envs):
        transform = gym.get_actor_joint_transforms(envs[i], pai_handles[i])
        # print(transform.size())
        j = 0
        transformation_matrixs = []
        for t in transform:
            position = torch.tensor([t['p']['x'], t['p']['y'], t['p']['z']])
            quaternion = torch.tensor(
                [t['r']['x'], t['r']['y'], t['r']['z'], t['r']['w']]
            )
            rotation_matrix = quaternion_to_rotation_matrix(quaternion)
            transformation_matrix = create_transformation_matrix(rotation_matrix, position)
            print("Transformation matrix:",j)
            j+=1
            print(transformation_matrix)
            transformation_matrixs.append(transformation_matrix)
        transformation_matrices_tensor = torch.stack(transformation_matrixs)
        print(transformation_matrices_tensor.shape)
        feet_r_tf_mt = torch.eye(4)
        feet_l_tf_mt = torch.eye(4)
        print(feet_r_tf_mt)
        for tf_index in range(6):
            feet_l_tf_mt = torch.inverse(transformation_matrices_tensor[tf_index,:,:]) @ feet_l_tf_mt 
            feet_r_tf_mt = torch.inverse(transformation_matrices_tensor[tf_index + 6,:,:]) @ feet_r_tf_mt 
        print(feet_r_tf_mt)
    # print("jacobian: ",jacobian[0])
    # print(j_L[0])
    # print(j_R[0])
    # for i in jacobian[0]:
    #     a = i[:,6:]
    #     print(a)
    break
    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# Cleanup the simulator
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
