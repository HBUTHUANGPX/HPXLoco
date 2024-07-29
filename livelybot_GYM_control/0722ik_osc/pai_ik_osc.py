from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import numpy as np
import math
import os
import time

from mat_function import *
import pinocchio as pin
from pinocchio.utils import *

# debug = True
debug = False
if debug:
    _dt = 1.0 / 10000.0
else:
    _dt = 1.0 / 1000.0

asset_root = "/home/hpx/HPXLoco/livelybot_GYM_control/assets"
asset_files = "urdf/pai_12dof/urdf/pai_12dof.urdf"
asset_names = "pai_12dof"

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

np.set_printoptions(precision=5, suppress=True, linewidth=100000, threshold=100000)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=500, threshold=20000000)


class Pai_ik_osc:
    def __init__(self):
        self.gym = gymapi.acquire_gym()  # initialize gym
        self.init_buff()
        self.init()
        self.warp_tensor()

    def init_args(self):
        args = gymutil.parse_arguments(
            description="Asset and Environment Information",
            custom_parameters=custom_parameters,
        )  # parse arguments
        return args

    def init_sim_params(self):
        sim_params = gymapi.SimParams()  # create simulation context
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.up_axis = gymapi.UP_AXIS_Z
        # sim_params.dt = 1.0 / 100.0
        sim_params.dt = _dt
        # sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        print("sim_params.use_gpu_pipeline:", sim_params.use_gpu_pipeline)
        if self.args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 8
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005
            sim_params.physx.num_threads = self.args.num_threads
            sim_params.physx.use_gpu = self.args.use_gpu
        else:
            raise Exception("This example can only be used with PhysX")
        return sim_params

    def init_controller(self):
        self.controller = self.args.controller
        assert self.controller in {
            "ik",
            "osc",
        }, f"Invalid controller specified -- options are (ik, osc). Got: {self.controller}"

    def init_gym(self):
        return self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            self.args.physics_engine,
            self.sim_params,
        )

    def init_viewer(self):
        return self.gym.create_viewer(self.sim, gymapi.CameraProperties())

    def init_plane(self):
        plane_params = gymapi.PlaneParams()  # add ground plane
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def init_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.disable_gravity = False
        # asset_options.fix_base_link = True
        return self.gym.load_asset(self.sim, asset_root, asset_files, asset_options)

    def init_dof_props(self):
        dof_props = self.gym.get_asset_dof_properties(self.current_asset)
        if self.controller == "ik":
            dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"][:].fill(800.0)
            dof_props["damping"][:].fill(40.0)
        else:  # osc
            dof_props["driveMode"][:].fill(gymapi.DOF_MODE_EFFORT)
            dof_props["stiffness"][:].fill(0.0)
            dof_props["damping"][:].fill(0.0)
        return dof_props

    def init_envs(self):
        self.num_envs = self.args.num_envs
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        default_dof_state = np.ndarray((12,), dtype=gymapi.DofState.dtype)
        self.default_dof_pos = [
            0.0,
            0.0,
            -0.3,
            0.6,
            -0.3,
            0.0,
            0.0,
            0.0,
            -0.3,
            0.6,
            -0.3,
            0.0,
        ]
        default_dof_state["pos"] = self.default_dof_pos
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.3855)

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            pai_handle = self.gym.create_actor(
                env, self.current_asset, pose, asset_names, -1, -1
            )
            self.pai_handles.append(pai_handle)
            self.gym.set_actor_dof_properties(env, pai_handle, self.dof_props)
            self.gym.set_actor_dof_states(
                env,
                pai_handle,
                default_dof_state,
                gymapi.STATE_POS,
            )
            feet_l_idx = self.gym.find_actor_rigid_body_index(
                env, pai_handle, "l_ankle_roll_link", gymapi.DOMAIN_SIM
            )
            self.link_indexs["feet_l"].append(feet_l_idx)
            feet_r_idx = self.gym.find_actor_rigid_body_index(
                env, pai_handle, "r_ankle_roll_link", gymapi.DOMAIN_SIM
            )
            self.link_indexs["feet_r"].append(feet_r_idx)
            base_idx = self.gym.find_actor_rigid_body_index(
                env, pai_handle, "base_link", gymapi.DOMAIN_SIM
            )
            # print(feet_r_idx)
            self.link_indexs["base"].append(base_idx)

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
        self.gym.viewer_camera_look_at(
            self.viewer, None, gymapi.Vec3(2, 0, 0.5), gymapi.Vec3(-1, 0, 0)
        )
        self.initial_state = np.copy(
            self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL)
        )

    def init(self):
        self.args = self.init_args()
        self.sim_params = self.init_sim_params()
        self.init_controller()
        self.sim = self.init_gym()
        if self.sim is None:
            raise Exception("Failed to create sim")
        self.viewer = self.init_viewer()
        if self.viewer is None:
            raise Exception("Failed to create viewer")
        self.init_plane()
        self.current_asset = self.init_asset()
        if self.current_asset is None:
            raise Exception("Failed to load asset")
        self.dof_props = self.init_dof_props()
        self.init_envs()
        self.last_viewer_update_time = self.gym.get_sim_time(self.sim)

    def init_buff(self):
        self.envs = []
        self.pai_handles = []
        self.link_indexs = {"feet_l": [], "feet_r": [], "base": []}
        self.cont = 0
        self.dir = -1
        self.viewer_refresh_rate = 1.0 / 100.0

    def warp_tensor(self):
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)

        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, 12, 1)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, 12, 1)

        self.link_dict = self.gym.get_asset_rigid_body_dict(self.current_asset)
        for key, value in self.link_dict.items():
            print(f"{key}: {value}")
            
        self.dof_dict = self.gym.get_asset_dof_dict(self.current_asset)
        for key, value in self.dof_dict.items():
            print(f"{key}: {value}")  
        for i in range(self.num_envs):
            env = self.envs[i]
            actor_handle = self.pai_handles[i]
            self.actor_joint_dict = self.gym.get_actor_joint_dict(env,actor_handle)
            print(self.actor_joint_dict) 
        body_names = self.gym.get_asset_rigid_body_names(self.current_asset)
        body_index = [self.link_dict[s] for s in body_names if "base" in s]
        print("body_index: ", body_index)
        self.feet_index = [self.link_dict[s] for s in body_names if "ankle_roll" in s]
        print("feet_index: ", self.feet_index)

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "pai_12dof")
        self.jacobian = gymtorch.wrap_tensor(_jacobian)

        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        self.effort_action = torch.zeros_like(self.pos_action)

    def evt_handle(self):
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "reset" and evt.value > 0:
                self.gym.set_sim_rigid_body_states(
                    self.sim, self.initial_state, gymapi.STATE_ALL
                )
                self.cont = 0
                self.dir = -1

    def step_physics(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def refresh_tensor(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

    def update_viewer(self):
        current_time = self.gym.get_sim_time(self.sim)
        if current_time - self.last_viewer_update_time >= self.viewer_refresh_rate:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.last_viewer_update_time = current_time
        self.gym.sync_frame_time(self.sim)

    def destroy(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def query_viewer_has_closed(self):
        return self.gym.query_viewer_has_closed(self.viewer)

    def tf_isaac_init(self):
        down_hight = 0.05
        self.feet_r_baselink_tf_mat_up = get_tf_mat(
            self.rb_states, self.link_indexs["base"], self.link_indexs["feet_r"]
        )
        self.feet_l_baselink_tf_mat_up = get_tf_mat(
            self.rb_states, self.link_indexs["base"], self.link_indexs["feet_l"]
        )
        self.feet_r_baselink_tf_mat_down = self.feet_r_baselink_tf_mat_up.clone()
        self.feet_l_baselink_tf_mat_down = self.feet_l_baselink_tf_mat_up.clone()
        self.feet_r_baselink_tf_mat_down[:, 2, 3] = (
            self.feet_r_baselink_tf_mat_up[:, 2, 3] + down_hight
        )
        self.feet_l_baselink_tf_mat_down[:, 2, 3] = (
            self.feet_l_baselink_tf_mat_up[:, 2, 3] + down_hight
        )
        # print(feet_r_baselink_tf_mat_down[0])

        self.baselink_feet_r_tf_mat_up = invert_tf_mat(self.feet_r_baselink_tf_mat_up)
        self.baselink_feet_l_tf_mat_up = invert_tf_mat(self.feet_l_baselink_tf_mat_up)
        self.baselink_feet_r_tf_mat_down = invert_tf_mat(
            self.feet_r_baselink_tf_mat_down
        )
        self.baselink_feet_l_tf_mat_down = invert_tf_mat(self.feet_l_baselink_tf_mat_up)
        # self.j_L = self.jacobian[:, self.feet_index[0], :, 6:12]
        # self.j_R = self.jacobian[:, self.feet_index[1], :, 12:]

    def ik(self):
        damping = 0.05
        feet_r_baselink_tf_mat_now = get_tf_mat(
            self.rb_states, self.link_indexs["base"], self.link_indexs["feet_r"]
        )
        feet_l_baselink_tf_mat_now = get_tf_mat(
            self.rb_states, self.link_indexs["base"], self.link_indexs["feet_l"]
        )
        if self.dir == 1:  # down
            feet_r_baselink_tf_mat_target = self.feet_r_baselink_tf_mat_down
            feet_l_baselink_tf_mat_target = self.feet_l_baselink_tf_mat_down
        elif self.dir == -1:  # up
            feet_r_baselink_tf_mat_target = self.feet_r_baselink_tf_mat_up
            feet_l_baselink_tf_mat_target = self.feet_l_baselink_tf_mat_up
        dpose_r = compute_dpose(
            feet_r_baselink_tf_mat_target, feet_r_baselink_tf_mat_now
        )
        dpose_l = compute_dpose(
            feet_l_baselink_tf_mat_target, feet_l_baselink_tf_mat_now
        )
        u_r = control_ik(dpose_r, damping, self.j_R, self.num_envs)
        u_l = control_ik(dpose_l, damping, self.j_L, self.num_envs)

        self.pos_action[:, :6] = self.dof_pos.squeeze(-1)[:, :6] + u_l
        self.pos_action[:, 6:] = self.dof_pos.squeeze(-1)[:, 6:] + u_r
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.pos_action)
        )

    def cont_turn_dir(self):
        self.cont += 1
        if self.cont > int(2 / self.sim_params.dt):
            self.cont = 0
            self.dir *= -1
            # print("turn direction: ", self.dir)

    def init_pinocchio_data(self):
        self.kp = 150.0
        self.kd = 2.0 * np.sqrt(self.kp)
        self.kp_null = 10.0
        self.kd_null = 2.0 * np.sqrt(self.kp_null)

        self.epsilon = 1e-6

        self.model = pin.buildModelFromUrdf(
            "/home/hpx/HPXLoco/livelybot_GYM_control/assets/urdf/pai_12dof/urdf/pai_12dof.urdf"
        )
        print("model name: " + self.model.name)
        self.data = self.model.createData()
        self.default_dof_pos_np = np.array(self.default_dof_pos)
        pin.forwardKinematics(self.model, self.data, self.default_dof_pos_np)
        pin.framesForwardKinematics(self.model, self.data, self.default_dof_pos_np)
        self.ankle_roll_frame_id = [
            self.model.getFrameId("l_ankle_roll_link"),
            self.model.getFrameId("r_ankle_roll_link"),
        ]
        self.ankle_roll_joint_id = [
            self.model.getJointId("left_ankle_roll_joint"),
            self.model.getJointId("right_ankle_roll_joint"),
        ]
        self.feet_to_base_tf_mat_init = [
            self.data.oMf[id].inverse().homogeneous for id in self.ankle_roll_frame_id
        ]
        self.feet_to_base_tf_mat_up = np.copy(self.feet_to_base_tf_mat_init)
        self.feet_to_base_tf_mat_down = np.copy(self.feet_to_base_tf_mat_init)
        self.feet_to_base_tf_mat_down[0][2, 3] -= 0.04
        self.feet_to_base_tf_mat_down[1][2, 3] -= 0.04

        self.base_to_feet_tf_mat_init = [
            self.data.oMf[id].homogeneous for id in self.ankle_roll_frame_id
        ]
        self.base_to_feet_tf_mat_up = np.copy(self.base_to_feet_tf_mat_init)
        self.base_to_feet_tf_mat_down = np.copy(self.base_to_feet_tf_mat_init)
        self.base_to_feet_tf_mat_down[0][2, 3] += 0.04
        self.base_to_feet_tf_mat_down[1][2, 3] += 0.04

        print(
            "self.data.oMf[self.ankle_roll_frame_id[0]].homogeneous: \r\n",
            self.data.oMf[self.ankle_roll_frame_id[0]].homogeneous,
        )
        print("self.base_to_feet_tf_mat_down: \r\n", self.base_to_feet_tf_mat_down[0])
        print("self.base_to_feet_tf_mat_up: \r\n", self.base_to_feet_tf_mat_up[0])
        self.joint_indices = [[5, 4, 3, 2, 1, 0], [11, 10, 9, 8, 7, 6]]

    def compute_dpose(goal_transform, current_transform):
        """
        计算从当前变换到目标变换的位姿误差。
        goal_transform: 形状为 (4, 4) 的目标变换矩阵数组
        current_transform: 形状为 (4, 4) 的当前变换矩阵数组
        返回: 形状为 (6, 1) 的位姿误差数组
        """
        # 计算相对变换矩阵
        relative_transform = np.linalg.inv(current_transform) @ goal_transform

        # 提取位置误差
        pos_err = relative_transform[:3, 3]

        # 提取旋转误差
        rotation_matrix_err = relative_transform[:3, :3]
        orn_err = np.zeros(3)
        orn_err[0] = rotation_matrix_err[2, 1] - rotation_matrix_err[1, 2]
        orn_err[1] = rotation_matrix_err[0, 2] - rotation_matrix_err[2, 0]
        orn_err[2] = rotation_matrix_err[1, 0] - rotation_matrix_err[0, 1]
        orn_err = orn_err / 2.0

        # 合并位置误差和旋转误差
        dpose = np.concatenate([pos_err, orn_err]).reshape(6, 1)

        return dpose

    def osc_up_down(self):
        action = np.zeros((4, 12), dtype=np.float32)
        # print("==============================", self.dir)
        # for i in range(1):
        for i in range(self.num_envs):
            # print("================================", i, "============================")
            if self.dir == 1:  # down
                self.base_to_feet_tf_mat_target = self.base_to_feet_tf_mat_down
                self.feet_to_base_tf_mat_target = self.feet_to_base_tf_mat_down
            elif self.dir == -1:
                self.base_to_feet_tf_mat_target = self.base_to_feet_tf_mat_up
                self.feet_to_base_tf_mat_target = self.feet_to_base_tf_mat_up
                
            while 1:
                self.dof_state_all = self.gym.get_actor_dof_states(
                    self.envs[i], self.pai_handles[i], gymapi.STATE_ALL
                )
                self.dof_pos_np = self.dof_state_all["pos"]
                self.dof_vel_np = self.dof_state_all["vel"]
                if np.isnan(self.dof_pos_np).any() or np.isnan(self.dof_vel_np).any():
                    return 1
                else:
                    break

            pin.forwardKinematics(self.model, self.data, self.dof_pos_np)
            pin.framesForwardKinematics(self.model, self.data, self.dof_pos_np)
            M = pin.crba(self.model, self.data, self.dof_pos_np)
            # for j in range(1):  # 0 l,1 r
            for j in range(2):  # 0 l,1 r
                # print(self.ankle_roll_frame_id[j])
                base_to_feet_tf_mat = self.data.oMf[self.ankle_roll_frame_id[j]]
                feet_to_base_tf_mat_now = base_to_feet_tf_mat.inverse()
                base_to_ankle_roll_link_J = pin.computeFrameJacobian(
                    self.model,
                    self.data,
                    self.dof_pos_np,
                    self.ankle_roll_frame_id[j],
                    pin.ReferenceFrame.LOCAL,
                )
                # print(base_to_ankle_roll_link_J)
                if j == 0:
                    dof_vel = self.dof_vel_np[:6]
                    J = base_to_ankle_roll_link_J[:, :6]
                elif j == 1:
                    dof_vel = self.dof_vel_np[6:]
                    J = base_to_ankle_roll_link_J[:, 6:]
                # print(J)
                # print("self.base_to_feet_tf_mat_target[j]:\r\n",self.base_to_feet_tf_mat_target[j])
                # print("base_to_feet_tf_mat:\r\n",base_to_feet_tf_mat.homogeneous)
                dpose = self.compute_dpos_3(
                    self.base_to_feet_tf_mat_target[j], base_to_feet_tf_mat.homogeneous
                )
                # print("dpose: ",dpose)
                if j == 0:
                    a = self.osc(
                        dpose,
                        self.default_dof_pos_np[:6],
                        M[:6, :6],
                        base_to_ankle_roll_link_J[:, :6],
                        self.dof_pos_np[:6],
                        self.dof_vel_np[:6],
                        base_to_ankle_roll_link_J[:, :6] @ self.dof_vel_np[:6],
                    )
                    action[i, :6] = a
                elif j == 1:
                    a = self.osc(
                        dpose,
                        self.default_dof_pos_np[6:],
                        M[6:, 6:],
                        base_to_ankle_roll_link_J[:, 6:],
                        self.dof_pos_np[6:],
                        self.dof_vel_np[6:],
                        base_to_ankle_roll_link_J[:, 6:] @ self.dof_vel_np[6:],
                    )
                    action[i, 6:] = a
            # print(action[i,:])
        
        action_tensor = torch.from_numpy(action)
        # print(action_tensor)
        rt_flag = self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(action_tensor)
        )
        # print(rt_flag)
        # self.osc()
        # print(self.default_dof_pos_np)

    def osc(self, dpos, default_dof_pos_np, mm, j_eef, dof_pos, dof_vel, end_vel):
        kp = 1500.0
        kd = 2.0 * np.sqrt(kp)
        kp_null = 10.0
        kd_null = 2.0 * np.sqrt(kp_null)
        mm_inv = np.linalg.inv(mm)
        m_eef_inv = j_eef @ mm_inv @ j_eef.T
        m_eef = np.linalg.inv(m_eef_inv)
        u = j_eef.T @ m_eef @ (kp * dpos - kd * end_vel)
        j_eef_inv = m_eef @ j_eef @ mm_inv
        u_null = kd_null * -dof_vel + kp_null * (
            (default_dof_pos_np - dof_pos + np.pi) % (2 * np.pi) - np.pi
        )
        u_null = mm @ u_null
        u += (np.eye(6) - j_eef.T @ j_eef_inv) @ u_null
        print(np.eye(6) @ u_null)
        return np.clip(u, -20., 20.)

    def orientation_error(self, desired, current):
        cc = self.quat_conjugate(current)
        q_r = self.quat_mul(desired, cc)
        return q_r[:3] * np.sign(q_r[3])

    def quat_conjugate(self, a):
        return np.array([-a[0], -a[1], -a[2], a[3]])

    def quat_mul(self, a, b):
        x1, y1, z1, w1 = a
        x2, y2, z2, w2 = b
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([x, y, z, w])

    def matrix_to_quat(self, matrix):
        m = matrix[:3, :3]
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (m[2, 1] - m[1, 2]) / S
            y = (m[0, 2] - m[2, 0]) / S
            z = (m[1, 0] - m[0, 1]) / S
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / S
            x = 0.25 * S
            y = (m[0, 1] + m[1, 0]) / S
            z = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / S
            x = (m[0, 1] + m[1, 0]) / S
            y = 0.25 * S
            z = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / S
            x = (m[0, 2] + m[2, 0]) / S
            y = (m[1, 2] + m[2, 1]) / S
            z = 0.25 * S
        return np.array([x, y, z, w])

    def compute_dpos_2(self, goal_transform, current_transform):
        """
        计算从当前变换到目标变换的位姿误差。
        goal_transform: 形状为 (4, 4) 的目标变换矩阵数组
        current_transform: 形状为 (4, 4) 的当前变换矩阵数组
        返回: 形状为 (1, 6) 的位姿误差数组
        """
        # 计算相对变换矩阵
        relative_transform = np.linalg.inv(current_transform) @ goal_transform
        # 提取位置误差
        pos_err = relative_transform[:3, 3]
        # 提取旋转误差
        goal_quat = self.matrix_to_quat(goal_transform)
        current_quat = self.matrix_to_quat(current_transform)
        orn_err = self.orientation_error(goal_quat, current_quat)
        # 合并位置误差和旋转误差
        dpose = np.concatenate([pos_err, orn_err]).reshape(1, 6)
        print("dpose: ", dpose)
        return dpose

    def compute_dpos_3(self, goal_transform, current_transform):
        """
        计算从当前变换到目标变换的位姿误差。
        goal_transform: 形状为 (4, 4) 的目标变换矩阵数组
        current_transform: 形状为 (4, 4) 的当前变换矩阵数组
        返回: 形状为 (1, 6) 的位姿误差数组
        """
        # 计算相对变换矩阵
        relative_transform = np.linalg.inv(current_transform) @ goal_transform
        pos_err = relative_transform[:3, 3]
        rotation_matrix_err = relative_transform[:3, :3]
        orn_err = np.zeros(3, dtype=np.float32)
        orn_err[0] = rotation_matrix_err[2, 1] - rotation_matrix_err[1, 2]
        orn_err[1] = rotation_matrix_err[0, 2] - rotation_matrix_err[2, 0]
        orn_err[2] = rotation_matrix_err[1, 0] - rotation_matrix_err[0, 1]
        orn_err = orn_err / 2.0
        dpos = np.hstack((pos_err, orn_err))
        # print("dpos: ", dpos)
        return dpos

    def ik_up_down(self):
        action = np.zeros((4, 12), dtype=np.float32)
        # for i in range(1):
        print("==============================", self.dir)
        for i in range(self.num_envs):
            if self.dir == 1:  # down
                self.base_to_feet_tf_mat_target = self.base_to_feet_tf_mat_down
                self.feet_to_base_tf_mat_target = self.feet_to_base_tf_mat_down
            elif self.dir == -1:
                self.base_to_feet_tf_mat_target = self.base_to_feet_tf_mat_up
                self.feet_to_base_tf_mat_target = self.feet_to_base_tf_mat_up

            while 1:
                self.dof_state_all = self.gym.get_actor_dof_states(
                    self.envs[i], self.pai_handles[i], gymapi.STATE_ALL
                )
                self.dof_pos_np = self.dof_state_all["pos"]
                self.dof_vel_np = self.dof_state_all["vel"]
                if np.isnan(self.dof_pos_np).any() or np.isnan(self.dof_vel_np).any():
                    return 1
                else:
                    break
            pin.forwardKinematics(self.model, self.data, self.dof_pos_np)
            pin.framesForwardKinematics(self.model, self.data, self.dof_pos_np)
            for j in range(2):  # 0 l,1 r
                base_to_feet_tf_mat = self.data.oMf[self.ankle_roll_frame_id[j]]
                feet_to_base_tf_mat_now = base_to_feet_tf_mat.inverse()
                base_to_ankle_roll_link_J = pin.computeFrameJacobian(
                    self.model,
                    self.data,
                    self.dof_pos_np,
                    self.ankle_roll_frame_id[j],
                    pin.ReferenceFrame.LOCAL,
                )
                if j == 0:
                    base_to_ankle_roll_link_J = base_to_ankle_roll_link_J[:, :6]
                    dof_pos = self.dof_pos_np[:6]
                    dof_vel = self.dof_vel_np[:6]
                elif j == 1:
                    base_to_ankle_roll_link_J = base_to_ankle_roll_link_J[:, 6:]
                    dof_pos = self.dof_pos_np[6:]
                    dof_vel = self.dof_vel_np[6:]


                # print("u:", u)
                u = self.control_ik(
                    self.compute_dpos_3(
                        self.base_to_feet_tf_mat_target[j],
                        base_to_feet_tf_mat.homogeneous,
                    ),
                    base_to_ankle_roll_link_J,
                    0.05,
                )
                # print("u:", u)
                if j == 0:
                    action[i, :6] = dof_pos + u
                elif j == 1:
                    action[i, 6:] = dof_pos + u
        action_tensor = torch.from_numpy(action.reshape(4, 12, 1))
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(action_tensor)
        )

    def control_ik(self, dpos, j_eef, damping):
        lmbda = np.eye(6) * (damping**2)
        j_eef_T = j_eef.T
        u = j_eef_T @ np.linalg.pinv(j_eef @ j_eef_T + lmbda) @ dpos
        return u


print("Working directory: %s" % os.getcwd())

a = Pai_ik_osc()
a.tf_isaac_init()
a.init_pinocchio_data()
# while 0:
viewer_refresh_rate = 1.0 / 60.0
cnt = 0
while not a.query_viewer_has_closed():
    a.evt_handle()
    a.step_physics()

    # a.refresh_tensor()
    if a.controller == "ik":
        a.ik_up_down()
    else:
        a.osc_up_down()
    if debug:
        time.sleep(0.001)
    # cnt+=1
    # if cnt>10:
    # break
    a.cont_turn_dir()
    a.update_viewer()

a.destroy()
