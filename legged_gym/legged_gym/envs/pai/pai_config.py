from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class PaiCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096  # 仿真环境的数量
        num_one_step_observations = 45  # 每个时间步的观察值的数量,
        num_observations = (
            num_one_step_observations * 6
        )  # 这个数据决定了obs_buf的长度，6倍长度意味着存放过去和现在一共6个时间步的观测值
        num_one_step_privileged_obs = (
            45 + 3 + 3 + 187
        )  # 每个时间步的特权观察值的数量 base_lin_vel, external_forces, scan_dots
        num_privileged_obs = num_one_step_privileged_obs * 1  # 特权观察值没有缓冲
        num_actions = 12  # 对应的是关节数量
        env_spacing = 3.0  # 环境之间的间距，此参数 当时用heightfields/trimeshes这两种meshtype时不适用
        send_timeouts = (
            True  # 是否向算法发送超时信息 send time out information to the algorithm
        )
        episode_length_s = 20  # 每个回合的时长 episode length in seconds

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.38]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "left_hip_yaw_joint": 0.0,  # [rad]
            "left_hip_roll_joint": 0.0,  # [rad]
            "left_hip_pitch_joint": 0.0,  # [rad]
            "left_knee_joint": 0.0,  # [rad]
            "left_ankle_pitch_joint": 0.0,  # [rad]
            "left_ankle_roll_joint": 0.0,  # [rad]
            "right_hip_yaw_joint": 0.0,  # [rad]
            "right_hip_roll_joint": 0.0,  # [rad]
            "right_hip_pitch_joint": 0.0,  # [rad]
            "right_knee_joint": 0.0,  # [rad]
            "right_ankle_pitch_joint": 0.0,  # [rad]
            "right_ankle_roll_joint": 0.0,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {'hip_yaw_joint': 40.0, 'hip_roll_joint': 40.0, 'hip_pitch_joint': 40.0,
                    'knee_joint': 40.0, 'ankle_pitch_joint': 40, 'ankle_roll_joint': 40,}
        damping = {'hip_yaw_joint': 1.0, 'hip_roll_joint': 1.0, 'hip_pitch_joint': 1.0, 
                    'knee_joint': 1.0, 'ankle_pitch_joint': 1.0, 'ankle_roll_joint': 1.0}
        
        action_scale = 0.5 # action scale: target angle = actionScale * action + defaultAngle
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # 控制动作更新的次数。每个策略时间步（policy DT）内的控制动作更新次数等于仿真时间步（sim DT）内的次数。
        # 刷新頻率 1/（dt*decimation） decimation = 4 dt为sim下的dt=0.005，算出来刷新频率为 1/（0.005*4）=50
        # decimation = 10 dt为sim下的dt=0.001，算出来刷新频率为 1/（0.001*10）=100
        hip_reduction = 1.0  # 缩减比例 

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/pai_12dof/urdf/pai_12dof.urdf'
        name = "pai_12dof_v2_0312"
        
        foot_name = "ankle_roll"
        knee_name = "calf"
        
        terminate_after_contacts_on = ['base_link'] #在接触后需要终止仿真的身体部位列表。
        penalize_contacts_on = ["base_link"]# 需要惩罚接触的身体部位列表。
        
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False
        
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_payload_mass = True
        payload_mass_range = [-1, 2]

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        randomize_link_mass = False
        link_mass_range = [0.9, 1.1]
        
        randomize_friction = True
        friction_range = [0.2, 1.25]
        
        randomize_restitution = False
        restitution_range = [0., 1.0]
        
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        
        randomize_kp = True
        kp_range = [0.9, 1.1]
        
        randomize_kd = True
        kd_range = [0.9, 1.1]
        
        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.5, 1.5]
        
        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8
        
        push_robots = True
        push_interval_s = 16
        max_push_vel_xy = 1.

        delay = True

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -1.0
            foot_clearance = -0.01
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time =  0.0
            collision = -0.0
            feet_stumble = -0.0
            stand_still = -0.
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = 0.0
            dof_vel_limits = 0.0
            torque_limits = 0.0

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.95 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        base_height_target = 0.30
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = -0.20

class PaiCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'HIMOnPolicyRunner'
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 60 # per iteration
        max_iterations = 2000 # number of policy updates
        
        save_interval = 20 # check for potential saves every this many iterations
        experiment_name = "pai_ppo"
        run_name = ""

        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
    
