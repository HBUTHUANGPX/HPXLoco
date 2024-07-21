from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class PaiRoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_one_step_observations = 45
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = (
            45 + 3 + 3 + 187
        )  # one_step_observations(45) + base_lin_vel(3) + external_forces(3) + scan_dots(187)
        num_privileged_obs = (
            num_one_step_privileged_obs * 1
        )  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training).
        # None is returned otherwise
        num_actions = 12
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        # rough terrain only:
        measure_heights = True
        measured_points_x = [
            -0.8,
            -0.7,
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        # trimesh only:
        slope_treshold = (
            0.75  # slopes above this threshold will be corrected to vertical surfaces
        )

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 3.0
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.38]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "left_hip_yaw_joint": 0.0,  # [rad]
            "right_hip_yaw_joint": 0.0,  # [rad]
            "left_hip_roll_joint": 0.0,  # [rad]
            "right_hip_roll_joint": 0.0,  # [rad]
            "left_hip_pitch_joint": -0.3,  # [rad]
            "right_hip_pitch_joint": 0.3,  # [rad]
            "left_knee_joint": -0.6,  # [rad]
            "right_knee_joint": 0.6,  # [rad]
            "left_ankle_pitch_joint": -0.3,  # [rad]
            "right_ankle_pitch_joint": 0.3,  # [rad]
            "left_ankle_roll_joint": 0.0,  # [rad]
            "right_ankle_roll_joint": 0.0,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {"joint": 40.0}  # [N*m/rad]
        damping = {"joint": 2.0}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/pai_12dof/urdf/pai_12dof.urdf"
        name = "pai_12dof"
        foot_name = "ankle_roll"
        knee_name = "calf"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = (
            False  # Some .obj meshes must be flipped from y-up to z-up
        )

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_payload_mass = True
        payload_mass_range = [-1, 2]

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        randomize_link_mass = False
        link_mass_range = [0.9, 1.1]

        randomize_friction = True
        friction_range = [0.2, 1.25]

        randomize_restitution = False
        restitution_range = [0.0, 1.0]

        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]

        randomize_kp = True
        kp_range = [0.9, 1.1]

        randomize_kd = True
        kd_range = [0.9, 1.1]

        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.5, 1.5]

        disturbance = True
        disturbance_range = [-5.0, 5.0]
        disturbance_interval = 8

        push_robots = True
        push_interval_s = 16
        max_push_vel_xy = 1.0

        delay = True

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            track_lin_ang_vel = 0.5
            vel_mismatch_exp = 0.5
            orientation = -0.2
            base_height = -1.0
            base_acc = 0.2
            
            dof_acc = -2.5e-7
            dof_vel = -5e-4
            torques = -1e-5
            joint_power = -2e-5
            smoothness = -0.001
            
            foot_clearance = -0.01
            feet_air_time = 0.1
            feet_distance = 0.16   # 0.2
            knee_distance = 0.16   # 0.2
            
            # termination = -0.0
            # collision = -0.0
            # feet_stumble = -0.0
            # stand_still = -0.0
            # dof_pos_limits = 0.0
            # dof_vel_limits = 0.0
            # torque_limits = 0.0
            
        min_dist = 0.15
        max_dist = 0.2
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = (
            0.95  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        base_height_target = 0.38
        max_contact_force = 700.0  # forces above this value are penalized
        clearance_height_target = -0.34

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.0
        clip_actions = 100.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = (
                2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            )

class PaiRoughCfgPPO(LeggedRobotCfgPPO):

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "HIMActorCritic"
        algorithm_class_name = "HIMPPO"
        num_steps_per_env = 50  # per iteration
        max_iterations = 200000  # number of policy updates
        # logging
        save_interval = 20  # check for potential saves every this many iterations
        run_name = ""
        experiment_name = "rough_pai"
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        