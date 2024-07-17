#include <iostream>
#include <torch/torch.h>
#include "torch/script.h"
#include </usr/include/eigen3/Eigen/Dense>
#include <cmath>
#include <deque>
#include "ros/ros.h"
#include "../include/serial_struct.h"
#include "../include/hardware/robot.h"
#include <fstream>
#include <thread>
#include <condition_variable>
#include <../include/sim2real.h>
#include <unistd.h>
#include "tf/transform_datatypes.h"

#include "std_msgs/Float64.h"
#include "sim2real_msg/PosState.h"
#include "sim2real_msg/Target.h"
#include "sim2real_msg/PosUrdfState.h"
#include "sim2real_msg/TargetUrdf.h"
#include <yaml-cpp/yaml.h>
#define CONFIG_PATH CMAKE_CURRENT_SOURCE_DIR "/config.yaml"

using namespace std;

void Robot::rbt_init()
{
    n.setParam("gsmp_controller_switch", "null");
    // ROS_INFO("FIND IMU");
    write();
}
void Robot::ankle_kinematics(double theta_1, double theta_2, double &theta_p, double &theta_r)
{
    double d1 = 0.036;
    double d2 = 0.023;
    double L1 = 0.025;
    double h1 = 0.112;
    double a1 = 0.115;
    double h2 = 0.065;
    double a2 = 0.07;

    a2 = h2;
    a1 = h1;
    d2 = L1;

    double D_L = 2 * d1 * d2;
    double E_L = 2 * L1 * d1 * sin(theta_2) - 2 * d1 * h2;
    double F_L = 2 * d2 * L1 * sin(theta_2) - 2 * d2 * h2;
    double G_L = 2 * d1 * d1;
    double H_L = 2 * d2 * L1 * cos(theta_2);
    double I_L = L1 * L1 + h2 * h2 + 2 * d1 * d1 + d2 * d2 - a2 * a2 - 2 * h2 * L1 * sin(theta_2);
    double D_R = D_L;
    double E_R = 2 * L1 * d1 * sin(theta_1) + 2 * d1 * h1;
    double F_R = 2 * d2 * L1 * sin(theta_1) + 2 * d2 * h1;
    double G_R = G_L;
    double H_R = 2 * d2 * L1 * cos(theta_1);
    double I_R = L1 * L1 + h1 * h1 + 2 * d1 * d1 + d2 * d2 - a1 * a1 + 2 * h1 * L1 * sin(theta_1);

    theta_p = 0;
    theta_r = 0;

    int max_iterations = 100;
    double tolerance = 1e-5;

    for (int i = 0; i < max_iterations; ++i)
    {
        double f1 = -D_L * sin(theta_p) * sin(theta_r) - E_L * sin(theta_r) + F_L * sin(theta_p) - G_L * cos(theta_r) - H_L * cos(theta_p) + I_L;
        double f2 = D_R * sin(theta_p) * sin(theta_r) - E_R * sin(theta_r) - F_R * sin(theta_p) - G_R * cos(theta_r) - H_R * cos(theta_p) + I_R;

        if (fabs(f1) < tolerance && fabs(f2) < tolerance)
        {
            break;
        }

        double J11 = -D_L * cos(theta_p) * sin(theta_r) + F_L * cos(theta_p) + H_L * sin(theta_p);
        double J12 = -D_L * sin(theta_p) * cos(theta_r) - E_L * cos(theta_r) + G_L * sin(theta_r);
        double J21 = D_R * cos(theta_p) * sin(theta_r) - F_R * cos(theta_p) - H_R * sin(theta_p);
        double J22 = D_R * sin(theta_p) * cos(theta_r) - E_R * cos(theta_r) + G_R * sin(theta_r);

        double det = J11 * J22 - J12 * J21;
        if (fabs(det) < 1e-10)
        {
            theta_p = 0.0;
            theta_r = 0.0;
            break;
        }

        double J11_inv = J22 / det;
        double J12_inv = -J12 / det;
        double J21_inv = -J21 / det;
        double J22_inv = J11 / det;

        double delta_theta_p = -(J11_inv * f1 + J12_inv * f2);
        double delta_theta_r = -(J21_inv * f1 + J22_inv * f2);

        theta_p = theta_p + delta_theta_p;
        theta_r = theta_r + delta_theta_r;
    }
}
void Robot::ankle_ikinematics(double theta_p, double theta_r, double &theta_1, double &theta_2)
{
    double d1 = 0.036;
    double d2 = 0.023;
    double L1 = 0.025;
    double h1 = 0.112;
    double a1 = 0.115;
    double h2 = 0.065;
    double a2 = 0.07;

    a2 = h2;
    a1 = h1;
    d2 = L1;

    double A_L = 2 * d2 * L1 * sin(theta_p) - 2 * L1 * d1 * sin(theta_r) - 2 * h2 * L1;
    double A_R = -2 * d2 * L1 * sin(theta_p) - 2 * L1 * d1 * sin(theta_r) + 2 * h1 * L1;
    double B_L = 2 * d2 * L1 * cos(theta_p);
    double B_R = 2 * d2 * L1 * cos(theta_p);
    double C_L = 2 * d1 * d1 + h2 * h2 - a2 * a2 + L1 * L1 + d2 * d2 - 2 * d1 * d1 * cos(theta_r) + 2 * d1 * h2 * sin(theta_r) - 2 * d2 * h2 * sin(theta_p) - 2 * d1 * d2 * sin(theta_p) * sin(theta_r);
    double C_R = 2 * d1 * d1 + h1 * h1 - a1 * a1 + L1 * L1 + d2 * d2 - 2 * d1 * d1 * cos(theta_r) - 2 * d1 * h1 * sin(theta_r) - 2 * d2 * h1 * sin(theta_p) + 2 * d1 * d2 * sin(theta_p) * sin(theta_r);

    double Len_L = A_L * A_L - C_L * C_L + B_L * B_L;
    double Len_R = A_R * A_R - C_R * C_R + B_R * B_R;

    if (Len_L > 0 && Len_R > 0)
    {
        theta_2 = 2 * atan((-A_L - sqrt(Len_L)) / (B_L + C_L));
        theta_1 = 2 * atan((-A_R + sqrt(Len_R)) / (B_R + C_R));
    }
    else
    {
        theta_1 = 0.0;
        theta_2 = 0.0;
    }
}

template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node &node)
{
    std::vector<T> values;
    for (const auto &val : node)
    {
        values.push_back(val.as<T>());
    }
    return values;
}

void Robot::readYaml()
{
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(CONFIG_PATH);
    }
    catch (YAML::BadFile &e)
    {
        // ROS_INFO("The file %s does not exist", CONFIG_PATH);
        return;
    }
    this->params.model_name = config["model_name"].as<std::string>();
    this->params.dt = config["dt"].as<double>();
    this->params.decimation = config["decimation"].as<int>();
    this->params.num_observations = config["num_observations"].as<int>();
    this->params.clip_obs = config["clip_obs"].as<double>();
    this->params.clip_actions_upper = torch::tensor(ReadVectorFromYaml<double>(config["clip_actions_upper"])).view({1, -1});
    this->params.clip_actions_lower = torch::tensor(ReadVectorFromYaml<double>(config["clip_actions_lower"])).view({1, -1});
    this->params.action_scale = config["action_scale"].as<double>();
    this->params.hip_scale_reduction = config["hip_scale_reduction"].as<double>();
    this->params.hip_scale_reduction_indices = ReadVectorFromYaml<int>(config["hip_scale_reduction_indices"]);
    this->params.num_of_dofs = config["num_of_dofs"].as<int>();
    this->params.lin_vel_scale = config["lin_vel_scale"].as<double>();
    this->params.ang_vel_scale = config["ang_vel_scale"].as<double>();
    this->params.dof_pos_scale = config["dof_pos_scale"].as<double>();
    this->params.dof_vel_scale = config["dof_vel_scale"].as<double>();
    // this->params.commands_scale = torch::tensor(ReadVectorFromYaml<double>(config["commands_scale"])).view({1, -1});
    this->params.commands_scale = torch::tensor({this->params.lin_vel_scale, this->params.lin_vel_scale, this->params.ang_vel_scale});
    this->params.rl_kp = torch::tensor(ReadVectorFromYaml<double>(config["rl_kp"])).view({1, -1});
    this->params.rl_kd = torch::tensor(ReadVectorFromYaml<double>(config["rl_kd"])).view({1, -1});
    this->params.fixed_kp = torch::tensor(ReadVectorFromYaml<double>(config["fixed_kp"])).view({1, -1});
    this->params.fixed_kd = torch::tensor(ReadVectorFromYaml<double>(config["fixed_kd"])).view({1, -1});
    this->params.torque_limits = torch::tensor(ReadVectorFromYaml<double>(config["torque_limits"])).view({1, -1});
    this->params.default_dof_pos = torch::tensor(ReadVectorFromYaml<double>(config["default_dof_pos"])).view({1, -1});
    this->params.init_dof_pos = torch::tensor(ReadVectorFromYaml<double>(config["init_dof_pos"])).view({1, -1});
    this->params.joint_controller_names = ReadVectorFromYaml<std::string>(config["joint_controller_names"]);
}

void Robot::quaternionToEulerArray()
{
    // // 四元数到欧拉角的转换
    float x = quat.x(), y = quat.y(), z = quat.z(), w = quat.w();
    float t0 = +2.0 * (w * x + y * z);
    float t1 = +1.0 - 2.0 * (x * x + y * y);
    float roll_x = std::atan2(t0, t1);

    float t2 = +2.0 * (w * y - z * x);
    if (t2 > 1.0)
    {
        t2 = 1.0;
    }
    else if (t2 < -1.0)
    {
        t2 = -1.0;
    }
    float pitch_y = std::asin(t2);

    float t3 = +2.0 * (w * z + x * y);
    float t4 = +1.0 - 2.0 * (y * y + z * z);
    float yaw_z = std::atan2(t3, t4);
    eu_ang << roll_x, pitch_y, yaw_z;
}

void Robot::read()
{
    float pos, vel, tau;
    for (int i = 0; i < cfg.num_actions; i++)
    {
        motor_back_t motor;
        // motor = *rb.Motors[cfg.map_index[i]]->get_current_motor_state();
        // q[i] = motor.position;
        // dq[i] = motor.velocity;

        rb.get_motor_state_dynamic_config(pos, vel, tau, cfg.map_index[i]);
        q[i] = pos;
        dq[i] = vel;

        pos_msg.q[i] = q[i];
        pos_msg.dq[i] = dq[i];
        pos_msg.tau[i] = tau;
        if (i == 3 || i == 9)
        {
            q[i] = q[i] + q[i - 1];
            dq[i] = dq[i] + dq[i - 1];
        }
        // if(i==9){
        //     q[i] = q[i] - q[i-1];
        //     dq[i] = dq[i] - dq[i-1];
        // }
    }
    // std_msgs::Float64 float_msg;
    // float_msg.data = q[0];
    // rev_pub.publish(float_msg);
    // ankle_kinematics();
    double q4 = q[4];
    double q5 = q[5];
    double q10 = q[10];
    double q11 = q[11];
    ankle_kinematics(q4, q5, q[4], q[5]);
    ankle_kinematics(q10, q11, q[10], q[11]);

    // ROS_INFO("theta1:%f,  theta2:%f,  thetap:%f,  thetar:%f",q4*180/3.14,q5*180/3.14,q[4]*180/3.14,q[5]*180/3.14);
    for (int i = 0; i < cfg.num_actions; i++)
    {
        // q[i] = q[i] * cfg.motor_direction[i] - params.default_dof_pos[0][i].item<double>() - params.init_dof_pos[0][i].item<double>();
        q[i] = q[i] * cfg.motor_direction[i] - params.default_dof_pos[0][i].item<double>();
        // q[i] = q[i] * cfg.motor_direction[i];
        dq[i] = dq[i] * cfg.motor_direction[i];
    }
    for (int i = 0; i < 12; ++i)
    {
        urdf_pos_state.urdf_q[i] = q[i];
    }
    urdf_pos_pub.publish(urdf_pos_state);
    quat.x() = yesenceIMU_.orientation.x;
    quat.y() = yesenceIMU_.orientation.y;
    quat.z() = yesenceIMU_.orientation.z;
    quat.w() = yesenceIMU_.orientation.w;
    // base_ang_vel<<  0.5 * yesenceIMU_.angular_velocity.x + 0.5 * base_ang_vel[0], 0.5 * yesenceIMU_.angular_velocity.y + 0.5 * base_ang_vel[1], 0.5 * yesenceIMU_.angular_velocity.z + 0.5 * base_ang_vel[2];
    base_ang_vel << yesenceIMU_.angular_velocity.x, yesenceIMU_.angular_velocity.y, yesenceIMU_.angular_velocity.z;
    quaternionToEulerArray();
    rviz_target_pos.header.stamp = ros::Time::now();
    for (int i = 0; i < 12; ++i)
    {
        rviz_target_pos.position[i] = q[i];
    }
    rviz_target_pub.publish(rviz_target_pos);

    for (int i = 0; i < 3; ++i)
    {
        pos_msg.eu_ang[i] = eu_ang[i];
        pos_msg.base_ang_vel[i] = base_ang_vel[i];
    }
    pos_pub.publish(pos_msg);
}

void Robot::write()
{
    for (int i = 0; i < cfg.num_actions; i++)
    {
        target_q[i] = (action[i] * cfg.action_scales + params.init_dof_pos[0][i].item<double>()) * cfg.motor_direction[i];
    }

    for (int i = 0; i < 12; ++i)
    {
        urdf_target_state.urdf_target_q[i] = target_q[i];
    }
    urdf_target_pub.publish(urdf_target_state);

    target_q[3] = target_q[3] - target_q[2];
    target_q[9] = target_q[9] - target_q[8];

    double target_q4 = target_q[4];
    double target_q5 = target_q[5];
    double target_q10 = target_q[10];
    double target_q11 = target_q[11];

    ankle_ikinematics(target_q4, target_q5, target_q[4], target_q[5]);
    ankle_ikinematics(target_q10, target_q11, target_q[10], target_q[11]);
    for(int i = 0; i < 12; ++i){
        target_msg.target_q[i] = target_q[i];
    }    
    target_pub.publish(target_msg);
    dynamic_offset_hip = -0.260;
    dynamic_offset_knee = 0.055;
    dynamic_offset_ankle = 0.0;
    float offset[12] = {0, 0, dynamic_offset_hip, dynamic_offset_knee, 0, 0,
                        0, 0, dynamic_offset_hip, dynamic_offset_knee, 0, 0};
    for (int i = 0; i < cfg.num_actions; i++)
    {
        rb.fresh_cmd_dynamic_config(target_q[i] - offset[i], 0, 0,
                                    params.rl_kp[0][i].item<double>(), params.rl_kd[0][i].item<double>(),
                                    cfg.map_index[i]);
    }
    rb.motor_send_2();
}

void Robot::update_action()
{

    for (int i = 0; i < cfg.frame_stack; ++i)
    {
        // �? hist_obs 中获取当前向�?
        const Eigen::VectorXd &current_vector = hist_obs[i];
        for (int j = 0; j < cfg.num_single_obs; j++)
        {
            input[0][i * cfg.num_single_obs + j] = current_vector[j];
        }
    }
    torch::Tensor input_tensor = torch::from_blob(input, {1, cfg.num_single_obs * cfg.frame_stack}, torch::kFloat32).clone();
    std::vector<c10::IValue> tmp_input;
    tmp_input.push_back(c10::IValue(input_tensor));

    torch::Tensor output = policy.forward(tmp_input).toTensor();
    // std::cout << output << std::endl;
    for (int i = 0; i < cfg.num_actions; i++)
    {
        action[i] = output.accessor<float, 2>()[0][i];
    }
    for (int i = 0; i < cfg.num_actions; i++)
    {

        if (action[i] > cfg.clip_actions)
        {
            action[i] = cfg.clip_actions;
        }
        else if (action[i] < -cfg.clip_actions)
        {
            action[i] = -cfg.clip_actions;
        }
    }
    return;
}

void Robot::update_obs()
{
    read(); // update dof_info & imu info
    eu_ang = eu_ang.unaryExpr([](float angle)
                              { return angle > M_PI ? angle - 2 * M_PI : angle; });

    obs[0] = std::sin(2 * M_PI * count_lowlevel * cfg.dt / 0.64);
    obs[1] = std::cos(2 * M_PI * count_lowlevel * cfg.dt / 0.64);
    obs[2] = cmd.vx * cfg.lin_vel_scale;
    obs[3] = cmd.vy * cfg.lin_vel_scale;
    obs[4] = cmd.dyaw * cfg.ang_vel_scale;
    obs.segment(5, 12) = q * cfg.dof_pos_scale;
    obs.segment(17, 12) = dq * cfg.dof_vel_scale;
    obs.segment(29, 12) = action;
    obs.segment(41, 3) = base_ang_vel;
    obs.segment(44, 3) = eu_ang;
    for (int i = 0; i < cfg.num_single_obs; ++i)
    {
        if (obs[i] > cfg.clip_observations)
        {
            obs[i] = cfg.clip_observations;
        }
        else if (obs[i] < -cfg.clip_observations)
        {
            obs[i] = -cfg.clip_observations;
        }
    }
    count_lowlevel += 10;
    hist_obs.push_back(obs);
    hist_obs.pop_front();
}

bool Robot::setupImu()
{
    imuSensorInterface_.registerHandle(hardware_interface::ImuSensorHandle(
        "base_imu", "base_imu", imuData_.ori, imuData_.ori_cov, imuData_.angular_vel, imuData_.angular_vel_cov,
        imuData_.linear_acc, imuData_.linear_acc_cov));
    imuData_.ori_cov[0] = 0.0012;
    imuData_.ori_cov[4] = 0.0012;
    imuData_.ori_cov[8] = 0.0012;

    imuData_.angular_vel_cov[0] = 0.0004;
    imuData_.angular_vel_cov[4] = 0.0004;
    imuData_.angular_vel_cov[8] = 0.0004;
    return true;
}

void Robot::joy_callback(const sensor_msgs::Joy::ConstPtr &msg)
{
    // ROS_INFO("joy callback");
    ROS_INFO("offset_hip:%f, offset_knee:%f, offset_ankle:%f, vx:%f, dyaw:%f, vy:%f", dynamic_offset_hip, dynamic_offset_knee, dynamic_offset_ankle, cmd.vx, cmd.dyaw, cmd.vy);
    switch (state)
    {
    case sys_state::close:
        // ROS_INFO("sys_state::close");

        if (msg->buttons[9] == 1)
            state = sys_state::init;
        if (msg->buttons[5] == 1)
            state = sys_state::exit;
        if (msg->buttons[3] == 1)
        {
            dynamic_offset_hip += 0.005;
        }
        if (msg->buttons[0] == 1)
        {
            dynamic_offset_hip -= 0.005;
        }
        if (msg->buttons[1] == 1)
        {
            dynamic_offset_knee += 0.005;
        }
        if (msg->buttons[2] == 1)
        {
            dynamic_offset_knee -= 0.005;
        }
        if (msg->buttons[7] == 1)
        {
            dynamic_offset_ankle += 0.005;
        }
        if (msg->buttons[6] == 1)
        {
            dynamic_offset_ankle -= 0.005;
        }
        break;
    case sys_state::init:
        // ROS_INFO("sys_state::init");

        if (msg->buttons[4] == 1)
        {
            cmd.vx = 0.01;
            state = sys_state::exec;
        }
        if (msg->buttons[5] == 1)
            state = sys_state::exit;
        if (msg->buttons[10] == 1)
            state = sys_state::close;
        if (msg->buttons[3] == 1)
        {
            dynamic_offset_hip += 0.005;
        }
        if (msg->buttons[0] == 1)
        {
            dynamic_offset_hip -= 0.005;
        }
        if (msg->buttons[1] == 1)
        {
            dynamic_offset_knee += 0.005;
        }
        if (msg->buttons[2] == 1)
        {
            dynamic_offset_knee -= 0.005;
        }
        if (msg->buttons[7] == 1)
        {
            dynamic_offset_ankle += 0.005;
        }
        if (msg->buttons[6] == 1)
        {
            dynamic_offset_ankle -= 0.005;
        }
        break;
    case sys_state::exec:
        // ROS_INFO("sys_state::exec");
        // for (motor *m : rb.Motors)
        // {
        //     ROS_INFO("Motrs: pos %f, vel %f, tqe %f\n", m->get_current_motor_state()->position, m->get_current_motor_state()->velocity, m->get_current_motor_state()->torque);
        //     rb.send_get_motor_state_cmd();
        // }
        cmd.vy = 0.00;

        if (msg->buttons[5] == 1)
            state = sys_state::exit;
        if (msg->buttons[9] == 1)
            state = sys_state::init;
        if (msg->axes[7] > 0)
        {
            if (cmd.vx < 0.4)
            {
                cmd.vx += 0.05;
            }
        }
        if (msg->axes[7] < 0)
        {
            if (cmd.vx > -0.4)
            {
                cmd.vx += -0.05;
            }
        }
        if (msg->axes[6] > 0)
        {
            //  cmd.vx=0.01;
            if (cmd.dyaw < 1.0)
            {
                cmd.dyaw += 0.1;
            }
            // if(cmd.vy<0.2)
            // {
            //     cmd.vy+=0.02;
            // }
        }
        if (msg->axes[6] < 0)
        {
            //  cmd.vx=0.01;
            if (cmd.dyaw > -1.0)
            {
                cmd.dyaw += -0.1;
            }
            // if(cmd.vy>-0.2)
            // {
            // 	cmd.vy+=-0.02;
            // }
        }

        if (msg->axes[3] < 10000)
        {
            //  cmd.dyaw=-0.0;
        }
        if (msg->axes[3] > 10000)
        {
            //  cmd.dyaw=0.0;
        }
        if (msg->buttons[3] == 1)
        {
            dynamic_offset_hip += 0.005;
        }
        if (msg->buttons[0] == 1)
        {
            dynamic_offset_hip -= 0.005;
        }
        if (msg->buttons[1] == 1)
        {
            dynamic_offset_knee += 0.005;
        }
        if (msg->buttons[2] == 1)
        {
            dynamic_offset_knee -= 0.005;
        }
        if (msg->buttons[7] == 1)
        {
            dynamic_offset_ankle += 0.005;
        }
        if (msg->buttons[6] == 1)
        {
            dynamic_offset_ankle -= 0.005;
        }

        // if(msg->buttons[3]==1)
        // {
        //    if(cmd.dyaw<0.5)
        //    {
        // 	  cmd.dyaw+=0.05;
        // 	}
        // }
        // if(msg->buttons[0]==1)
        // {
        // 	if(cmd.dyaw>-0.5)
        // 	{
        // 		cmd.dyaw+=-0.05;
        // 	}
        // }
        // if(msg->buttons[1]==1)
        // {
        //      cmd.dyaw=0.0;
        // }
        // if(msg->buttons[2]==1)
        // {
        //      cmd.dyaw=0.0;
        // }

        break;
    default:
        return;
    }
}

void Robot::run()
{
    update_obs();
    update_action();
    write();
}

void executable()
{
    ros::Rate r(100);
    Robot rbt("/home/hpx/HPXLoco/src/sim2real/policy/2024_0716_HIM_2_policy.pt");
    // while(1){

    // }
    while (ros::ok())
    {
        switch (rbt.get_state())
        {
        case sys_state::close:
            r.sleep();
            // ROS_INFO("robot state close press A to init state\npress right top button to exit state\n");

            break;

        case sys_state::init:

            // ROS_INFO("robot state init\npress A to exec state\npress B to close state\npress RB button to exit state\n");
            rbt.rbt_init();
            // r.sleep();

            // for(int i=0; i<100; i++){
            //     rbt.read();
            //     }
            rbt.read();
            // ROS_INFO("robot state end");

            break;

        case sys_state::exec:
            // ROS_INFO("robot state exec, stay at origin \npress A and left top axes to control yaw \npress A and left bottom axes to control x,y velocity\n press B to init state \n press press RB button to exit state \n");
            rbt.run();
            r.sleep();

            break;

        case sys_state::exit:
            // ROS_INFO("robot state exit");

            return;

        default:
            break;
        }
    }
}

int main(int argc, char **argv)
{
    int start = 0;
    ros::init(argc, argv, "sim2real");
    // ros::Time::init();
    ros::NodeHandle nh;
    // ros::spin();
    std::thread exec(&executable);

    ros::spin();

    return 0;
}
