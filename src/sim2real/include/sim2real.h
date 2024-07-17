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
#include <hardware_interface/imu_sensor_interface.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Joy.h>
#include <sensor_msgs/JointState.h>
#include "sim2real_msg/PosState.h"
#include "sim2real_msg/Target.h"
#include "sim2real_msg/PosUrdfState.h"
#include "sim2real_msg/TargetUrdf.h"
#include <yaml-cpp/yaml.h>
#define CONFIG_PATH CMAKE_CURRENT_SOURCE_DIR "/config.yaml"
ros::Publisher cmd_pub;
ros::Publisher rev_pub;

using namespace std;

enum class sys_state{
    close=0,
    init=1,
    exec=2,
    exit=3
};

struct ModelParams{
    std::string model_name;
    double dt;
    int decimation;
    int num_observations;
    double damping;
    double stiffness;
    double action_scale;
    double hip_scale_reduction;
    std::vector<int> hip_scale_reduction_indices;
    int num_of_dofs;
    double lin_vel_scale;
    double ang_vel_scale;
    double dof_pos_scale;
    double dof_vel_scale;
    double clip_obs;
    torch::Tensor clip_actions_upper;
    torch::Tensor clip_actions_lower;
    torch::Tensor torque_limits;
    torch::Tensor rl_kd;
    torch::Tensor rl_kp;
    torch::Tensor fixed_kp;
    torch::Tensor fixed_kd;
    torch::Tensor commands_scale;
    torch::Tensor default_dof_pos;
    torch::Tensor init_dof_pos;
    std::vector<std::string> joint_controller_names;
};

class Robot
{
public:
    Robot(std::string _policy_path):n()
    {
        readYaml();
        cfg.policy_path = _policy_path;
        for (int i = 0; i < cfg.frame_stack; ++i) {
            hist_obs.push_back(Eigen::VectorXd::Zero(cfg.num_single_obs));
        }
        policy = torch::jit::load(cfg.policy_path);
        // init hardware interface
    ros::NodeHandle nh;
    odom_sub_ = nh.subscribe<sensor_msgs::Imu>("/imu/data", 1, &Robot::OdomCallBack_, this);
    joy_sub = nh.subscribe<sensor_msgs::Joy>("joy", 10,&Robot::joy_callback, this);
    pos_pub = n.advertise<sim2real_msg::PosState>("pos_state", 100);
    target_pub = n.advertise<sim2real_msg::Target>("target_state", 100);
    urdf_pos_pub = n.advertise<sim2real_msg::PosUrdfState>("urdf_pos_state", 100);
    urdf_target_pub = n.advertise<sim2real_msg::TargetUrdf>("urdf_target_state", 100);
    rviz_target_pub = n.advertise<sensor_msgs::JointState>("/joint_states", 100);     
    rviz_pos_pub = n.advertise<sensor_msgs::JointState>("/pos_joint_states", 100);
    std::vector<std::string> joint_names={
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint"};
    rviz_target_pos.name = joint_names; 
    rviz_target_pos.position.resize(12); // Resize positions array to 12

    dynamic_offset_hip = 0.0;
    dynamic_offset_knee = 0.0;
    dynamic_offset_ankle = 0.0;

    setupImu();

    }
    ~Robot(){

    }

    void readYaml();
    bool setupImu();
    void rbt_init();
    void ankle_kinematics(double theta_1, double theta_2, double &theta_p, double &theta_r);
    void ankle_ikinematics(double theta_p, double theta_r, double &theta_1, double &theta_2);
    void update_obs();
    void quaternionToEulerArray();
    void update_action();
    void read();
    void write();
    void run();
    void joy_callback(const sensor_msgs::Joy::ConstPtr& msg);
    sys_state get_state(){
        return state;
    }
    ModelParams params;
private:
    ros::Subscriber joy_sub;
    sys_state state=sys_state::close;
    struct Config{
        std::string policy_path;
        int num_actions = 12, num_single_obs = 47, frame_stack = 15;
        float action_scales = 0.25, tau_limit = 15.0 , clip_observations = 18.0, clip_actions = 18.0, dt = 0.001;
        float lin_vel_scale = 2.0, ang_vel_scale = 1.0, dof_pos_scale = 1.0, dof_vel_scale = 0.05;

        int map_index[12] = {5, 4, 3, 2, 1, 0,  11, 10, 9, 8, 7, 6};
        // float motor_direction[12] = {-1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0};
        float motor_direction[12] = {-1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
                                     -1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    } cfg;
    struct EncosImuData
    {       
        double ori[4];
        double ori_cov[9];
        double angular_vel[3];
        double angular_vel_cov[9];
        double linear_acc[3];
        double linear_acc_cov[9];
    };
    struct Command{
        float vx = 1.0, vy = 0.0, dyaw = 0.0;
    } cmd;
    ros::NodeHandle n;
    livelybot_serial::robot rb;
    torch::jit::script::Module policy;
    int count_lowlevel=0;
    Eigen::VectorXd q = Eigen::VectorXd::Zero(cfg.num_actions);
    Eigen::VectorXd dq = Eigen::VectorXd::Zero(cfg.num_actions);
    Eigen::VectorXd obs = Eigen::VectorXd::Zero(cfg.num_single_obs);
    Eigen::VectorXd target_q = Eigen::VectorXd::Zero(cfg.num_actions);
    Eigen::VectorXd target_q_record = Eigen::VectorXd::Zero(cfg.num_actions);
    Eigen::VectorXd cmd_q = Eigen::VectorXd::Zero(cfg.num_actions);
    Eigen::VectorXd action = Eigen::VectorXd::Zero(cfg.num_actions);
    std::deque<Eigen::VectorXd> hist_obs;
    float input[1][15*47] = {};
    Eigen::Quaterniond quat;
    Eigen::Quaterniond quat_record;
    Eigen::Vector3d eu_ang, base_ang_vel;
    hardware_interface::ImuSensorInterface imuSensorInterface_; 
    EncosImuData imuData_{};
    ros::Subscriber odom_sub_;
    ros::Publisher pos_pub;
    ros::Publisher target_pub;
    ros::Publisher urdf_pos_pub;
    ros::Publisher urdf_target_pub;
    ros::Publisher rviz_target_pub;  
    ros::Publisher rviz_pos_pub;     
    sensor_msgs::Imu yesenceIMU_;
    sensor_msgs::JointState rviz_target_pos;
    sensor_msgs::JointState rviz__pos;
    sim2real_msg::PosState pos_msg;
    sim2real_msg::Target target_msg;
    sim2real_msg::PosUrdfState urdf_pos_state;
    sim2real_msg::TargetUrdf urdf_target_state;
    float dynamic_offset_hip;
    float dynamic_offset_knee;
    float dynamic_offset_ankle;

    void OdomCallBack_(const sensor_msgs::Imu::ConstPtr &odom)
    {
        // ROS_INFO("OdomCallBack_");
      yesenceIMU_ = *odom;
    };



};