#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <geometry_msgs/msg/twist.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>
#include <string>
#include <vector>

// ONNX Runtime
#include <onnxruntime_cxx_api.h>

// Unitree messages
#include <unitree_go/msg/low_cmd.hpp>
#include <unitree_go/msg/low_state.hpp>
#include <unitree_go/msg/sport_mode_state.hpp>
#include <unitree_hg/msg/imu_state.hpp>
#include <unitree_hg/msg/low_cmd.hpp>
#include <unitree_hg/msg/low_state.hpp>

// Common headers
#include "builtin_interfaces/msg/time.hpp"
#include "common/gamepad.hpp"
#include "common/motor_crc_hg.h"
// Local headers
#include "textop_ctrl/command_helper.hpp"
#include "textop_ctrl/config_loader.hpp"
#include "textop_ctrl/motion_loader.hpp"
#include "textop_ctrl/observation_computer.hpp"
#include "textop_ctrl/onnx_policy.hpp"
#include "textop_ctrl/remote_controller.hpp"
#include "textop_ctrl/rotation_helper.hpp"

const int G1_NUM_MOTOR = 29;

class TextOpOnnxController : public rclcpp::Node
{
   public:
    TextOpOnnxController() : Node("textop_onnx_controller")
    {
        // Load configuration
        std::string config_path =
            this->declare_parameter("config_path", ".../src/textop_ctrl/config/g1_29dof.yaml");
        std::string onnx_model_path =
            this->declare_parameter("onnx_model_path", ".../src/textop_ctrl/models/policy.onnx");
        std::string sportmode_topic = this->declare_parameter("sportmode_topic", "/odommodestate");
        std::string mjc_reset_topic = this->declare_parameter("mjc_reset_topic", "/mjc/reset");

        RCLCPP_INFO(this->get_logger(), "Loading config from: %s", config_path.c_str());
        RCLCPP_INFO(this->get_logger(), "Loading ONNX model from: %s", onnx_model_path.c_str());

        config_ = std::make_unique<Config>(config_path);

        // Initialize ONNX policy
        policy_ = std::make_unique<ONNXPolicy>(onnx_model_path);

        // Initialize motion loader with topic-based data loading
        std::string motion_topic = this->declare_parameter("motion_topic", "/dar/motion");
        RCLCPP_INFO(this->get_logger(), "Subscribing to motion topic: %s", motion_topic.c_str());
        motion_loader_ = std::make_shared<MotionLoader>(this, motion_topic);

        // Initialize observation computer
        observation_computer_ = std::make_unique<ObservationComputer>(motion_loader_, this);

        // Initialize state variables
        action_.resize(config_->num_actions, 0.0f);  // IsaacLab order
        obs_.resize(config_->num_obs, 0.0f);         // IsaacLab order
        target_dof_pos_.resize(29, 0.0f);            // MuJoCo order
        counter_ = 0;
        motion_t_ = 0;
        lock_t_ = false;

        // Initialize target positions and last action with default angles

        const auto& isaaclab_to_mujoco =
            observation_computer_->get_isaaclab_to_mujoco_reindex();
        for (int i = 0; i < 29; ++i)
        {
            target_dof_pos_[i] =
                static_cast<float>(start_pose[i]);  // MuJoCo order
            // target_dof_pos_[i] =
            //     static_cast<float>(observation_computer_->default_angles_[i]);  // MuJoCo order


            int isaaclab_idx = isaaclab_to_mujoco[i];

            action_[isaaclab_idx] = (start_pose[i] - static_cast<float>(observation_computer_->default_angles_[i]))/static_cast<float>(config_->action_scale[i]);
            // static_cast<float>(config_->action_scale[i]) +
                    // static_cast<float>(observation_computer_->default_angles_[i])
            // std::cout << "action[isaaclab_idx]: " << gamepad_.lx << std::endl;

            // action_.resize(config_->num_actions, 0.0f);  // IsaacLab order
        }

        // Initialize low command - make sure all 35 motors are initialized
        almi_ctrl::init_cmd_hg(low_cmd_, mode_machine_, mode_pr_);  // MuJoCo order

        // Create publishers and subscribers
        lowcmd_publisher_ =
            this->create_publisher<unitree_hg::msg::LowCmd>(config_->lowcmd_topic, 10);

        lowstate_subscriber_ = this->create_subscription<unitree_hg::msg::LowState>(
            config_->lowstate_topic, 10, [this](unitree_hg::msg::LowState::SharedPtr message)
            { this->LowStateHandler(message); });

        // Subscribe odometry (sport mode state) for position and velocity in odom frame
        sportmode_subscriber_ = this->create_subscription<unitree_go::msg::SportModeState>(
            sportmode_topic, 10,
            [this](unitree_go::msg::SportModeState::SharedPtr msg)
            {
                std::array<float, 3> pos{0,0,0};
                // std::array<float, 3> pos{msg->position[0], msg->position[1], msg->position[2]};
                // std::array<float, 3> vel{0,0,0};
                std::array<float, 3> vel{msg->velocity[0], msg->velocity[1], msg->velocity[2]};
                // std::cout << "DEBUG: Received odom pos: [" << pos[0] << ", " << pos[1] << ", " << pos[2]
                //           << "], \tvel: [" << vel[0] << ", " << vel[1] << ", " << vel[2] << "]"
                //           << std::endl;
                if (observation_computer_)
                {
                    observation_computer_->set_odometry(pos, vel);
                }
            });
        mjc_reset_publisher_ =
            this->create_publisher<builtin_interfaces::msg::Time>(mjc_reset_topic, 10);

        // Create DAR toggle publisher
        std::string dar_toggle_topic = this->declare_parameter("dar_toggle_topic", "/dar/toggle");
        dar_toggle_publisher_ =
            this->create_publisher<builtin_interfaces::msg::Time>(dar_toggle_topic, 10);

        // Create timer for control loop
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(config_->control_dt * 1000)),
            [this]() { this->Control(); });

        RCLCPP_INFO(this->get_logger(),
                    "TextOp ONNX Controller initialized successfully for 29DOF robot");
    }

   private:
    void LowStateHandler(unitree_hg::msg::LowState::SharedPtr message)
    {
        // Simply update the latest state
        latest_low_state_ = message;
        // gamepad_rx.RF_RX = (unitree::common::xRockerBtnDataStruct)(message->wireless_remote);

        memcpy(gamepad_rx.buff, message->wireless_remote.data(), 40);  // NOLINT
        gamepad_.update(gamepad_rx.RF_RX);
        tick_ = message->tick;

        // Check for gamepad B button press to trigger shutdown
        if (gamepad_.B.pressed)
        {
            shutdown_requested_.store(true);
            RCLCPP_INFO(this->get_logger(), "Shutdown requested - stopping control loop...");
            rclcpp::shutdown();
            return;
        }

        // Check for gamepad Y button - LockXY mode: only active when Y is held down
        if (observation_computer_)
        {
            observation_computer_->set_lock_xy_mode(gamepad_.Y.pressed);
        }

        // Check for gamepad X button - toggle lock_t_
        if (gamepad_.X.pressed)
        {
            if (lock_t_)
            {
                lock_t_ = false;
                RCLCPP_INFO(this->get_logger(), "Lock_t toggled to: %s",
                            lock_t_ ? "true" : "false");
                motion_t_++;
            }
        }

        // Example Usage:
        // std::cout << "gamepad_.lx: " << gamepad_.lx << std::endl;
        // std::cout << "gamepad_.ly: " << gamepad_.ly << std::endl;
        // std::cout << "gamepad_.rx: " << gamepad_.rx << std::endl;
        // std::cout << "gamepad_.ry: " << gamepad_.ry << std::endl;
        // std::cout << "gamepad_.R1: " << gamepad_.R1.pressed << std::endl;
        // std::cout << "gamepad_.L1: " << gamepad_.L1.pressed << std::endl;
        // std::cout << "LowStateHandler:" << tick_ << std::endl;
    }

    std::vector<float> start_pose = {
        -0.2f, 0.0,0.0,0.42f,-0.23f,0.0f,
        -0.2f, 0.0,0.0,0.42f,-0.23f,0.0f,
        0.0f, 0.0, 0.0,
        0.0, 0.2f, 0.15f, 1.2f, 0.0f, 0.0f, 0.0f,
        0.0, -0.2f, -0.15f, 1.2f, 0.0f, 0.0f, 0.0f, 
    };
    std::vector<float> start_kp ={
        400,400,400,400,400,400,
        400,400,400,400,400,400,
        400,400,400,
        20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20,
    };
    std::vector<float> start_kd ={
        10,10,10,10,10,10,
        10,10,10,10,10,10,
        10,10,10,
        0.5,0.5,0.5,0.5,0.5,0.5,0.5,
        0.5,0.5,0.5,0.5,0.5,0.5,0.5,   
    };



    bool CheckInitialize()
    {
        if (is_initialized_ == 4)
        {
            return true;
        }
        static int _init_count = 0;
        // std::cout << "is_initialized_: " << is_initialized_ << std::endl;

        // Initialize

        // For mujoco: we additionally send a reset msg
        auto now = this->get_clock()->now();
        builtin_interfaces::msg::Time t;
        t.sec = static_cast<int32_t>(now.seconds());                          // 秒
        t.nanosec = static_cast<uint32_t>(now.nanoseconds() % 1000000000LL);  // 纳秒余数
        mjc_reset_publisher_->publish(t);

        if (is_initialized_ == 0 && latest_low_state_)
        {
            is_initialized_ = 1;
            std::cout << "In [Zero Torque] Mode, Press 'Start' to continue " << std::endl;
        }
        else if (!latest_low_state_)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "No low state received yet");
            return false;
        }

        if ((is_initialized_ == 1) && (gamepad_.start.pressed))
        {
            is_initialized_ = 2;
            _init_count = 0;
            std::cout << "In [Damping] Mode, Press 'A' to continue " << std::endl;
        }
        if ((is_initialized_ == 1) && (!gamepad_.start.pressed))
        {
            // Update low command with target positions and gains
            for (int i = 0; i < 29; ++i)
            {
                int motor_idx = i;
                // i: MuJoCo order
                if (motor_idx < 35)
                {  // Ensure valid motor index
                    low_cmd_.motor_cmd[motor_idx].q = 0.0f;
                    low_cmd_.motor_cmd[motor_idx].dq = 0.0f;   // Target velocity
                    low_cmd_.motor_cmd[motor_idx].tau = 0.0f;  // Will be calculated by PD control
                    low_cmd_.motor_cmd[motor_idx].kp = 0.0f;
                    low_cmd_.motor_cmd[motor_idx].kd =
                        0.01f;  // leave a small tau to identify the status.
                }
            }

            // Calculate CRC and publish
            get_crc(low_cmd_);
            lowcmd_publisher_->publish(low_cmd_);
        }

        if ((is_initialized_ == 2))
        {
            if (gamepad_.A.pressed)
            {
                is_initialized_ = 3;

                // Send DAR toggle message to activate DAR
                auto now = this->get_clock()->now();
                builtin_interfaces::msg::Time toggle_msg;
                toggle_msg.sec = static_cast<int32_t>(now.seconds());
                toggle_msg.nanosec = static_cast<uint32_t>(now.nanoseconds() % 1000000000LL);
                dar_toggle_publisher_->publish(toggle_msg);
                RCLCPP_INFO(this->get_logger(), "DAR toggle message sent to activate DAR");
            }
            _init_count++;
            float alpha = std::min(_init_count / 100.0f, 1.0f);
            // std::cout << "Damping... " << _init_count << "/" << alpha << std::endl;
            for (int i = 0; i < 29; ++i)
            {
                target_dof_pos_[i] = latest_low_state_->motor_state[i].q * (1 - alpha) +
                                     static_cast<float>(start_pose[i]) *
                                         alpha;  // MuJoCo order
                // target_dof_pos_[i] = latest_low_state_->motor_state[i].q * (1 - alpha) +
                //                      static_cast<float>(observation_computer_->default_angles_[i]) *
                //                          alpha;  // MuJoCo order
            }

            send_motor_commands_start_pose();
        }
        if ((is_initialized_ == 3))
        {
            // Check if motion data is available
            if (!motion_loader_->has_motion_data())
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                     "Waiting for motion data from topic...");
            }
            else
            {
                is_initialized_ = 4;
            }
            _init_count++;
            float alpha = std::min(_init_count / 100.0f, 1.0f);
            // std::cout << "Damping... " << _init_count << "/" << alpha << std::endl;
            for (int i = 0; i < 29; ++i)
            {
                target_dof_pos_[i] = latest_low_state_->motor_state[i].q * (1 - alpha) +
                                     static_cast<float>(start_pose[i]) *
                                         alpha;  // MuJoCo order
            }

            send_motor_commands_start_pose();
        }

        return false;
    }

    void Control()
    {
        try
        {
            // Check if shutdown has been requested
            if (shutdown_requested_.load())
            {
                RCLCPP_INFO(this->get_logger(), "Shutdown requested - stopping control loop...");
                safe_exit();
                return;
            }

            if (!CheckInitialize())
            {
                return;
            }

            // Create observation
            obs_ = observation_computer_->compute_observation(*latest_low_state_, motion_t_,
                                                              action_);  // IsaacLab order

            // Run ONNX inference
            action_ = policy_->predict(obs_);
            // action_: isaaclab order

            // Transform action to target_dof_pos using isaaclab_to_mujoco_reindex
            // and apply action scaling - CORRECT VERSION
            const auto& isaaclab_to_mujoco =
                observation_computer_->get_isaaclab_to_mujoco_reindex();
            for (int i = 0; i < 29; ++i)
            {
                // Get the isaaclab index for this mujoco joint
                int isaaclab_idx = isaaclab_to_mujoco[i];
                target_dof_pos_[i] =
                    action_[isaaclab_idx] * static_cast<float>(config_->action_scale[i]) +
                    static_cast<float>(observation_computer_->default_angles_[i]);
                // action: IsaacLab order
                // action_scale: MuJoCo order
                // default_angles: MuJoCo order
                // target_dof_pos: MuJoCo order
            }

            // Send motor commands
            send_motor_commands();

            // Update motion time step
            if (!lock_t_)
            {
                motion_t_++;
            }

            if (motion_t_ >= motion_loader_->T)
            {
                motion_t_--;
                lock_t_ = true;
                RCLCPP_INFO(this->get_logger(), "Motion ends. Lock motion_t.");
                //
                // RCLCPP_INFO(this->get_logger(),
                //             "Motion time step reached %d, the end of motion data", motion_t_);
                // safe_exit();
                // motion_t_ = 0;  // Loop back to beginning
            }
            counter_++;
        }
        catch (const std::bad_alloc& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Memory allocation failed in Control(): %s", e.what());
            safe_exit();
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Exception in Control(): %s", e.what());
            safe_exit();
        }
        catch (...)
        {
            RCLCPP_ERROR(this->get_logger(), "Unknown exception in Control()");
            safe_exit();
        }
    }

    void send_motor_commands_start_pose()
    {
        // Safety check before sending commands
        if (!sanity_check())
        {
            return;  // Exit early if safety check fails
        }

        // Update low command with target positions and gains
        for (int i = 0; i < 29; ++i)
        {
            int motor_idx = i;
            // i: MuJoCo order
            if (motor_idx < 35)
            {  // Ensure valid motor index
                low_cmd_.motor_cmd[motor_idx].q = target_dof_pos_[i];
                low_cmd_.motor_cmd[motor_idx].dq = 0.0f;   // Target velocity
                low_cmd_.motor_cmd[motor_idx].tau = 0.0f;  // Will be calculated by PD control
                low_cmd_.motor_cmd[motor_idx].kp = static_cast<float>(start_kp[i]);
                low_cmd_.motor_cmd[motor_idx].kd = static_cast<float>(start_kd[i]);
            }
        }

        // Calculate CRC and publish
        get_crc(low_cmd_);
        lowcmd_publisher_->publish(low_cmd_);
    }

    void send_motor_commands()
    {
        // Safety check before sending commands
        if (!sanity_check())
        {
            return;  // Exit early if safety check fails
        }

        // Update low command with target positions and gains
        for (int i = 0; i < 29; ++i)
        {
            int motor_idx = i;
            // i: MuJoCo order
            if (motor_idx < 35)
            {  // Ensure valid motor index
                low_cmd_.motor_cmd[motor_idx].q = target_dof_pos_[i];
                low_cmd_.motor_cmd[motor_idx].dq = 0.0f;   // Target velocity
                low_cmd_.motor_cmd[motor_idx].tau = 0.0f;  // Will be calculated by PD control
                low_cmd_.motor_cmd[motor_idx].kp = static_cast<float>(config_->kps[i]);
                low_cmd_.motor_cmd[motor_idx].kd = static_cast<float>(config_->kds[i]);
            }
        }

        // Calculate CRC and publish
        get_crc(low_cmd_);
        lowcmd_publisher_->publish(low_cmd_);
    }

    bool sanity_check()
    {
        // Check if shutdown has been requested
        if (shutdown_requested_.load())
        {
            safe_exit();
            return false;
        }

        // Check if we have valid low state data
        if (!latest_low_state_)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "No low state received for safety check");
            return false;
        }

        // Check motor position and velocity limits
        for (int motor_idx = 0; motor_idx < 29; ++motor_idx)
        {
            float target_q = low_cmd_.motor_cmd[motor_idx].q;
            float current_q = latest_low_state_->motor_state[motor_idx].q;
            float current_dq = latest_low_state_->motor_state[motor_idx].dq;

            // Check if position difference is too large
            if (std::abs(target_q - current_q) > 3.0f)
            {
                RCLCPP_ERROR(this->get_logger(),
                             "delta q %d is too large.\n"
                             "target q\t: %f\n"
                             "target dq\t: %f\n"
                             "q\t\t: %f\n"
                             "dq\t\t: %f\n",
                             motor_idx, target_q, low_cmd_.motor_cmd[motor_idx].dq, current_q,
                             current_dq);
                safe_exit();
                return false;
            }

            // Check if velocity is too large
            if (std::abs(current_dq) > 25.0f)
            {
                RCLCPP_ERROR(this->get_logger(),
                             "dq %d is too large.\n"
                             "target q\t: %f\n"
                             "target dq\t: %f\n"
                             "q\t\t: %f\n"
                             "dq\t\t: %f\n",
                             motor_idx, target_q, low_cmd_.motor_cmd[motor_idx].dq, current_q,
                             current_dq);
                safe_exit();
                return false;
            }
            // RCLCPP_INFO(this->get_logger(),
            //             "Motor %d - target q: %.3f, target dq: %.3f, current q: %.3f, current dq:
            //             "
            //             "%.3f",
            //             motor_idx, target_q, low_cmd_.motor_cmd[motor_idx].dq, current_q,
            //             current_dq);
        }

        return true;
    }

    void safe_exit()
    {
        RCLCPP_ERROR(this->get_logger(), "Safe exit triggered - shutting down controller");
        shutdown_requested_.store(true);
        rclcpp::shutdown();
    }

   private:
    // Configuration and policy
    std::unique_ptr<Config> config_;
    std::unique_ptr<ONNXPolicy> policy_;
    std::shared_ptr<MotionLoader> motion_loader_;
    std::unique_ptr<ObservationComputer> observation_computer_;

    // State variables
    std::vector<float> action_;          // IsaacLab order
    std::vector<float> obs_;             // IsaacLab order
    std::vector<float> target_dof_pos_;  // MuJoCo order
    int counter_;
    int motion_t_;
    bool lock_t_;

    // Robot state - simplified, no locks
    unitree_hg::msg::LowCmd low_cmd_;                        // MuJoCo order
    unitree_hg::msg::LowState::SharedPtr latest_low_state_;  // MuJoCo order
    uint8_t mode_machine_ = 5;
    uint8_t mode_pr_ = 0;  // 0 for PR, 1 for AB
    uint32_t tick_ = 0;
    uint32_t is_initialized_ = 0;
    unitree::common::Gamepad gamepad_;
    unitree::common::REMOTE_DATA_RX gamepad_rx;

    // Shutdown control
    std::atomic<bool> shutdown_requested_ = false;

    // ROS2
    rclcpp::Publisher<unitree_hg::msg::LowCmd>::SharedPtr lowcmd_publisher_;
    rclcpp::Subscription<unitree_hg::msg::LowState>::SharedPtr lowstate_subscriber_;
    rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr sportmode_subscriber_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<builtin_interfaces::msg::Time>::SharedPtr mjc_reset_publisher_;
    rclcpp::Publisher<builtin_interfaces::msg::Time>::SharedPtr dar_toggle_publisher_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TextOpOnnxController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
