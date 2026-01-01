#pragma once

#include <mujoco/mujoco.h>
#include <unitree/dds_wrapper/robots/g1/g1.h>
#include <unitree/dds_wrapper/robots/go2/go2.h>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "param.h"
#include "physics_joystick.h"
#include "reset_dds_wrapper.hpp"

#define MOTOR_SENSOR_NUM 3

class UnitreeSDK2BridgeBase
{
   public:
    UnitreeSDK2BridgeBase(mjModel *model, mjData *data) : mj_model_(model), mj_data_(data)
    {
        _check_sensor();
        if (param::config.print_scene_information == 1)
        {
            printSceneInformation();
        }
        if (param::config.use_joystick == 1)
        {
            if (param::config.joystick_type == "xbox")
            {
                joystick = std::make_shared<XBoxJoystick>(param::config.joystick_device,
                                                          param::config.joystick_bits);
                // std::cout << "XBoxJoystick created" << std::endl;
            }
            else if (param::config.joystick_type == "switch")
            {
                joystick = std::make_shared<SwitchJoystick>(param::config.joystick_device,
                                                            param::config.joystick_bits);
            }
            else if (param::config.joystick_type == "keyboard")
            {
                joystick = std::make_shared<EventKeyboardJoystick>(
                    param::config.joystick_device   // e.g.: /dev/input/event3
                );
            }
            else
            {
                std::cerr << "Unsupported joystick type: " << param::config.joystick_type
                          << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    void printSceneInformation()
    {
        auto printObjects = [this](const char *title, int count, int type, auto getIndex)
        {
            std::cout << "<<------------- " << title << " ------------->> " << std::endl;
            for (int i = 0; i < count; i++)
            {
                const char *name = mj_id2name(mj_model_, type, i);
                if (name)
                {
                    std::cout << title << "_index: " << getIndex(i) << ", " << "name: " << name;
                    if (type == mjOBJ_SENSOR)
                    {
                        std::cout << ", dim: " << mj_model_->sensor_dim[i];
                    }
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl;
        };

        printObjects("Link", mj_model_->nbody, mjOBJ_BODY, [](int i) { return i; });
        printObjects("Joint", mj_model_->njnt, mjOBJ_JOINT, [](int i) { return i; });
        printObjects("Actuator", mj_model_->nu, mjOBJ_ACTUATOR, [](int i) { return i; });

        int sensorIndex = 0;
        printObjects("Sensor", mj_model_->nsensor, mjOBJ_SENSOR,
                     [&](int i)
                     {
                         int currentIndex = sensorIndex;
                         sensorIndex += mj_model_->sensor_dim[i];
                         return currentIndex;
                     });
    }

   protected:
    int num_motor_ = 0;
    int dim_motor_sensor_ = 0;

    mjData *mj_data_;
    mjModel *mj_model_;

    int have_imu_ = false;
    int have_frame_sensor_ = false;

    std::shared_ptr<unitree::common::UnitreeJoystick> joystick = nullptr;

    void _check_sensor()
    {
        num_motor_ = mj_model_->nu;
        dim_motor_sensor_ = MOTOR_SENSOR_NUM * num_motor_;

        for (int i = dim_motor_sensor_; i < mj_model_->nsensor; i++)
        {
            const char *name = mj_id2name(mj_model_, mjOBJ_SENSOR, i);
            if (strcmp(name, "imu_quat") == 0)
            {
                have_imu_ = true;
            }
            if (strcmp(name, "frame_pos") == 0)
            {
                have_frame_sensor_ = true;
            }
        }
    }
};

template <typename LowCmd_t, typename LowState_t, typename Reset_t>
class RobotBridge : public UnitreeSDK2BridgeBase
{
    using HighState_t = unitree::robot::go2::publisher::SportModeState;
    using WirelessController_t = unitree::robot::go2::publisher::WirelessController;

   public:
    RobotBridge(mjModel *model, mjData *data) : UnitreeSDK2BridgeBase(model, data)
    {
        lowcmd = std::make_shared<LowCmd_t>("rt/lowcmd");
        reset = std::make_shared<Reset_t>("rt/mjc/reset");
        reset->set_timeout_ms(50);
        lowstate = std::make_unique<LowState_t>();
        lowstate->joystick = joystick;
        highstate = std::make_unique<HighState_t>("rt/odommodestate");
        wireless_controller = std::make_unique<WirelessController_t>();
        wireless_controller->joystick = joystick;
        thread_ = std::make_shared<unitree::common::RecurrentThread>(
            "unitree_bridge", UT_CPU_ID_NONE, 1000, std::bind(&RobotBridge::run, this));
    }

    void run()
    {
        if (!mj_data_)
            return;
        if (lowstate->joystick)
        {
            lowstate->joystick->update();
        }
        // lowcmd
        {
            std::lock_guard<std::mutex> lock(lowcmd->mutex_);

            if (lowcmd->isTimeout())
            {
                for (int i = 0; i < mj_model_->nu; i++)
                {
                    mj_data_->ctrl[i] = 0;
                }
            }
            else
            {
                for (int i(0); i < num_motor_; i++)
                {
                    auto &m = lowcmd->msg_.motor_cmd()[i];
                    mj_data_->ctrl[i] = m.tau() + m.kp() * (m.q() - mj_data_->sensordata[i]) +
                                        m.kd() * (m.dq() - mj_data_->sensordata[i + num_motor_]);
                }
            }
        }
        // reset
        {
            std::lock_guard<std::mutex> lock(reset->mutex_);
            auto new_sec = reset->msg_.sec();
            auto new_nanosec = reset->msg_.nanosec();

            bool new_reset = (new_sec != last_reset_sec_ || new_nanosec != last_reset_nanosec_);

            double time_changed = (double)new_sec + (double)new_nanosec * 1e-9 -
                                  ((double)last_reset_sec_ + (double)last_reset_nanosec_ * 1e-9);

            // std::cout << "time_changed: " << time_changed << std::endl;
            // std::cout << "new_reset: " << new_reset << std::endl;

            if (new_reset && (time_changed > 0.1))
            {
                // 更新记录的时间
                last_reset_sec_ = new_sec;
                last_reset_nanosec_ = new_nanosec;

                // 执行reset操作
                for (int i = 0; i < mj_model_->nu; i++)
                {
                    mj_data_->ctrl[i] = 0;
                }
                for (int i = 0; i < mj_model_->nv; i++)
                {
                    mj_data_->qvel[i] = 0;
                    // mj_data_->sensordata[i + num_motor_] = 0;
                    // mj_data_->sensordata[i + 2 * num_motor_] = 0;
                }
                for (int i = 0; i < mj_model_->nq; i++)
                {
                    mj_data_->qpos[i] = 0;
                    // mj_data_->sensordata[i] = 0;
                }
                mj_data_->qpos[2] = 0.79;
                mj_data_->qpos[3] = 1.0;
                // mj_forward(mj_model_, mj_data_);

                // mj_resetData(mj_model_, mj_data_);

                std::cout << "reset done at time: " << last_reset_sec_ << "s "
                          << last_reset_nanosec_ << "ns" << std::endl;
            }
        }

        // lowstate
        if (lowstate->trylock())
        {
            lowstate->msg_.tick() += 1;
            // std::cout << "lowstate tick: " << lowstate->msg_.tick() << std::endl;
            for (int i(0); i < num_motor_; i++)
            {
                lowstate->msg_.motor_state()[i].q() = mj_data_->sensordata[i];
                lowstate->msg_.motor_state()[i].dq() = mj_data_->sensordata[i + num_motor_];
                lowstate->msg_.motor_state()[i].tau_est() =
                    mj_data_->sensordata[i + 2 * num_motor_];
            }
            if (have_frame_sensor_)
            {
                lowstate->msg_.imu_state().quaternion()[0] =
                    mj_data_->sensordata[dim_motor_sensor_ + 0];
                lowstate->msg_.imu_state().quaternion()[1] =
                    mj_data_->sensordata[dim_motor_sensor_ + 1];
                lowstate->msg_.imu_state().quaternion()[2] =
                    mj_data_->sensordata[dim_motor_sensor_ + 2];
                lowstate->msg_.imu_state().quaternion()[3] =
                    mj_data_->sensordata[dim_motor_sensor_ + 3];

                double w = lowstate->msg_.imu_state().quaternion()[0];
                double x = lowstate->msg_.imu_state().quaternion()[1];
                double y = lowstate->msg_.imu_state().quaternion()[2];
                double z = lowstate->msg_.imu_state().quaternion()[3];

                lowstate->msg_.imu_state().rpy()[0] =
                    atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
                lowstate->msg_.imu_state().rpy()[1] = asin(2 * (w * y - z * x));
                lowstate->msg_.imu_state().rpy()[2] =
                    atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));

                lowstate->msg_.imu_state().gyroscope()[0] =
                    mj_data_->sensordata[dim_motor_sensor_ + 4];
                lowstate->msg_.imu_state().gyroscope()[1] =
                    mj_data_->sensordata[dim_motor_sensor_ + 5];
                lowstate->msg_.imu_state().gyroscope()[2] =
                    mj_data_->sensordata[dim_motor_sensor_ + 6];

                lowstate->msg_.imu_state().accelerometer()[0] =
                    mj_data_->sensordata[dim_motor_sensor_ + 7];
                lowstate->msg_.imu_state().accelerometer()[1] =
                    mj_data_->sensordata[dim_motor_sensor_ + 8];
                lowstate->msg_.imu_state().accelerometer()[2] =
                    mj_data_->sensordata[dim_motor_sensor_ + 9];
            }

            // Sync wireless_controller data to lowstate using the correct method
            if (lowstate->joystick)
            {
                unitree::common::REMOTE_DATA_RX key = lowstate->joystick->combine();
                memcpy(&lowstate->msg_.wireless_remote()[0], &key,
                       sizeof(unitree::common::REMOTE_DATA_RX));
            }

            lowstate->unlockAndPublish();
        }
        // highstate
        if (have_frame_sensor_ && highstate->trylock())
        {
            highstate->msg_.position()[0] = mj_data_->sensordata[dim_motor_sensor_ + 10];
            highstate->msg_.position()[1] = mj_data_->sensordata[dim_motor_sensor_ + 11];
            highstate->msg_.position()[2] = mj_data_->sensordata[dim_motor_sensor_ + 12];
            // world frame pos
            highstate->msg_.velocity()[0] = mj_data_->sensordata[dim_motor_sensor_ + 13];
            highstate->msg_.velocity()[1] = mj_data_->sensordata[dim_motor_sensor_ + 14];
            highstate->msg_.velocity()[2] = mj_data_->sensordata[dim_motor_sensor_ + 15];
            // world frame linvel
            highstate->unlockAndPublish();
        }
        // wireless_controller
        if (wireless_controller->joystick)
        {
            wireless_controller->unlockAndPublish();
        }
    }

    std::unique_ptr<HighState_t> highstate;
    std::unique_ptr<WirelessController_t> wireless_controller;
    std::shared_ptr<LowCmd_t> lowcmd;
    std::unique_ptr<LowState_t> lowstate;
    std::shared_ptr<Reset_t> reset;

   private:
    unitree::common::RecurrentThreadPtr thread_;
    // 添加成员变量记录上一次reset的时间
    int32_t last_reset_sec_ = 0;
    uint32_t last_reset_nanosec_ = 0;
};

using Go2Bridge = RobotBridge<unitree::robot::go2::subscription::LowCmd,
                              unitree::robot::go2::publisher::LowState, ResetSubs>;
using G1Bridge = RobotBridge<unitree::robot::g1::subscription::LowCmd,
                             unitree::robot::g1::publisher::LowState, ResetSubs>;
