
#pragma once

#include <unitree/dds_wrapper/common/Publisher.h>
#include <unitree/dds_wrapper/common/Subscription.h>
#include <unitree/dds_wrapper/common/crc.h>

#include <memory>
#include <unitree/idl/ros2/Time_.hpp>

class ResetSubs : public unitree::robot::SubscriptionBase<builtin_interfaces::msg::dds_::Time_>
{
   public:
    using SharedPtr = std::shared_ptr<ResetSubs>;

    ResetSubs(std::string topic = "rt/mjc/reset") : unitree::robot::SubscriptionBase<MsgType>(topic)
    {
    }
};

class ResetPubs : public unitree::robot::RealTimePublisher<builtin_interfaces::msg::dds_::Time_>
{
   public:
    ResetPubs(std::string topic = "rt/mjc/reset")
        : unitree::robot::RealTimePublisher<MsgType>(topic)
    {
    }
};
