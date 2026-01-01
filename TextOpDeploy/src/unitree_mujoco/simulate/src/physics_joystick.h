#pragma once

#include <iostream>
#include <memory>

#include <linux/input.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <cstring>
#include <unordered_map>

#include <unitree/dds_wrapper/common/unitree_joystick.hpp>

#include "joystick/joystick.h"

class XBoxJoystick : public unitree::common::UnitreeJoystick
{
   public:
    XBoxJoystick(std::string device, int bits = 15) : unitree::common::UnitreeJoystick()
    {
        js_ = std::make_unique<Joystick>(device);
        if (!js_->isFound())
        {
            std::cout << "Error: Joystick open failed." << std::endl;
            exit(1);
        }
        max_value_ = 1 << (bits - 1);
    }

    void update() override
    {
        js_->getState();
        back(js_->button_[6]);
        start(js_->button_[7]);
        LB(js_->button_[4]);
        RB(js_->button_[5]);
        A(js_->button_[0]);
        B(js_->button_[1]);
        X(js_->button_[2]);
        Y(js_->button_[3]);
        up(js_->axis_[7] < 0);
        down(js_->axis_[7] > 0);
        left(js_->axis_[6] < 0);
        right(js_->axis_[6] > 0);
        LT(js_->axis_[2] > 0);
        RT(js_->axis_[5] > 0);
        lx(double(js_->axis_[0]) / max_value_);
        ly(-double(js_->axis_[1]) / max_value_);
        rx(double(js_->axis_[3]) / max_value_);
        ry(-double(js_->axis_[4]) / max_value_);
    }

   private:
    std::unique_ptr<Joystick> js_;
    int max_value_;
};

class SwitchJoystick : public unitree::common::UnitreeJoystick
{
   public:
    SwitchJoystick(std::string device, int bits = 15) : unitree::common::UnitreeJoystick()
    {
        js_ = std::make_unique<Joystick>(device);
        if (!js_->isFound())
        {
            std::cout << "Error: Joystick open failed." << std::endl;
            exit(1);
        }
        max_value_ = 1 << (bits - 1);
    }

    void update() override
    {
        js_->getState();
        back(js_->button_[10]);
        start(js_->button_[11]);
        LB(js_->button_[6]);
        RB(js_->button_[7]);
        A(js_->button_[0]);
        B(js_->button_[1]);
        X(js_->button_[3]);
        Y(js_->button_[4]);
        up(js_->axis_[7] < 0);
        down(js_->axis_[7] > 0);
        left(js_->axis_[6] < 0);
        right(js_->axis_[6] > 0);
        LT(js_->axis_[5] > 0);
        RT(js_->axis_[4] > 0);
        lx(double(js_->axis_[0]) / max_value_);
        ly(-double(js_->axis_[1]) / max_value_);
        rx(double(js_->axis_[2]) / max_value_);
        ry(-double(js_->axis_[3]) / max_value_);
    }

   private:
    std::unique_ptr<Joystick> js_;
    int max_value_;
};


class EventKeyboardJoystick : public unitree::common::UnitreeJoystick
{
public:
    explicit EventKeyboardJoystick(const std::string& device)
        : unitree::common::UnitreeJoystick()
    {
        fd_ = open(device.c_str(), O_RDONLY | O_NONBLOCK);
        if (fd_ < 0)
        {
            perror("Failed to open keyboard event device");
            exit(1);
        }

        initKeyMap();

        std::cout << "[EventKeyboardJoystick] Using device: " << device << std::endl;
    }

    ~EventKeyboardJoystick()
    {
        if (fd_ >= 0)
            close(fd_);
    }

    void update() override
    {
        struct input_event ev;
        ssize_t n;

        while ((n = read(fd_, &ev, sizeof(ev))) == sizeof(ev))
        {
            if (ev.type == EV_KEY)
            {
                bool pressed = (ev.value != 0);
                key_state_[ev.code] = pressed;

                // if(ev.code !=0){
                //     std::cout << "[EventKeyboardJoystick] DEBUG: " << ev.code << " <-" << ev.value << std::endl;
                // }
            }
        }


        applyState();
    }

private:
    int fd_;
    std::unordered_map<int, bool> key_state_;

    /* ================== 初始化 ================== */

    void initKeyMap()
    {
        // 所有会用到的 key code 先注册
        int keys[] = {
            KEY_W, KEY_S, KEY_A, KEY_D,
            KEY_I, KEY_K, KEY_J, KEY_L,
            KEY_1, KEY_2, KEY_3, KEY_4,
            KEY_Q, KEY_E,
            KEY_U, KEY_V,
            KEY_SPACE,
            KEY_BACKSPACE
        };

        for (int k : keys)
            key_state_[k] = false;
    }

    /* ================== 映射到 UnitreeJoystick ================== */

    void applyState()
    {
        constexpr double v = 0.8;
        constexpr double w = 1.0;

        // ---- 左摇杆 ----
        lx((key(KEY_A) ?  v : 0.0) + (key(KEY_D) ? -v : 0.0));
        ly((key(KEY_W) ?  v : 0.0) + (key(KEY_S) ? -v : 0.0));

        // ---- 右摇杆 ----
        rx((key(KEY_J) ?  w : 0.0) + (key(KEY_L) ? -w : 0.0));
        ry((key(KEY_I) ?  w : 0.0) + (key(KEY_K) ? -w : 0.0));

        // ---- 方向键语义（可选）----
        up(key(KEY_W));
        down(key(KEY_S));
        left(key(KEY_A));
        right(key(KEY_D));

        // ---- 按钮（完整覆盖）----
        A(key(KEY_1));
        B(key(KEY_2));
        X(key(KEY_3));
        Y(key(KEY_4));

        LB(key(KEY_Q));
        RB(key(KEY_E));

        LT(key(KEY_U));
        RT(key(KEY_V));

        start(key(KEY_SPACE));
        back(key(KEY_BACKSPACE));
    }

    bool key(int code) const
    {
        auto it = key_state_.find(code);
        return it != key_state_.end() && it->second;
    }
};