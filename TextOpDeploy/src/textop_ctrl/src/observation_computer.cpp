#include "textop_ctrl/observation_computer.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unitree_hg/msg/low_state.hpp>

ObservationComputer::ObservationComputer(std::shared_ptr<MotionLoader> motion_loader,
                                         rclcpp::Node* node)
    : motion_loader_(motion_loader), node_(node)
{
    // Initialize joint reindexing arrays based on Python script
    // isaaclab joint names order
    std::vector<std::string> isaaclab_joint_names = {"left_hip_pitch_joint",
                                                     "right_hip_pitch_joint",
                                                     "waist_yaw_joint",
                                                     "left_hip_roll_joint",
                                                     "right_hip_roll_joint",
                                                     "waist_roll_joint",
                                                     "left_hip_yaw_joint",
                                                     "right_hip_yaw_joint",
                                                     "waist_pitch_joint",
                                                     "left_knee_joint",
                                                     "right_knee_joint",
                                                     "left_shoulder_pitch_joint",
                                                     "right_shoulder_pitch_joint",
                                                     "left_ankle_pitch_joint",
                                                     "right_ankle_pitch_joint",
                                                     "left_shoulder_roll_joint",
                                                     "right_shoulder_roll_joint",
                                                     "left_ankle_roll_joint",
                                                     "right_ankle_roll_joint",
                                                     "left_shoulder_yaw_joint",
                                                     "right_shoulder_yaw_joint",
                                                     "left_elbow_joint",
                                                     "right_elbow_joint",
                                                     "left_wrist_roll_joint",
                                                     "right_wrist_roll_joint",
                                                     "left_wrist_pitch_joint",
                                                     "right_wrist_pitch_joint",
                                                     "left_wrist_yaw_joint",
                                                     "right_wrist_yaw_joint"};

    // mujoco joint names order
    std::vector<std::string> mujoco_joint_names = {
        "left_hip_pitch_joint",      "left_hip_roll_joint",        "left_hip_yaw_joint",
        "left_knee_joint",           "left_ankle_pitch_joint",     "left_ankle_roll_joint",
        "right_hip_pitch_joint",     "right_hip_roll_joint",       "right_hip_yaw_joint",
        "right_knee_joint",          "right_ankle_pitch_joint",    "right_ankle_roll_joint",
        "waist_yaw_joint",           "waist_roll_joint",           "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",   "left_shoulder_yaw_joint",
        "left_elbow_joint",          "left_wrist_roll_joint",      "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",      "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",  "right_elbow_joint",          "right_wrist_roll_joint",
        "right_wrist_pitch_joint",   "right_wrist_yaw_joint"};

    // Create reindexing arrays - CORRECT VERSION
    isaaclab_to_mujoco_reindex_.resize(29);
    mujoco_to_isaaclab_reindex_.resize(29);

    // isaaclab_to_mujoco: for each isaaclab joint, find its mujoco index
    for (int i = 0; i < 29; ++i)
    {
        auto it = std::find(mujoco_joint_names.begin(), mujoco_joint_names.end(),
                            isaaclab_joint_names[i]);
        mujoco_to_isaaclab_reindex_[i] = std::distance(mujoco_joint_names.begin(), it);
    }

    // mujoco_to_isaaclab: for each mujoco joint, find its isaaclab index
    for (int i = 0; i < 29; ++i)
    {
        auto it = std::find(isaaclab_joint_names.begin(), isaaclab_joint_names.end(),
                            mujoco_joint_names[i]);
        isaaclab_to_mujoco_reindex_[i] = std::distance(isaaclab_joint_names.begin(), it);
    }

    // Initialize default joint angles (matching Python default_angles)
    // Based on Python joint_pos_config dictionary - using mujoco order
    default_angles_ = std::vector<float>(29, 0.0f);  // Initialize all to 0.0
    // Mujoco order

    // Apply specific default values based on joint patterns - CORRECT VERSION
    std::vector<std::pair<std::string, float>> joint_defaults = {
        {"left_hip_pitch_joint", -0.312f},    {"right_hip_pitch_joint", -0.312f},
        {"left_knee_joint", 0.669f},          {"right_knee_joint", 0.669f},
        {"left_ankle_pitch_joint", -0.363f},  {"right_ankle_pitch_joint", -0.363f},
        {"left_elbow_joint", 0.6f},           {"right_elbow_joint", 0.6f},
        {"left_shoulder_roll_joint", 0.2f},   {"left_shoulder_pitch_joint", 0.2f},
        {"right_shoulder_roll_joint", -0.2f}, {"right_shoulder_pitch_joint", 0.2f}};

    for (const auto& [joint_name, default_val] : joint_defaults)
    {
        auto it = std::find(mujoco_joint_names.begin(), mujoco_joint_names.end(), joint_name);
        if (it != mujoco_joint_names.end())
        {
            int idx = std::distance(mujoco_joint_names.begin(), it);
            default_angles_[idx] = default_val;
        }
    }

    std::cout << "default_angles_: [" << vector_to_string(default_angles_) << "]" << std::endl;
    std::cout << "mujoco_to_isaaclab_reindex_: [" << vector_to_string(mujoco_to_isaaclab_reindex_)
              << "]" << std::endl;
    std::cout << "isaaclab_to_mujoco_reindex_: [" << vector_to_string(isaaclab_to_mujoco_reindex_)
              << "]" << std::endl;

    // exit(0);
}

std::vector<float> ObservationComputer::compute_observation(
    const unitree_hg::msg::LowState& low_state, int motion_t,
    const std::vector<float>& last_actions)  // IsaacLab order
{
    try
    {
        int obs_dim = 428;  // 290 + 15 + 30 + 3 + 3 + 29 + 29 + 29 = 428 (matching Python)
        auto USE_PROJ_GRAV = true;
        if (USE_PROJ_GRAV)
        {
            obs_dim += 3;
        }
        std::vector<float> obs(obs_dim, 0.0f);

        if (motion_t < 0)
        {
            std::cout << "motion_t < 0, returning zero observation" << std::endl;
            return obs;
        }

        if (motion_t == 0)
        {
            // Set Default Robot World Frame
            setup_init_frame(low_state);
        }

        // Align robot and ref XY coordinates if in LockXY mode
        if (lock_xy_mode_)
        {
            align_robot_ref_xy(motion_t);
        }

        int idx = 0;

        // 0. command (290,) - future joint pos + vel
        std::vector<float> command = get_command(motion_t);  // IsaacLab order
        std::copy(command.begin(), command.end(), obs.begin() + idx);
        // std::cout << "command[" << command.size() << "]: [" << vector_to_string(command) << "]"
        //           << std::endl;
        idx += command.size();

        // 1. motion_anchor_pos_b (15,) - future anchor pos in body frame
        std::vector<float> motion_anchor_pos = motion_anchor_pos_b_future(low_state, motion_t);
        std::copy(motion_anchor_pos.begin(), motion_anchor_pos.end(), obs.begin() + idx);
        // std::cout << "motion_anchor_pos[" << motion_anchor_pos.size() << "]: ["
        //           << vector_to_string(motion_anchor_pos) << "]" << std::endl;
        idx += motion_anchor_pos.size();

        // 2. motion_anchor_ori_b (30,) - future anchor ori in body frame
        std::vector<float> motion_anchor_ori = motion_anchor_ori_b_future(low_state, motion_t);
        std::copy(motion_anchor_ori.begin(), motion_anchor_ori.end(), obs.begin() + idx);
        // std::cout << "motion_anchor_ori[" << motion_anchor_ori.size() << "]: ["
        //           << vector_to_string(motion_anchor_ori) << "]" << std::endl;
        idx += motion_anchor_ori.size();

        if (USE_PROJ_GRAV)
        {
            // 3. projgrav (3,) - projgrav
            std::vector<float> projgrav = get_projected_gravity(low_state);
            std::copy(projgrav.begin(), projgrav.end(), obs.begin() + idx);
            idx += projgrav.size();
        }

        // 3. base_lin_vel (3,) - base linear velocity
        std::vector<float> base_lin_vel = get_base_lin_vel(low_state);
        // TODO: check body frame or world frame
        std::copy(base_lin_vel.begin(), base_lin_vel.end(), obs.begin() + idx);
        // std::cout << "base_lin_vel[" << base_lin_vel.size() << "]: ["
        //           << vector_to_string(base_lin_vel) << "]" << std::endl;
        idx += base_lin_vel.size();

        // 4. base_ang_vel (3,) - base angular velocity
        std::vector<float> base_ang_vel = get_base_ang_vel(low_state);
        // TODO: check body frame or world frame
        std::copy(base_ang_vel.begin(), base_ang_vel.end(), obs.begin() + idx);
        // std::cout << "base_ang_vel[" << base_ang_vel.size() << "]: ["
        //           << vector_to_string(base_ang_vel) << "]" << std::endl;
        idx += base_ang_vel.size();

        // 5. joint_pos (29,) - relative joint positions
        std::vector<float> joint_pos = get_joint_pos_rel(low_state);  // IsaacLab order
        std::copy(joint_pos.begin(), joint_pos.end(), obs.begin() + idx);
        // std::cout << "joint_pos[" << joint_pos.size() << "]: [" << vector_to_string(joint_pos)
        //           << "]" << std::endl;
        idx += joint_pos.size();

        // 6. joint_vel (29,) - relative joint velocities
        std::vector<float> joint_vel = get_joint_vel_rel(low_state);  // IsaacLab order
        std::copy(joint_vel.begin(), joint_vel.end(), obs.begin() + idx);
        // std::cout << "joint_vel[" << joint_vel.size() << "]: [" << vector_to_string(joint_vel)
        //           << "]" << std::endl;
        idx += joint_vel.size();

        // 7. actions (29,) - last actions
        std::vector<float> last_action = get_last_action(last_actions);  // IsaacLab order
        std::copy(last_action.begin(), last_action.end(), obs.begin() + idx);
        // std::cout << "last_action[" << last_action.size() << "]: [" <<
        // vector_to_string(last_action)
        //           << "]" << std::endl;
        idx += last_action.size();

        // Print final observation summary
        // std::cout << "Final observation[" << obs.size() << "]: [" << vector_to_string(obs) << "]"
        //           << std::endl;

        // exit(0);

        return obs;
    }
    catch (const std::bad_alloc& e)
    {
        std::cerr << "Memory allocation failed in compute_observation: " << e.what() << std::endl;
        throw;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception in compute_observation: " << e.what() << std::endl;
        throw;
    }
    catch (...)
    {
        std::cerr << "Unknown exception in compute_observation" << std::endl;
        throw;
    }
}

std::vector<float> ObservationComputer::get_command(int motion_t)
{
    // IsaacLab order
    const int future_steps = motion_loader_->future_steps;
    std::vector<float> cmd(290, 0.0f);  // 29 * 2 * 5 = 290

    if (motion_t < 0)
    {
        return cmd;
    }

    // Get future joint positions and velocities
    std::vector<float> joint_pos_future;
    std::vector<float> joint_vel_future;

    for (int i = 0; i < future_steps; ++i)
    {
        int step_idx = std::min(motion_t + i, motion_loader_->T - 1);

        // Add joint positions
        for (int j = 0; j < 29; ++j)
        {
            joint_pos_future.push_back(motion_loader_->joint_pos[step_idx][j]);
            // IsaacLab order
        }

        // Add joint velocities
        for (int j = 0; j < 29; ++j)
        {
            joint_vel_future.push_back(motion_loader_->joint_vel[step_idx][j]);
            // IsaacLab order
        }
    }

    // Concatenate positions and velocities
    std::copy(joint_pos_future.begin(), joint_pos_future.end(), cmd.begin());
    std::copy(joint_vel_future.begin(), joint_vel_future.end(), cmd.begin() + 145);

    return cmd;
}

std::vector<float> ObservationComputer::motion_anchor_pos_b_future(
    const unitree_hg::msg::LowState& low_state, int motion_t)
{
    std::vector<float> pos_b(15, 0.0f);  // 3 * 5 = 15

    if (motion_t < 0)
    {
        return pos_b;
    }

    // Get robot anchor pose: position from odometry (world), orientation from IMU
    std::array<float, 3> robot_pos = robot_pos_w_;
    std::array<float, 4> robot_quat = {
        low_state.imu_state.quaternion[0], low_state.imu_state.quaternion[1],
        low_state.imu_state.quaternion[2], low_state.imu_state.quaternion[3]};

    // Get future motion anchor poses
    for (int i = 0; i < motion_loader_->future_steps; ++i)
    {
        int step_idx = std::min(motion_t + i, motion_loader_->T - 1);

        std::array<float, 3> ref_pos =
            motion_loader_->body_pos[step_idx][motion_loader_->anchor_body_index];
        std::array<float, 4> ref_quat =
            motion_loader_->body_ori[step_idx][motion_loader_->anchor_body_index];

        auto [ref_pos_b, ref_quat_b] = transform_ref_to_robot_frame(ref_pos, ref_quat);
        // Transform to body frame using proper frame transformation
        auto [pos_b_step, _] =
            subtract_frame_transforms(robot_pos, robot_quat, ref_pos_b, ref_quat_b);

        pos_b[i * 3 + 0] = pos_b_step[0];
        pos_b[i * 3 + 1] = pos_b_step[1];
        pos_b[i * 3 + 2] = pos_b_step[2];

        if(true){
            // std::cout<< "MANUAL ZERO \n";
            pos_b[i * 3 + 0] = 0;
            pos_b[i * 3 + 1] = 0;
            pos_b[i * 3 + 2] = 0;
        }
    }

    if (lock_xy_mode_)
    {
        /*
            Weiji Xie Note:
            1. pos_b=0 at t=0, doesn't means the robot won't move
            2. pos_b=0 at t=0~T-1, also doesn't means the robot won't move
            3. in training time, since the delta pos is randomized, the robot will learn to follow
           'motion_joint_pos', instead of only the reference frame

        */
        for (int i = 0; i < motion_loader_->future_steps; ++i)
        {
            pos_b[i * 3 + 0] = pos_b[0];
            pos_b[i * 3 + 1] = pos_b[1];
        }
        std::cout << " locked xy mode | first frame pos_b: [" << pos_b[0] << ", " << pos_b[1]
                  << ", " << pos_b[2] << "]" << std::endl;
        // std::cout << " locked xy mode | last frame pos_b: [" << pos_b[pos_b.size() - 3] << ", "
        //           << pos_b[pos_b.size() - 2] << ", " << pos_b[pos_b.size() - 1] << "]" <<
        //           std::endl;
    }

    if (false)
    {
        for (int i = motion_loader_->future_steps-1; i >= 0 ; i--)
        {
            pos_b[i * 3 + 0] -= pos_b[0];
            pos_b[i * 3 + 1] -= pos_b[1];
        }
    }

    return pos_b;
}

std::vector<float> ObservationComputer::motion_anchor_ori_b_future(
    const unitree_hg::msg::LowState& low_state, int motion_t)
{
    std::vector<float> ori_b(30, 0.0f);  // 6 * 5 = 30

    if (motion_t < 0)
    {
        return ori_b;
    }

    // Get robot anchor pose: position from odometry (world), orientation from IMU
    std::array<float, 3> robot_pos = robot_pos_w_;
    std::array<float, 4> robot_quat = {
        low_state.imu_state.quaternion[0], low_state.imu_state.quaternion[1],
        low_state.imu_state.quaternion[2], low_state.imu_state.quaternion[3]};

    // Get future motion anchor orientations
    for (int i = 0; i < motion_loader_->future_steps; ++i)
    {
        int step_idx = std::min(motion_t + i, motion_loader_->T - 1);

        std::array<float, 3> ref_pos =
            motion_loader_->body_pos[step_idx][motion_loader_->anchor_body_index];
        std::array<float, 4> ref_quat =
            motion_loader_->body_ori[step_idx][motion_loader_->anchor_body_index];

        auto [ref_pos_b, ref_quat_b] = transform_ref_to_robot_frame(ref_pos, ref_quat);
        // Transform to body frame using proper frame transformation
        auto [_, ori_b_step] =
            subtract_frame_transforms(robot_pos, robot_quat, ref_pos_b, ref_quat_b);

        // Convert to rotation matrix and take first 2 rows
        std::vector<float> mat = matrix_from_quat(ori_b_step);

        ori_b[i * 6 + 0] = mat[0];  // First row
        ori_b[i * 6 + 1] = mat[1];
        ori_b[i * 6 + 2] = mat[3];
        ori_b[i * 6 + 3] = mat[4];  // Second row
        ori_b[i * 6 + 4] = mat[6];
        ori_b[i * 6 + 5] = mat[7];
    }

    return ori_b;
}

std::vector<float> ObservationComputer::get_base_lin_vel(const unitree_hg::msg::LowState& low_state)
{
    // Get base linear velocity: from odometry (world) then rotate to body frame
    std::array<float, 3> linvel_w = robot_linvel_w_;

    std::array<float, 4> quat = {
        low_state.imu_state.quaternion[0], low_state.imu_state.quaternion[1],
        low_state.imu_state.quaternion[2], low_state.imu_state.quaternion[3]};

    std::array<float, 3> linvel_b = quat_rotate_inverse(quat, linvel_w);

    return {linvel_b[0], linvel_b[1], linvel_b[2]};
}

std::vector<float> ObservationComputer::get_base_ang_vel(const unitree_hg::msg::LowState& low_state)
{
    // Angular velocity is already in body frame
    return {low_state.imu_state.gyroscope[0], low_state.imu_state.gyroscope[1],
            low_state.imu_state.gyroscope[2]};
}

std::vector<float> ObservationComputer::get_joint_pos_rel(
    const unitree_hg::msg::LowState& low_state)
{
    // IsaacLab order
    std::vector<float> joint_pos(29);

    // Extract joint positions from motor states (MuJoCo order)
    for (int i = 0; i < 29; ++i)
    {
        // mujoco_to_isaaclab_reindex: for each mujoco joint, find its isaaclab index
        int mujoco_idx = mujoco_to_isaaclab_reindex_[i];
        joint_pos[i] = low_state.motor_state[mujoco_idx].q - default_angles_[mujoco_idx];
        // low_state.motor_state[i].q: MuJoCo order
        // default_angles_: MuJoCo order
        // joint_pos: IsaacLab order
    }

    return joint_pos;
}

std::vector<float> ObservationComputer::get_joint_vel_rel(
    const unitree_hg::msg::LowState& low_state)
{
    std::vector<float> joint_vel(29);

    // Extract joint velocities from motor states (MuJoCo order)
    for (int i = 0; i < 29; ++i)
    {
        // mujoco_to_isaaclab_reindex: for each mujoco joint, find its isaaclab index
        int mujoco_idx = mujoco_to_isaaclab_reindex_[i];
        joint_vel[i] = low_state.motor_state[mujoco_idx].dq;
        // low_state.motor_state[i].dq: MuJoCo order
        // joint_vel: IsaacLab order
    }

    return joint_vel;
}

std::vector<float> ObservationComputer::get_last_action(const std::vector<float>& last_actions)
{
    // IsaacLab order
    return last_actions;
}

// Quaternion math utilities (matching Python math_np.py)

std::array<float, 4> ObservationComputer::quat_conjugate(const std::array<float, 4>& q)
{
    return {q[0], -q[1], -q[2], -q[3]};
}

std::array<float, 4> ObservationComputer::quat_mul(const std::array<float, 4>& q1,
                                                   const std::array<float, 4>& q2)
{
    float w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
    float w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];

    float w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    float x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    float y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    float z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;

    return {w, x, y, z};
}

std::array<float, 4> ObservationComputer::quat_inv(const std::array<float, 4>& q)
{
    std::array<float, 4> conj = quat_conjugate(q);
    float norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    float eps = 1e-9f;
    float inv_norm_sq = 1.0f / std::max(norm_sq, eps);

    return {conj[0] * inv_norm_sq, conj[1] * inv_norm_sq, conj[2] * inv_norm_sq,
            conj[3] * inv_norm_sq};
}

std::array<float, 3> ObservationComputer::quat_apply(const std::array<float, 4>& q,
                                                     const std::array<float, 3>& v)
{
    // Convert quaternion to rotation matrix and apply
    std::vector<float> rot_mat = matrix_from_quat(q);

    std::array<float, 3> result;
    result[0] = rot_mat[0] * v[0] + rot_mat[1] * v[1] + rot_mat[2] * v[2];
    result[1] = rot_mat[3] * v[0] + rot_mat[4] * v[1] + rot_mat[5] * v[2];
    result[2] = rot_mat[6] * v[0] + rot_mat[7] * v[1] + rot_mat[8] * v[2];

    return result;
}

std::array<float, 3> ObservationComputer::quat_rotate_inverse(const std::array<float, 4>& q,
                                                              const std::array<float, 3>& v)
{
    float q_w = q[0];
    std::array<float, 3> q_vec = {q[1], q[2], q[3]};

    // Component a: v * (2.0 * q_w^2 - 1.0)
    std::array<float, 3> a = {v[0] * (2.0f * q_w * q_w - 1.0f), v[1] * (2.0f * q_w * q_w - 1.0f),
                              v[2] * (2.0f * q_w * q_w - 1.0f)};

    // Component b: cross(q_vec, v) * q_w * 2.0
    std::array<float, 3> cross_product = {q_vec[1] * v[2] - q_vec[2] * v[1],
                                          q_vec[2] * v[0] - q_vec[0] * v[2],
                                          q_vec[0] * v[1] - q_vec[1] * v[0]};

    std::array<float, 3> b = {cross_product[0] * q_w * 2.0f, cross_product[1] * q_w * 2.0f,
                              cross_product[2] * q_w * 2.0f};

    // Component c: q_vec * dot(q_vec, v) * 2.0
    float dot_product = q_vec[0] * v[0] + q_vec[1] * v[1] + q_vec[2] * v[2];
    std::array<float, 3> c = {q_vec[0] * dot_product * 2.0f, q_vec[1] * dot_product * 2.0f,
                              q_vec[2] * dot_product * 2.0f};

    return {a[0] - b[0] + c[0], a[1] - b[1] + c[1], a[2] - b[2] + c[2]};
}

std::vector<float> ObservationComputer::matrix_from_quat(const std::array<float, 4>& quat)
{
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];

    std::vector<float> mat(9);

    mat[0] = 1.0f - 2.0f * (y * y + z * z);
    mat[1] = 2.0f * (x * y - w * z);
    mat[2] = 2.0f * (x * z + w * y);
    mat[3] = 2.0f * (x * y + w * z);
    mat[4] = 1.0f - 2.0f * (x * x + z * z);
    mat[5] = 2.0f * (y * z - w * x);
    mat[6] = 2.0f * (x * z - w * y);
    mat[7] = 2.0f * (y * z + w * x);
    mat[8] = 1.0f - 2.0f * (x * x + y * y);

    return mat;
}

std::pair<std::array<float, 3>, std::array<float, 4>>
ObservationComputer::subtract_frame_transforms(const std::array<float, 3>& pos_a,
                                               const std::array<float, 4>& quat_a,
                                               const std::array<float, 3>& pos_b,
                                               const std::array<float, 4>& quat_b)
{
    // Implementation matching Python subtract_frame_transforms function
    // Compute q10 = quat_inv(quat_a)
    std::array<float, 4> q10 = quat_inv(quat_a);

    // Compute relative quaternion: q12 = quat_mul(q10, quat_b)
    std::array<float, 4> q12 = quat_mul(q10, quat_b);

    // Compute relative position: t12 = quat_apply(q10, pos_b - pos_a)
    std::array<float, 3> pos_diff = {pos_b[0] - pos_a[0], pos_b[1] - pos_a[1], pos_b[2] - pos_a[2]};
    std::array<float, 3> t12 = quat_apply(q10, pos_diff);

    return {t12, q12};
}

// Add helper function for pretty printing vectors
template <typename T>
std::string ObservationComputer::vector_to_string(const std::vector<T>& vec, int max_elements)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);

    int elements_to_show = std::min(static_cast<int>(vec.size()), max_elements);

    for (int i = 0; i < elements_to_show; ++i)
    {
        if (i > 0)
            oss << ", ";
        oss << vec[i];
    }

    if (int(vec.size()) > max_elements)
    {
        oss << ", ... (showing " << max_elements << " of " << vec.size() << " elements)";
    }

    return oss.str();
}

std::vector<float> ObservationComputer::get_projected_gravity(
    const unitree_hg::msg::LowState& low_state)
{
    // Get quaternion from IMU (wxyz format)
    std::array<float, 4> quaternion = {
        low_state.imu_state.quaternion[0],  // w
        low_state.imu_state.quaternion[1],  // x
        low_state.imu_state.quaternion[2],  // y
        low_state.imu_state.quaternion[3]   // z
    };

    float qw = quaternion[0];
    float qx = quaternion[1];
    float qy = quaternion[2];
    float qz = quaternion[3];

    std::vector<float> gravity_orientation(3);

    gravity_orientation[0] = 2.0f * (-qz * qx + qw * qy);
    gravity_orientation[1] = -2.0f * (qz * qy + qw * qx);
    gravity_orientation[2] = 1.0f - 2.0f * (qw * qw + qz * qz);

    return gravity_orientation;
}

std::array<float, 4> ObservationComputer::calc_heading_quat(const std::array<float, 4>& quat)
{
    // Extract yaw angle from quaternion
    float yaw = std::atan2(2.0f * (quat[0] * quat[3] + quat[1] * quat[2]),
                           1.0f - 2.0f * (quat[2] * quat[2] + quat[3] * quat[3]));

    // Create quaternion with only yaw rotation
    float half_yaw = yaw * 0.5f;
    return {std::cos(half_yaw), 0.0f, 0.0f, std::sin(half_yaw)};
}

void ObservationComputer::setup_init_frame(const unitree_hg::msg::LowState& low_state)
{
    // if (frame_initialized_)
    // {
    //     return;  // Already initialized
    // }

    // Get robot's current pose
    std::array<float, 4> robot_quat = {
        low_state.imu_state.quaternion[0],  // w
        low_state.imu_state.quaternion[1],  // x
        low_state.imu_state.quaternion[2],  // y
        low_state.imu_state.quaternion[3]   // z
    };

    // Calculate robot's heading quaternion (yaw only)
    std::array<float, 4> robot_heading_quat = calc_heading_quat(robot_quat);

    // Store robot's initial state
    robot_init_yaw_ =
        std::atan2(2.0f * (robot_quat[0] * robot_quat[3] + robot_quat[1] * robot_quat[2]),
                   1.0f - 2.0f * (robot_quat[2] * robot_quat[2] + robot_quat[3] * robot_quat[3]));
    robot_init_pos_ = robot_pos_w_;
    robot_init_quat_ = robot_heading_quat;

    // Get reference motion's initial frame (first frame of motion)
    ref_init_pos_ = motion_loader_->body_pos[0][motion_loader_->anchor_body_index];
    ref_init_quat_ =
        calc_heading_quat(motion_loader_->body_ori[0][motion_loader_->anchor_body_index]);

    // Compute relative transformation: robot_init_quat * ref_init_quat_inv
    std::array<float, 4> ref_init_quat_inv = quat_inv(ref_init_quat_);
    ref_to_robot_quat_ = quat_mul(robot_init_quat_, ref_init_quat_inv);

    frame_initialized_ = true;

    // std::cout << "Frame initialized - Robot yaw: " << robot_init_yaw_ << ", Robot pos: ["
    //           << robot_init_pos_[0] << ", " << robot_init_pos_[1] << ", " << robot_init_pos_[2]
    //           << "]"
    //           << ", Ref pos: [" << ref_init_pos_[0] << ", " << ref_init_pos_[1] << ", "
    //           << ref_init_pos_[2] << "]" << std::endl;
    RCLCPP_INFO(node_->get_logger(),
                "Frame initialized - Robot yaw: %f, Robot pos: [%f, %f, %f], Ref pos: [%f, %f, %f]",
                robot_init_yaw_, robot_init_pos_[0], robot_init_pos_[1], robot_init_pos_[2],
                ref_init_pos_[0], ref_init_pos_[1], ref_init_pos_[2]);
}

std::pair<std::array<float, 3>, std::array<float, 4>>
ObservationComputer::transform_ref_to_robot_frame(const std::array<float, 3>& ref_pos,
                                                  const std::array<float, 4>& ref_quat)
{
    if (!frame_initialized_)
    {
        // Return identity if not initialized
        return {ref_pos, ref_quat};
    }

    // Transform position: robot_init_pos + robot_init_quat * ref_init_quat_inv * (ref_pos -
    // ref_init_pos)
    std::array<float, 4> ref_init_quat_inv = quat_inv(ref_init_quat_);
    std::array<float, 3> pos_diff = {ref_pos[0] - ref_init_pos_[0], ref_pos[1] - ref_init_pos_[1],
                                     ref_pos[2] - ref_init_pos_[2]};
    std::array<float, 3> pos_rel = quat_apply(ref_init_quat_inv, pos_diff);
    std::array<float, 3> pos_new = {robot_init_pos_[0] + pos_rel[0],
                                    robot_init_pos_[1] + pos_rel[1],
                                    robot_init_pos_[2] + pos_rel[2]};

    // Transform quaternion: ref_to_robot_quat * ref_quat
    std::array<float, 4> quat_new = quat_mul(ref_to_robot_quat_, ref_quat);

    return {pos_new, quat_new};
}

void ObservationComputer::align_robot_ref_xy(int motion_t)
{
    if (!frame_initialized_ || motion_t < 0)
    {
        return;
    }

    int step_idx = std::min(motion_t, motion_loader_->T - 1);
    std::array<float, 3> ref_pos =
        motion_loader_->body_pos[step_idx][motion_loader_->anchor_body_index];

    // Step 4: Compute intermediate variables
    std::array<float, 3> pos_diff = {ref_pos[0] - ref_init_pos_[0], ref_pos[1] - ref_init_pos_[1],
                                     ref_pos[2] - ref_init_pos_[2]};
    std::array<float, 4> ref_init_quat_inv = quat_inv(ref_init_quat_);
    std::array<float, 3> pos_rel = quat_apply(ref_init_quat_inv, pos_diff);

    // Use locked robot XY instead of current robot_pos_w_[XY]
    // This ensures alignment to the position when LockXY was enabled, not the current position
    robot_init_pos_[0] = locked_robot_xy_[0] - pos_rel[0];
    robot_init_pos_[1] = locked_robot_xy_[1] - pos_rel[1];
    // Z coordinate is not modified
}
