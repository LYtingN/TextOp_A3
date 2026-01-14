#!/bin/bash
# A3机器人完整训练配置（基于G1配置调整）

python scripts/rsl_rl/train.py \
    --headless \
    --log_project_name TextOpTracker \
    --task=Tracking-Flat-A3-v0 \
    --motion_file=duida_* \
    --run_name=a3_full \
    agent.experiment_name=A3Training \
    agent.max_iterations=1000000 \
    --num_envs=8192 \
    env.commands.motion.anchor_body_name="torso_Link" \
    env.commands.motion.future_steps=5 \
    env.commands.motion.random_static_prob=-1.0 \
    env.rewards.feet_slide.params.pfail_threshold=1.0 \
    env.rewards.soft_landing.params.pfail_threshold=1.0 \
    env.rewards.overspeed.params.pfail_threshold=1.0 \
    env.rewards.overeffort.params.pfail_threshold=1.0 \
    env.rewards.feet_slide.weight=-0.3 \
    env.rewards.soft_landing.weight=-0.0003 \
    env.rewards.overspeed.weight=-1.0 \
    env.rewards.overeffort.weight=-1.0 \
    env.commands.motion.enable_adaptive_sampling=True \
    env.commands.motion.ads_type=v2 \
    env.commands.motion.adaptive_beta=0.5 \
    env.commands.motion.adaptive_alpha=0.1 \
    env.commands.motion.adaptive_uniform_ratio=0.1 \
    agent.policy.actor_hidden_dims=[2048,1024,512] \
    agent.policy.critic_hidden_dims=[2048,1024,512] \
    --seed=1 \
    --device=cuda:0
