#!/bin/bash
# A3机器人最小训练配置
# 这是一个简化的训练命令，适合快速开始训练

python scripts/rsl_rl/train.py \
    --headless \
    --task=Tracking-Flat-A3-v0 \
    --motion_file=duida_a \
    --run_name=a3_minimal \
    agent.experiment_name=A3Training \
    agent.max_iterations=100000 \
    --num_envs=4096 \
    --seed=1 \
    --device=cuda:0
