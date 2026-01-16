# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to train RL agent with RSL-RL."""
"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--motion_file", type=str, required=True, help="The name of the wand registry.")
parser.add_argument(
    "--print_joint_names",
    action="store_true",
    default=False,
    help="Print IsaacLab runtime joint order (robot.data.joint_names) once at startup.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import glob
import gymnasium as gym
import os
import torch
from datetime import datetime
from pathlib import Path

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import textop_tracker.tasks  # noqa: F401
from textop_tracker.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False



@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    motion_files = glob.glob(str(Path("./artifacts") / Path(args_cli.motion_file) / "motion.npz"))
    if not motion_files:
        motion_files = glob.glob(str(Path("./artifacts") / f"{args_cli.motion_file}.npz"))
    if not motion_files:
        motion_files = glob.glob(str(Path("./artifacts") / f"{args_cli.motion_file}*.npz"))
    if not motion_files:
        raise FileNotFoundError(
            f"No motion file found matching '{args_cli.motion_file}' in ./artifacts. "
            f"Tried: ./artifacts/{args_cli.motion_file}/motion.npz, "
            f"./artifacts/{args_cli.motion_file}.npz, "
            f"./artifacts/{args_cli.motion_file}*.npz"
        )
    # 单文件命令版：只取第一条，写入 motion_file 字段
    env_cfg.commands.motion.motion_file = motion_files[0]
    print(f"[INFO] Found {len(motion_files)} motion file(s): {motion_files}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Optional debug: print runtime joint order (this is the order your NPZ joint_pos/joint_vel must follow)
    if args_cli.print_joint_names:
        try:
            # ManagerBasedRLEnv has `.scene` with a `robot` entry
            robot = env.unwrapped.scene["robot"]  # type: ignore[attr-defined]
            names = list(robot.data.joint_names)  # type: ignore[attr-defined]
            print(f"[DEBUG] Runtime joint order (len={len(names)}):")
            for i, n in enumerate(names):
                print(f"  {i:02d}: {n}")
        except Exception as e:
            print(f"[WARNING] Failed to print joint names: {e}")
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Debug: print observation/action dimensions once at startup
    try:
        obs, extras = env.get_observations()
        num_obs = int(obs.shape[1])
        num_actions = int(env.num_actions)
        print(f"[INFO] Obs shape: {tuple(obs.shape)} (num_obs={num_obs}), num_actions={num_actions}")
        # Optional: show if privileged observations are present
        obs_dict = extras.get("observations", {}) if isinstance(extras, dict) else {}
        if isinstance(obs_dict, dict):
            priv_keys = [k for k in ("critic", "teacher", "rnd_state") if k in obs_dict]
            if priv_keys:
                shapes = {k: tuple(obs_dict[k].shape) for k in priv_keys}
                print(f"[INFO] Extra observation keys: {priv_keys} with shapes {shapes}")
    except Exception as e:
        print(f"[WARNING] Failed to print obs/action dimensions: {e}")

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=None)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training

    (runner.learn)(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
