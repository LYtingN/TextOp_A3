"""Replay motion from a local NPZ file.

Usage:
    python scripts/replay_npz.py --motion_file ./data/aiming1_subject1.npz
"""

import argparse
import numpy as np
import torch
import time
from isaaclab.app import AppLauncher
import threading
# CLI args
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--motion_file", type=str, required=True, help="Path to local motion .npz file")

# add AppLauncher args (e.g., --headless)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# robot config & motion loader
# from whole_body_tracking.robots.g1 import X2_CYLINDER_CFG
from whole_body_tracking.robots.A3 import A3_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = A3_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    # load motion directly from file
    motion_file = args_cli.motion_file
    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )

    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    # --- configurable slowdown factor ---
    slowdown_factor = 1  # e.g., 0.3 = play at 30% real-time speed
    print_flag = {"trigger": False}
    # --- background input thread ---
    def input_thread():
        while True:
            _ = input()  # waits for Enter
            print_flag["trigger"] = True

    threading.Thread(target=input_thread, daemon=True).start()
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # only render, no physics
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        # --- handle print request ---
        if print_flag["trigger"]:
            print_flag["trigger"] = False
            print(f"Current frame: {int(time_steps[0].item())}")
        # --- slow down playback ---
        if slowdown_factor > 0:
            time.sleep(sim_dt / slowdown_factor)
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
