


#!/usr/bin/env python3
"""
Replay motion from CSV/PKL in Isaac Lab and export NPZ (with wrist joints dropped).
Also saves a resampled binary contact_mask using Zero-Order Hold (ZOH) at the output FPS.

- Saves locally only. All Weights & Biases (wandb) references removed.
- Output path is derived from --output_name ("foo" -> ./foo.npz). You can pass a path-like name
  such as "out/motion_run1" and it will save to ./out/motion_run1.npz (directories will be created).
"""

import argparse
import numpy as np
import torch
from typing import Optional, Tuple, List
from pathlib import Path
import threading
import queue
import sys
import select

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Replay motion from CSV/PKL and output NPZ (saved locally).")
parser.add_argument("--input_file", type=str,  help="Path to input motion (CSV or PKL).")
parser.add_argument("--input_dir", type=str, help="Path to folder of input motions (all CSV/PKL inside).")
parser.add_argument("--input_fps", type=int, default=30, help="FPS of the input CSV (ignored for PKL).")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Frame range: START END (both inclusive, 1-based). If omitted, uses all frames.",
)
# parser.add_argument("--output_name", type=str, required=True, help="Base name or path stem for the output motion npz ('.npz' auto-added).")
parser.add_argument("--output_fps", type=int, default=50, help="FPS of the output motion.")
parser.add_argument("--input_format", type=str, default="auto", choices=["auto", "csv", "pkl"],
                    help="Input file type. 'auto' infers from extension.")
parser.add_argument("--save_dir", type=str, default="./", help="Directory to save the output npz.")
parser.add_argument("--ask_save", action="store_true", 
                    help="If set, ask whether to save after each motion conversion (only effective in --input_dir mode).")

# Let AppLauncher add its args (e.g., --device)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Require at least one of file/dir
if not args_cli.input_file and not args_cli.input_dir:
    parser.error("Please provide --input_file or --input_dir")
# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Isaac Lab imports (after simulator launch)
# -----------------------------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

# Your robot config
from textop_tracker.robots.a3 import A3_CFG


# -----------------------------------------------------------------------------
# Joint order (29 DoF) and reduction (drop wrists)
# -----------------------------------------------------------------------------
# JOINT_NAMES_26 = [
#     "idx01_left_hip_roll",
#     "idx02_left_hip_yaw",
#     "idx03_left_hip_pitch",
#     "idx04_left_tarsus",
#     "idx05_left_toe_pitch",
#     "idx06_left_toe_roll",
#     "idx07_right_hip_roll",
#     "idx08_right_hip_yaw",
#     "idx09_right_hip_pitch",
#     "idx10_right_tarsus",
#     "idx11_right_toe_pitch",
#     "idx12_right_toe_roll",
#     "left_shoulder_pitch_joint",
#     "left_shoulder_roll_joint",
#     "left_shoulder_yaw_joint",
#     "left_elbow_pitch_joint",
#     "left_arm_yaw_joint",
#     "left_wrist_roll_joint",
#     "left_wrist_pitch_joint",
#     "right_shoulder_pitch_joint",
#     "right_shoulder_roll_joint",
#     "right_shoulder_yaw_joint",
#     "right_elbow_joint",
#     "right_arm_yaw_joint",
#     "right_wrist_roll_joint",
#     "right_wrist_pitch_joint",
# ]
JOINT_NAMES_30 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


# --- DOF index filtering (drop head_joint and fixed wrist joints) ---
HEAD_JOINT_NAME = "head_joint"
HEAD_DROP_IDXS = []
if HEAD_JOINT_NAME in JOINT_NAMES_30:
    HEAD_DROP_IDXS.append(JOINT_NAMES_30.index(HEAD_JOINT_NAME))

# Fixed wrist joints (not actuated, excluded from motion data)
WRIST_FIXED_JOINTS = [
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
WRIST_DROP_IDXS = []
for joint_name in WRIST_FIXED_JOINTS:
    if joint_name in JOINT_NAMES_30:
        WRIST_DROP_IDXS.append(JOINT_NAMES_30.index(joint_name))

# Combine all joints to drop
ALL_DROP_IDXS = HEAD_DROP_IDXS + WRIST_DROP_IDXS
DOF_KEEP_IDXS = [i for i in range(len(JOINT_NAMES_30)) if i not in ALL_DROP_IDXS]

print(f"[DEBUG] Dropping joints: HEAD={HEAD_DROP_IDXS}, WRIST_FIXED={WRIST_DROP_IDXS}")
print(f"[DEBUG] DOF_KEEP_IDXS (len={len(DOF_KEEP_IDXS)}):", DOF_KEEP_IDXS)


# -----------------------------------------------------------------------------
# Scene config
# -----------------------------------------------------------------------------
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
    robot: ArticulationCfg = A3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# -----------------------------------------------------------------------------
# Motion Loader
# -----------------------------------------------------------------------------
class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: Optional[Tuple[int, int]],
        input_format: str,
        dof_keep_indices: List[int],
    ):
        self.motion_file = motion_file
        self.input_fps_arg = input_fps  # only for CSV
        self.output_fps = output_fps
        self.output_dt = 1.0 / float(self.output_fps)
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self.input_format = input_format
        self.dof_keep_indices = dof_keep_indices

        # contact mask buffers
        self.motion_contact_mask_input: Optional[torch.Tensor] = None
        self.motion_contact_mask: Optional[torch.Tensor] = None
        self._load_motion()

        self._interpolate_motion()
        self._compute_velocities()

    def _infer_format(self) -> str:
        if self.input_format != "auto":
            return self.input_format
        ext = self.motion_file.lower().rsplit(".", 1)[-1]
        if ext == "csv":
            return "csv"
        if ext == "pkl":
            return "pkl"
        raise ValueError(f"Cannot infer input format from extension .{ext}")

    def _load_motion(self):
        fmt = self._infer_format()

        if fmt == "csv":
            if self.frame_range is None:
                motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
            else:
                motion = torch.from_numpy(
                    np.loadtxt(
                        self.motion_file,
                        delimiter=",",
                        skiprows=self.frame_range[0] - 1,
                        max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                    )
                )
            motion = motion.to(torch.float32).to(self.device)

            self.motion_base_poss_input = motion[:, :3]
            base_quat_xyzw = motion[:, 3:7]
            self.motion_base_rots_input = base_quat_xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz
            dof_all = motion[:, 7:]
            self.motion_dof_poss_input = dof_all[:, self.dof_keep_indices]

            self.input_fps = int(self.input_fps_arg) if self.input_fps_arg else 30
            self.input_dt = 1.0 / float(self.input_fps)
            self.input_frames = motion.shape[0]

        elif fmt == "pkl":
            import joblib
            data = joblib.load(self.motion_file)

            # unwrap single-key dict like {"0-Transitions_dance1": {...}}
            if isinstance(data, dict) and len(data) == 1 and isinstance(next(iter(data.values())), dict):
                data = next(iter(data.values()))

            self.input_fps = int(data["fps"])
            self.input_dt = 1.0 / float(self.input_fps)

            # Use keys from your structure
            root_pos = np.asarray(data.get("root_trans_offset"), dtype=np.float32)   # (T,3)
            root_xyzw = np.asarray(data.get("root_rot"), dtype=np.float32)           # (T,4) xyzw
            dof_all = np.asarray(data.get("dof"), dtype=np.float32)                  # (T,29)
            print('dof_all',dof_all.shape)
            # dof_all=dof_all[:,:26]
            # Optional contact mask - use detailed_foot_contacts
            # Extract indices: 3,4,7,8 (left foot) and 11,12,15,16 (right foot)
            # Note: 1-based to 0-based: 3,4,7,8 -> 2,3,6,7 and 11,12,15,16 -> 10,11,14,15
            contact_mask = None
            detailed_foot_contacts = data.get("detailed_foot_contacts", None)
            if detailed_foot_contacts is not None:
                detailed_foot_contacts = np.asarray(detailed_foot_contacts, dtype=np.float32)  # (T, 16)
                # Extract left foot contacts (indices 2,3,6,7) and right foot contacts (indices 10,11,14,15)
                # Left foot: 3,4,7,8 (1-based) -> 2,3,6,7 (0-based)
                # Right foot: 11,12,15,16 (1-based) -> 10,11,14,15 (0-based)
                left_foot_indices = [2, 3, 6, 7]   # 1-based: 3, 4, 7, 8
                right_foot_indices = [10, 11, 14, 15]  # 1-based: 11, 12, 15, 16
                selected_indices = left_foot_indices + right_foot_indices
                contact_mask = detailed_foot_contacts[:, selected_indices]  # (T, 8)
                # Normalize to {0,1} using condition > 0.1
                contact_mask = (contact_mask > 0.1).astype(np.float32)

            # Optional slicing
            if self.frame_range is not None:
                s, e = self.frame_range
                root_pos = root_pos[s - 1:e]
                root_xyzw = root_xyzw[s - 1:e]
                dof_all = dof_all[s - 1:e]
                if contact_mask is not None:
                    contact_mask = contact_mask[s - 1:e]

            self.motion_base_poss_input = torch.as_tensor(root_pos, device=self.device)
            base_quat_xyzw = torch.as_tensor(root_xyzw, device=self.device)
            self.motion_base_rots_input = base_quat_xyzw[:, [3, 0, 1, 2]]  # xyzw → wxyz
            dof_all_t = torch.as_tensor(dof_all, device=self.device)
            print("[DEBUG] dof_all_t.shape:", dof_all_t.shape)
            print("[DEBUG] max(self.dof_keep_indices):", max(self.dof_keep_indices))
            print("[DEBUG] len(self.dof_keep_indices):", len(self.dof_keep_indices))
            print("[DEBUG] self.dof_keep_indices:", self.dof_keep_indices)

            # Drop wrist joints
            self.motion_dof_poss_input = dof_all_t[:, self.dof_keep_indices]

            # Contact mask (binary), keep on device for cheap indexing
            # Shape: (T, 8) where first 4 are left foot, last 4 are right foot
            if contact_mask is not None:
                cm = torch.as_tensor(contact_mask, device=self.device, dtype=torch.float32)
                if cm.ndim == 1:
                    cm = cm.unsqueeze(-1)
                # Already converted to binary (0/1) using > 0.1, but apply threshold again for consistency
                self.motion_contact_mask_input = (cm > 0.1).to(torch.float32)

            self.input_frames = self.motion_base_poss_input.shape[0]
            self.duration = (self.input_frames - 1) * self.input_dt
            print(
                f"[Joblib PKL] Loaded {self.input_frames} frames @ {self.input_fps} fps "
                f"(kept J'={self.motion_dof_poss_input.shape[1]})"
            )

        else:
            raise ValueError(f"Unsupported input format: {fmt}")

        self.duration = (self.input_frames - 1) * self.input_dt
        print(
            f"Motion loaded ({self.motion_file}), duration: {self.duration:.3f} sec, "
            f"frames: {self.input_frames}, in_fps={self.input_fps}, kept J'={self.motion_dof_poss_input.shape[1]}"
        )

    def _interpolate_motion(self):
        # Sample output timeline (exclude last endpoint for consistent blend math)
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)

        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)

        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )

        # -----------------------------
        # ZOH resample for contact mask
        # -----------------------------
        if self.motion_contact_mask_input is not None:
            cm_in = self.motion_contact_mask_input
            # Zero-order hold: take the "left" index
            self.motion_contact_mask = cm_in[index_0]
            # print('self.motion_contact_mask',self.motion_contact_mask)
        else:
            self.motion_contact_mask = None

        print(
            f"Motion interpolated: in_frames={self.input_frames}, in_fps={self.input_fps} -> "
            f"out_frames={self.output_frames}, out_fps={self.output_fps}; J'={self.motion_dof_poss.shape[1]}"
        )

    @staticmethod
    def _lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    @staticmethod
    def _slerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor):
        # map continuous time to [0, input_frames-1]
        phase = times / self.duration
        index_0 = torch.floor(phase * (self.input_frames - 1)).long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        # Linear velocities via finite differences
        if self.motion_base_poss.shape[0] >= 2:
            self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
            self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        else:
            self.motion_base_lin_vels = torch.zeros_like(self.motion_base_poss)
            self.motion_dof_vels = torch.zeros_like(self.motion_dof_poss)

        # Angular velocities from quaternion derivative
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    @staticmethod
    def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
        T = rotations.shape[0]
        if T < 3:
            return torch.zeros((T, 3), dtype=rotations.dtype, device=rotations.device)
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # (T-2, 4)
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # (T-2, 3)
        # pad to length T (repeat endpoints)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(self):
        state = (
            self.motion_base_poss[self.current_idx:self.current_idx + 1],
            self.motion_base_rots[self.current_idx:self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx:self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx:self.current_idx + 1],
            self.motion_dof_poss[self.current_idx:self.current_idx + 1],
            self.motion_dof_vels[self.current_idx:self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


# --- Non-blocking input listener for skip functionality ---
class NonBlockingInput:
    """Non-blocking input listener that runs in a separate thread."""
    def __init__(self):
        self.input_queue = queue.Queue()
        self.thread = None
        self.running = False
        
    def _input_listener(self):
        """Thread function that listens for input."""
        while self.running:
            try:
                # Check if input is available (non-blocking)
                if sys.stdin.isatty():
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if ready:
                        char = sys.stdin.read(1)
                        if char:
                            self.input_queue.put(char.lower())
                else:
                    # Fallback for non-TTY (e.g., when piped)
                    try:
                        line = sys.stdin.readline()
                        if line:
                            self.input_queue.put(line.strip().lower()[0] if line.strip() else '')
                    except (EOFError, KeyboardInterrupt):
                        break
            except (EOFError, KeyboardInterrupt, OSError):
                break
            except Exception:
                # Ignore other exceptions and continue
                pass
    
    def start(self):
        """Start the input listener thread."""
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self.thread = threading.Thread(target=self._input_listener, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop the input listener thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)
    
    def has_skip_command(self):
        """Check if user has pressed 's' to skip."""
        chars = []
        while not self.input_queue.empty():
            try:
                chars.append(self.input_queue.get_nowait())
            except queue.Empty:
                break
        return 's' in chars


# -----------------------------------------------------------------------------
# Simulation runner
# -----------------------------------------------------------------------------
# def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: List[str]):
#     motion = MotionLoader(
#         motion_file=args_cli.input_file,
#         input_fps=args_cli.input_fps,
#         output_fps=args_cli.output_fps,
#         device=sim.device,
#         frame_range=tuple(args_cli.frame_range) if args_cli.frame_range else None,
#         input_format=args_cli.input_format,
#         dof_keep_indices=DOF_KEEP_IDXS,
#     )

#     robot = scene["robot"]
#     robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

#     # Resolve local output path from --output_name
    
#     # out_path = Path(args_cli.output_name)
#     # if out_path.suffix.lower() != ".npz":
#     #     out_path = out_path.with_suffix(".npz")
#     # out_path = out_path.expanduser()
#     # out_path.parent.mkdir(parents=True, exist_ok=True)
#     # Example: --save_dir ./out
#     save_dir = Path(args_cli.save_dir).expanduser()
#     save_dir.mkdir(parents=True, exist_ok=True)

#     # Keep same stem as input file
#     in_path = Path(args_cli.input_file)
#     out_path = save_dir / (in_path.stem + ".npz")

#     log = {
#         "fps": np.array([args_cli.output_fps], dtype=np.int32),
#         "joint_pos": [],
#         "joint_vel": [],
#         "body_pos_w": [],
#         "body_quat_w": [],
#         "body_lin_vel_w": [],
#         "body_ang_vel_w": [],
#     }
#     file_saved = False

#     while simulation_app.is_running():
#         idx_play = motion.current_idx
#         ((
#             motion_base_pos,
#             motion_base_rot,
#             motion_base_lin_vel,
#             motion_base_ang_vel,
#             motion_dof_pos,
#             motion_dof_vel,
#         ), reset_flag) = motion.get_next_state()
#         # # --- DEBUG PRINT: contact mask for this output frame ---
#         # if motion.motion_contact_mask is not None:
#         #     cm = motion.motion_contact_mask[idx_play].detach().cpu().numpy().astype(int).tolist()
#         #     # Optional: highlight changes vs previous frame
#         #     if not hasattr(motion, "_cm_prev"):
#         #         print(f"[contact] frame={idx_play+1}/{motion.output_frames} mask={cm}")
#         #     else:
#         #         changed = (np.array(cm) != np.array(motion._cm_prev)).any()
#         #         tag = " (CHANGE)" if changed else ""
#         #         print(f"[contact] frame={idx_play+1}/{motion.output_frames} mask={cm}{tag}")
#         #     motion._cm_prev = cm
#         # Root
#         root_states = robot.data.default_root_state.clone()
#         root_states[:, :3] = motion_base_pos
#         root_states[:, 3:7] = motion_base_rot
#         root_states[:, 7:10] = motion_base_lin_vel
#         root_states[:, 10:] = motion_base_ang_vel
#         robot.write_root_state_to_sim(root_states)

#         # Joints
#         joint_pos = robot.data.default_joint_pos.clone()
#         joint_vel = robot.data.default_joint_vel.clone()
#         joint_pos[:, robot_joint_indexes] = motion_dof_pos
#         joint_vel[:, robot_joint_indexes] = motion_dof_vel
#         robot.write_joint_state_to_sim(joint_pos, joint_vel)

#         # Render only (no physics step)
#         sim.render()
#         scene.update(sim.get_physics_dt())

#         pos_lookat = root_states[0, :3].cpu().numpy()
#         sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

#         # Accumulate logs until we save once (first full pass)
#         if not file_saved:
#             log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
#             log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
#             log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
#             log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
#             log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
#             log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

#         if reset_flag and not file_saved:
#             file_saved = True
#             for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
#                 log[k] = np.stack(log[k], axis=0)

#             # --------------------------------------
#             # Save contact_mask (ZOH at output FPS)
#             # --------------------------------------
#             if motion.motion_contact_mask is not None:
#                 cm = motion.motion_contact_mask.detach().cpu().numpy().astype(np.float32)
#                 # Align to the number of frames we actually logged
#                 T_logged = log["joint_pos"].shape[0]
#                 if cm.shape[0] != T_logged:
#                     T = min(T_logged, cm.shape[0])
#                     cm = cm[:T]
#                     # Also trim all logs to T for consistency
#                     for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
#                         log[k] = log[k][:T]
#                 log["contact_mask"] = cm  # shape (T, 2) typically
#                 # print('cm',cm)
#             # Save locally (compressed)
#             np.savez_compressed(out_path, **log)
#             print(f"[INFO]: Motion saved locally to: {out_path.resolve()}")
# --- Simulation runner (REPLACE the entire run_simulator with this) ---
def run_one_motion_file(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    joint_names: List[str],
    motion_path: Path,
    save_dir: Path,
    ask_save: bool = False,
) -> Optional[Path]:
    motion = MotionLoader(
        motion_file=str(motion_path),
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=tuple(args_cli.frame_range) if args_cli.frame_range else None,
        input_format=args_cli.input_format,
        dof_keep_indices=DOF_KEEP_IDXS[:25],
    )

    robot = scene["robot"]

    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    out_path = _unique_outpath(save_dir, motion_path.stem)

    log = {
        "fps": np.array([args_cli.output_fps], dtype=np.int32),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    
    # Initialize input listener if ask_save is enabled
    input_listener = None
    user_skipped = False
    if ask_save:
        input_listener = NonBlockingInput()
        input_listener.start()
        print(f"[INFO] Playing motion '{motion_path.name}'. Press 's' to skip at any time.")

    while simulation_app.is_running():
        # Check for skip command if ask_save is enabled
        if ask_save and input_listener and input_listener.has_skip_command():
            user_skipped = True
            print(f"\n[INFO] Skip command detected. Stopping playback of '{motion_path.name}'...")
            break
        idx_play = motion.current_idx
        ((
            motion_base_pos,
            motion_base_rot,
            motion_base_lin_vel,
            motion_base_ang_vel,
            motion_dof_pos,
            motion_dof_vel,
        ), reset_flag) = motion.get_next_state()

        # Root
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # Joints
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        # joint_vel[:, robot_joint_indexes] =10000
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # Render only (no physics step)
        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.5, 2.5, 0.5]), pos_lookat)

        # Accumulate logs until we save once (first full pass)
        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            # Stop input listener if it was started
            if input_listener:
                input_listener.stop()
            
            for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                log[k] = np.stack(log[k], axis=0)

            # Save contact_mask (ZOH at output FPS)
            if motion.motion_contact_mask is not None:
                cm = motion.motion_contact_mask.detach().cpu().numpy().astype(np.float32)
                T_logged = log["joint_pos"].shape[0]
                if cm.shape[0] != T_logged:
                    T = min(T_logged, cm.shape[0])
                    cm = cm[:T]
                    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                        log[k] = log[k][:T]
                log["contact_mask"] = cm

            # Ask user whether to save if ask_save is enabled
            should_save = True
            if ask_save:
                while True:
                    response = input(f"\n[PROMPT] Save motion '{motion_path.name}' to '{out_path}'? (y/n): ").strip().lower()
                    if response in ('y', 'yes', ''):
                        should_save = True
                        break
                    elif response in ('n', 'no'):
                        should_save = False
                        print(f"[INFO]: Skipping save for '{motion_path.name}'")
                        break
                    else:
                        print("[ERROR]: Please enter 'y' or 'n'")

            if should_save:
                np.savez_compressed(out_path, **log)
                print(f"[INFO]: Motion saved locally to: {out_path}")
                # Important: return so the caller can move on to the next file
                return out_path
            else:
                # Return None if user chose not to save
                return None

    # Stop input listener if loop exited
    if input_listener:
        input_listener.stop()
    
    # If user skipped, return None without saving
    if user_skipped:
        print(f"[INFO]: Motion '{motion_path.name}' was skipped and not saved.")
        return None

    # If the app closed unexpectedly before saving:
    raise RuntimeError(f"Simulation ended before saving {motion_path}")

# --- Helpers (ADD this block somewhere above main) ---
def _gather_motion_files(input_file: Optional[str], input_dir: Optional[str]) -> List[Path]:
    files: List[Path] = []
    if input_dir:
        d = Path(input_dir).expanduser()
        if not d.exists():
            raise FileNotFoundError(f"--input_dir not found: {d}")
        # non-recursive; change to rglob if you want recursion
        files.extend(sorted(d.glob("*.csv")))
        files.extend(sorted(d.glob("*.pkl")))
    if input_file:
        files.append(Path(input_file).expanduser())
    # deduplicate while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    if not uniq:
        raise FileNotFoundError("No input files found (supported: .csv, .pkl).")
    return uniq

def _unique_outpath(save_dir: Path, stem: str) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{stem}.npz"
    if not out.exists():
        return out
    i = 1
    while True:
        cand = save_dir / f"{stem}_{i}.npz"
        if not cand.exists():
            return cand
        i += 1
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
# def main():
#     sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
#     sim_cfg.dt = 1.0 / float(args_cli.output_fps)
#     sim = SimulationContext(sim_cfg)

#     scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
#     scene = InteractiveScene(scene_cfg)

#     sim.reset()
#     print("[INFO]: Setup complete...")

#     run_simulator(
#         sim,
#         scene,
#         joint_names=[
#             "left_hip_pitch_joint",
#             "left_hip_roll_joint",
#             "left_hip_yaw_joint",
#             "left_knee_joint",
#             "left_ankle_pitch_joint",
#             "left_ankle_roll_joint",
#             "right_hip_pitch_joint",
#             "right_hip_roll_joint",
#             "right_hip_yaw_joint",
#             "right_knee_joint",
#             "right_ankle_pitch_joint",
#             "right_ankle_roll_joint",
#             "waist_yaw_joint",
#             "waist_pitch_joint",
#             "waist_roll_joint",
#             "left_shoulder_pitch_joint",
#             "left_shoulder_roll_joint",
#             "left_shoulder_yaw_joint",
#             "left_elbow_joint",
#             "right_shoulder_pitch_joint",
#             "right_shoulder_roll_joint",
#             "right_shoulder_yaw_joint",
#             "right_elbow_joint",
#         ],
#     )
# --- Main (REPLACE your main() with this) ---
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / float(args_cli.output_fps)
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")

    # Build file list
    files = _gather_motion_files(args_cli.input_file, args_cli.input_dir)

    # Save dir once
    save_dir = Path(args_cli.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    joint_names = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # "head_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    # Fixed wrist joints (not actuated, excluded from motion data)
    # "left_wrist_pitch_joint",
    # "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    # Fixed wrist joints (not actuated, excluded from motion data)
    # "right_wrist_pitch_joint",
    # "right_wrist_yaw_joint",
    ]


    # Only enable ask_save in input_dir mode
    ask_save_enabled = args_cli.ask_save and args_cli.input_dir is not None
    
    total = len(files)
    for i, f in enumerate(files, 1):
        print(f"[{i}/{total}] Processing: {f.name}")
        outp = run_one_motion_file(sim, scene, joint_names, f, save_dir, ask_save=ask_save_enabled)
        if outp is not None:
            print(f"[{i}/{total}] Done -> {outp}")
        else:
            print(f"[{i}/{total}] Done -> (skipped)")



if __name__ == "__main__":
    main()
    simulation_app.close()
