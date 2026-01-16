from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from textop_tracker.robots.a3 import A3_ACTION_SCALE, A3_CFG
from textop_tracker.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class A3FlatEnvCfg(TrackingEnvCfg):
    """Tracking task configuration for AgiBot A3 (URDF-based humanoid)."""

    def __post_init__(self):
        super().__post_init__()

        # Robot asset
        self.scene.robot = A3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Actions
        self.actions.joint_pos.scale = A3_ACTION_SCALE

        self.commands.motion.anchor_body_name = "pelvis_link"

        self.commands.motion.body_names = [
            "pelvis_link",
            "left_hip_roll_Link",
            "left_knee_Link",
            "left_ankle_roll_Link",
            "right_hip_roll_Link",
            "right_knee_Link",
            "right_ankle_roll_Link",
            "torso_Link",
            "left_shoulder_roll_Link",
            "left_elbow_Link",
            "left_wrist_roll_Link",
            "right_shoulder_roll_Link",
            "right_elbow_Link",
            "right_wrist_roll_Link",
        ]

     
