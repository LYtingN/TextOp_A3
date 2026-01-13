import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from textop_tracker.assets import ASSET_DIR

# NOTE:
# - AgiBot A3 assets live under `textop_tracker/assets/agibot_a3/`.
# - Link/joint naming in the provided URDF uses `*_Link` (capital L). Be careful when referencing body names.


A3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        # Use the lite-contacts variant for RL (explicit collision geometries / names).
        asset_path=f"{ASSET_DIR}/agibot_a3/urdf/a3_lite_contacts.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        # Conservative PD gains. Tune per-robot if needed.
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            # A3 URDF uses similar joint naming to Unitree (hip/knee/ankle).
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            # Arms (if present in the motion / task).
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        # Simple default actuator model for all joints. Refine into groups if you need per-joint limits.
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=120.0,
            velocity_limit_sim=30.0,
            stiffness=80.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)


# Action scaling for JointPositionActionCfg.
# A small scale keeps joint-position commands stable with generic PD gains.
A3_ACTION_SCALE: float = 0.25

