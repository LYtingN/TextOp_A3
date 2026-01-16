import isaaclab.sim as sim_utils
from .actuator import DelayedImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from textop_tracker.assets import ASSET_DIR
##
# Actuator config.
##
def reflected_inertia_from_two_stage_planetary(
  rotor_inertia: tuple[float, float, float],
  gear_ratio: tuple[float, float, float],
) -> float:
  """Compute reflected inertia of a two-stage planetary gearbox."""
  assert gear_ratio[0] == 1
  r1 = rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2
  r2 = rotor_inertia[1] * gear_ratio[2] ** 2
  r3 = rotor_inertia[2]
  return r1 + r2 + r3

ROTOR_INERTIAS_PFP_78_58 = (
  3.0451e-5,
  4.13223685e-7,
  1.16223167e-7,
)
GEARS_PFP_78_58 = (
  1,
  4,
  19.67 / 4.0,
)
ARMATURE_PFP_78_58 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_PFP_78_58, GEARS_PFP_78_58
)

ROTOR_INERTIAS_PFP_93_65 = (
  1.33777e-4,
  8.85547899e-7,
  4.5497007e-7,
)
GEARS_PFP_93_65 = (
  1,
  3.96428,
  21.906 / 3.96428,
)
ARMATURE_PFP_93_65 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_PFP_93_65, GEARS_PFP_93_65
)

ROTOR_INERTIAS_PFP_41_48 = (
  1.1471e-6,
  4.5743675e-8,
  2.402828e-9,
)
GEARS_PFP_41_48 = (
  1,
  4.941176471,
  24.41522491 / 4.941176471,
)
ARMATURE_PFP_41_48 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_PFP_41_48, GEARS_PFP_41_48
)

ROTOR_INERTIAS_PFP_59_60 = (
  1.0016e-5,
  3.2434211e-8,
  1.5055598e-8,
)
GEARS_PFP_59_60 = (
  1,
  4,
  22.1 / 4.0,
)
ARMATURE_PFP_59_60 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_PFP_59_60, GEARS_PFP_59_60
)

ROTOR_INERTIAS_PFP_110_75 = (
  2.8771e-4,
  1.069539474e-6,
  3.843493e-7,
)
GEARS_PFP_110_75 = (
  1,
  4,
  20.0 / 4.0,
)
ARMATURE_PFP_110_75 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_PFP_110_75, GEARS_PFP_110_75
)

A3_CYLINDER_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/agibot_a3/urdf/agibot_a3.urdf",
        # asset_path=f"{ASSET_DIR}/agibot_a3/urdf/a3_lite_contacts.urdf",
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.09),
        joint_pos={
            "left_hip_pitch_joint": -0.1311,
            "right_hip_pitch_joint": -0.1311,
            "left_hip_roll_joint": 0.0056,
            "right_hip_roll_joint": -0.0056,
            "left_hip_yaw_joint": -0.0348,
            "right_hip_yaw_joint": 0.0348,
            "left_knee_joint": 0.2468,
            "right_knee_joint": 0.2468,
            "left_ankle_pitch_joint": -0.1204,
            "right_ankle_pitch_joint": -0.1204,
            "left_ankle_roll_joint": -0.0078,
            "right_ankle_roll_joint": 0.0078,
            # Waist.
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            # Arms.
            "left_shoulder_pitch_joint": 0.3,
            "right_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.12,
            "right_shoulder_roll_joint": -0.12,
            "left_shoulder_yaw_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.8,
            "right_elbow_joint": 0.8,
            "left_wrist_roll_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            # Fixed wrist joints (not in URDF, excluded from robot)
            # "left_wrist_pitch_joint": 0.0,
            # "right_wrist_pitch_joint": 0.0,
            # "left_wrist_yaw_joint": 0.0,
            # "right_wrist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_roll_joint": 220.0,
                ".*_hip_yaw_joint": 220.0,
                ".*_hip_pitch_joint": 220.0,
                ".*_knee_joint": 320.0,
            },
            velocity_limit_sim={
                ".*_hip_roll_joint": 21.57,
                ".*_hip_yaw_joint": 21.57,
                ".*_hip_pitch_joint": 21.57,
                ".*_knee_joint": 19.89,
            },
            stiffness={
                ".*_hip_roll_joint": 120,
                ".*_hip_yaw_joint": 80,
                ".*_hip_pitch_joint": 80,
                ".*_knee_joint": 250.0,
            },
            damping={
                ".*_hip_roll_joint": 4.0,
                ".*_hip_yaw_joint": 3.0,
                ".*_hip_pitch_joint": 3.0,
                ".*_knee_joint": 8.0,
            },
            armature={
                ".*_hip_roll_joint":ARMATURE_PFP_93_65,
                ".*_hip_yaw_joint": ARMATURE_PFP_93_65,
                ".*_hip_pitch_joint":ARMATURE_PFP_93_65,
                ".*_knee_joint": ARMATURE_PFP_110_75,
            },
            # friction={
            #     ".*_hip_roll_joint":0.15,
            #     ".*_hip_yaw_joint": 0.15,
            #     ".*_hip_pitch_joint":0.15,
            #     ".*_knee_joint": 0.15,
            # },
            # dynamic_friction={
            #     ".*_hip_roll_joint":0.1,
            #     ".*_hip_yaw_joint": 0.1,
            #     ".*_hip_pitch_joint":0.1,
            #     ".*_knee_joint": 0.1,
            # },
        ),
        "feet": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 120,#115,
                ".*_ankle_roll_joint": 54,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 30.0,
                ".*_ankle_roll_joint": 20.0,
            },
            stiffness={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 2.0,
            },
            armature={
                ".*_ankle_pitch_joint": ARMATURE_PFP_78_58*5.1,
                ".*_ankle_roll_joint": ARMATURE_PFP_78_58*1.53,
            },
            # friction={
            #     ".*_ankle_pitch_joint": 0.25,
            #     ".*_ankle_roll_joint": 0.25,
            # },
            # dynamic_friction={
            #     ".*_ankle_pitch_joint": 0.2,
            #     ".*_ankle_roll_joint": 0.2,
            # },
        ),
        "waist": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim={
                "waist_yaw_joint": 220.0,
                "waist_roll_joint": 46.0,
                "waist_pitch_joint": 115,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 21.57,
                "waist_roll_joint": 20.0,
                "waist_pitch_joint": 30.0,
            },
            stiffness={
                "waist_yaw_joint": 85,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            damping={
                "waist_yaw_joint": 3.0,
                "waist_roll_joint": 2.0,
                "waist_pitch_joint": 2.0,
            },
            armature={
                "waist_yaw_joint":ARMATURE_PFP_93_65,
                "waist_roll_joint":ARMATURE_PFP_78_58*1.21,
                "waist_pitch_joint": ARMATURE_PFP_78_58*7.3,
            },
            # friction={
            #     "waist_yaw_joint": 0.15,
            #     "waist_roll_joint": 0.15,
            #     "waist_pitch_joint": 0.15,
            # },
            # dynamic_friction={
            #     "waist_yaw_joint": 0.1,
            #     "waist_roll_joint": 0.1,
            #     "waist_pitch_joint": 0.1,
            # },
        ),
        "arms": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                # ".*_wrist_pitch_joint",
                # ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 60.0,
                ".*_shoulder_roll_joint": 60.0,
                ".*_shoulder_yaw_joint": 24.0,
                ".*_elbow_joint": 24.0,
                ".*_wrist_roll_joint": 24.0,
                # ".*_wrist_pitch_joint": 6.0,
                # ".*_wrist_yaw_joint": 6.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 17.8,
                ".*_elbow_joint": 17.8,
                ".*_wrist_roll_joint": 17.8,
                # ".*_wrist_pitch_joint": 20.8,
                # ".*_wrist_yaw_joint": 20.8,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 30.0,
                ".*_elbow_joint": 30.0,
                ".*_wrist_roll_joint": 30.0,
                # ".*_wrist_pitch_joint": 20.0,
                # ".*_wrist_yaw_joint": 20.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 3.0,
                ".*_shoulder_roll_joint": 3.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
                ".*_wrist_roll_joint": 2.0,
                # ".*_wrist_pitch_joint": 2.0,
                # ".*_wrist_yaw_joint": 2.0,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_PFP_78_58,
                ".*_shoulder_roll_joint": ARMATURE_PFP_78_58,
                ".*_shoulder_yaw_joint": ARMATURE_PFP_59_60,
                ".*_elbow_joint":ARMATURE_PFP_59_60,
                ".*_wrist_roll_joint": ARMATURE_PFP_59_60,
                # ".*_wrist_pitch_joint": ARMATURE_PFP_41_48,
                # ".*_wrist_yaw_joint": ARMATURE_PFP_41_48,
            },
            # friction={
            #     ".*_shoulder_pitch_joint": 0.1,
            #     ".*_shoulder_roll_joint": 0.1,
            #     ".*_shoulder_yaw_joint": 0.1,
            #     ".*_elbow_joint": 0.1,
            #     ".*_wrist_roll_joint": 0.1,
            #     # ".*_wrist_pitch_joint": 0.1,
            #     # ".*_wrist_yaw_joint": 0.1,
            # },
            # dynamic_friction={
            #     ".*_shoulder_pitch_joint": 0.1,
            #     ".*_shoulder_roll_joint": 0.1,
            #     ".*_shoulder_yaw_joint": 0.1,
            #     ".*_elbow_joint": 0.1,
            #     ".*_wrist_roll_joint": 0.1,
            #     # ".*_wrist_pitch_joint": 0.1,
            #     # ".*_wrist_yaw_joint": 0.1,
            # },
        ),
    },
)

A3_ACTION_SCALE = {}
for a in A3_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            A3_ACTION_SCALE[n] = 0.5

# A3_ACTION_SCALE_FILTERED = {
#     k: v for k, v in A3_ACTION_SCALE.items()
#     if any(j for j in [
#         "hip_roll", "hip_yaw", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"
#     ] if j in k)
# }
# print(A3_ACTION_SCALE_FILTERED)

# Alias for compatibility with flat_env_cfg.py
A3_CFG = A3_CYLINDER_CFG
