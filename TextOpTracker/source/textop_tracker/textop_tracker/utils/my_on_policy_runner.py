import os
import pickle
from typing import Optional

import numpy as np
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from textop_tracker.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

# class MyOnPolicyRunner(OnPolicyRunner):
#     def save(self, path: str, infos=None):
#         """Save the model and training information."""
#         super().save(path, infos)
#         if self.logger_type in ["wandb"]:
#             policy_path = path.split("model")[0]
#             filename = policy_path.split("/")[-2] + ".onnx"
#             export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
#             attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
#             wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device="cpu",
        registry_name: Optional[str] = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        unwrapped_env = self.env.unwrapped

        policy_path = path.split("model")[0]
        onnx_filename = "latest.onnx"
        export_motion_policy_as_onnx(
            unwrapped_env, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=onnx_filename
        )
        if self.logger_type in ["wandb"]:
            attach_onnx_metadata(unwrapped_env, wandb.run.name, path=policy_path, filename=onnx_filename)
            wandb.save(policy_path + onnx_filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None

        # For DEBUG:
        if unwrapped_env.command_manager.get_term("motion").cfg.enable_adaptive_sampling:
            fail_count = unwrapped_env.command_manager.get_term("motion").failed_motion_count.cpu().numpy()
            success_count = unwrapped_env.command_manager.get_term("motion").success_motion_count.cpu().numpy()
            total_count = fail_count + success_count
            p_fail = fail_count / (total_count + 1e-8)
            p_fail_sample_v2 = (p_fail**unwrapped_env.command_manager.get_term("motion").cfg.adaptive_beta)
            p_fail_sample_v2 = p_fail_sample_v2 / (p_fail_sample_v2.sum() + 1e-8)
            sampling_probabilities_v2 = (
                p_fail_sample_v2 * (1 - unwrapped_env.command_manager.get_term("motion").cfg.adaptive_uniform_ratio) +
                unwrapped_env.command_manager.get_term("motion").cfg.adaptive_uniform_ratio /
                float(unwrapped_env.command_manager.get_term("motion").num_motion)
            )
            adpsam_count = {
                "failed_motion_count": fail_count,
                "success_motion_count": success_count,
                "p_fail": p_fail,
                "p_fail_sample_v2": p_fail_sample_v2,
                "sampling_probabilities_v2": sampling_probabilities_v2,
            }
            # (adpsam_count, step=self.current_learning_iteration)
            pickle.dump(adpsam_count, open(path[:-len(".pt")] + "-adpsam_count.pkl", "wb"))
