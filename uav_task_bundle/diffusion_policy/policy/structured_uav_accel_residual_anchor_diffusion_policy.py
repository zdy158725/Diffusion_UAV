from typing import Dict, Optional

import torch
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.policy.structured_uav_anchor_diffusion_policy import (
    StructuredUAVAnchorDiffusionPolicy,
)
from diffusion_policy.policy.structured_uav_diffusion_transformer_lowdim_policy import (
    STRUCTURED_BATCH_KEYS,
)


class StructuredUAVAccelResidualAnchorDiffusionPolicy(
    StructuredUAVAnchorDiffusionPolicy
):
    def __init__(
        self,
        *args,
        base_mode: str = "accel",
        base_velocity_window: int = 3,
        base_accel_window: int = 3,
        base_accel_clip: Optional[float] = None,
        residual_normalizer_key: str = "accel_residual_waypoint_target",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.base_mode = str(base_mode)
        if self.base_mode != "accel":
            raise ValueError(f'Only base_mode="accel" is supported, got {self.base_mode}.')

        self.base_velocity_window = int(base_velocity_window)
        self.base_accel_window = int(base_accel_window)
        if self.base_velocity_window <= 0 or self.base_accel_window <= 0:
            raise ValueError("Base velocity/accel windows must be positive.")

        self.base_accel_clip = (
            None if base_accel_clip is None else float(base_accel_clip)
        )
        if self.base_accel_clip is not None and self.base_accel_clip <= 0.0:
            raise ValueError("base_accel_clip must be positive or null.")
        self.residual_normalizer_key = str(residual_normalizer_key)

    def _has_structured_inputs(self, data: Dict[str, torch.Tensor]) -> bool:
        return all(key in data for key in STRUCTURED_BATCH_KEYS)

    def _build_accel_base_anchor(
        self,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if "agent_obs" not in data:
            raise ValueError("Batch is missing agent_obs for accel residual base.")

        agent_obs = data["agent_obs"].to(device=self.device, dtype=self.dtype)
        target_pos = agent_obs[:, :, 0, : self.action_dim]
        if target_pos.shape[1] < 3:
            raise ValueError(
                "Accel residual base requires at least 3 observation steps, got "
                f"{target_pos.shape[1]}."
            )

        deltas = target_pos[:, 1:, :] - target_pos[:, :-1, :]
        velocity_window = min(self.base_velocity_window, deltas.shape[1])
        velocity = deltas[:, -velocity_window:, :].mean(dim=1)

        delta_deltas = deltas[:, 1:, :] - deltas[:, :-1, :]
        accel_window = min(self.base_accel_window, delta_deltas.shape[1])
        accel = delta_deltas[:, -accel_window:, :].mean(dim=1)
        if self.base_accel_clip is not None:
            accel_norm = torch.linalg.norm(accel, dim=-1, keepdim=True).clamp(min=1e-8)
            clip_scale = torch.clamp(self.base_accel_clip / accel_norm, max=1.0)
            accel = accel * clip_scale

        steps = torch.as_tensor(
            self.anchor_steps,
            device=target_pos.device,
            dtype=target_pos.dtype,
        ).view(1, self.anchor_horizon, 1)
        return velocity.unsqueeze(1) * steps + 0.5 * accel.unsqueeze(1) * steps.square()

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        generator: torch.Generator = None,
    ) -> Dict[str, torch.Tensor]:
        if not self._has_structured_inputs(obs_dict):
            raise ValueError(
                "StructuredUAVAccelResidualAnchorDiffusionPolicy requires structured observation fields."
            )
        if self.residual_normalizer_key not in self.normalizer.params_dict:
            raise RuntimeError(
                f'Normalizer is missing "{self.residual_normalizer_key}" stats.'
            )

        batch_size = obs_dict["agent_obs"].shape[0]
        device = self.device
        dtype = self.dtype
        cond = self._encode_structured_cond(obs_dict)
        base_anchor = self._build_accel_base_anchor(obs_dict)

        cond_data = torch.zeros(
            size=(batch_size, self.anchor_horizon, self.action_dim),
            device=device,
            dtype=dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond,
            generator=generator,
            token_offset=self.get_action_token_offset(),
            **self.kwargs,
        )
        nresidual_pred = nsample[..., : self.action_dim]
        residual_anchor_pred = self.normalizer[self.residual_normalizer_key].unnormalize(
            nresidual_pred
        )
        anchor_pred = base_anchor + residual_anchor_pred
        action_pred = self._build_piecewise_linear_path(anchor_pred)
        base_action = self._build_piecewise_linear_path(base_anchor)
        return {
            "action": action_pred,
            "action_pred": action_pred,
            "anchor_pred": anchor_pred,
            "anchor_steps": torch.as_tensor(
                self.anchor_steps, device=device, dtype=torch.long
            ),
            "base_action": base_action,
            "base_anchor": base_anchor,
            "residual_anchor_pred": residual_anchor_pred,
        }

    def compute_loss(self, batch):
        if not self._has_structured_inputs(batch):
            raise ValueError(
                "StructuredUAVAccelResidualAnchorDiffusionPolicy requires structured batch fields."
            )
        if "waypoint_target" not in batch:
            raise ValueError("Batch is missing waypoint_target for residual anchor diffusion.")
        if self.residual_normalizer_key not in batch:
            raise ValueError(
                f"Batch is missing {self.residual_normalizer_key} for residual anchor diffusion."
            )
        if self.residual_normalizer_key not in self.normalizer.params_dict:
            raise RuntimeError(
                f'Normalizer is missing "{self.residual_normalizer_key}" stats.'
            )

        cond = self._encode_structured_cond(batch)
        base_anchor = self._build_accel_base_anchor(batch)
        residual_target = batch[self.residual_normalizer_key].to(
            device=base_anchor.device,
            dtype=base_anchor.dtype,
        )
        trajectory = self.normalizer[self.residual_normalizer_key].normalize(
            residual_target
        )
        condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        batch_size = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.model(
            noisy_trajectory,
            timesteps,
            cond,
            token_offset=self.get_action_token_offset(),
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
            alpha_t = self.noise_scheduler.alphas_cumprod.to(
                device=trajectory.device, dtype=trajectory.dtype
            )[timesteps]
            alpha_t = alpha_t.view(-1, *([1] * (trajectory.ndim - 1)))
            x0_pred = (
                noisy_trajectory - torch.sqrt(1.0 - alpha_t) * pred
            ) / torch.sqrt(alpha_t)
        elif pred_type == "sample":
            target = trajectory
            x0_pred = pred
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss_diff = loss.mean()

        pred_residual_phys = self.normalizer[self.residual_normalizer_key].unnormalize(
            x0_pred
        )
        pred_anchor_phys = base_anchor + pred_residual_phys
        pred_full_relpos = self._build_piecewise_linear_path(pred_anchor_phys)
        base_full_relpos = self._build_piecewise_linear_path(base_anchor)
        start, end = self.get_action_window_indices()
        gt_full_relpos = batch["action"][:, start:end].to(
            device=pred_full_relpos.device,
            dtype=pred_full_relpos.dtype,
        )

        traj_step_error = torch.linalg.norm(pred_full_relpos - gt_full_relpos, dim=-1)
        loss_traj_ade = traj_step_error.mean(dim=1).mean()
        loss_traj_fde = torch.linalg.norm(
            pred_full_relpos[:, -1, :] - gt_full_relpos[:, -1, :], dim=-1
        ).mean()

        base_step_error = torch.linalg.norm(base_full_relpos - gt_full_relpos, dim=-1)
        base_traj_ade = base_step_error.mean(dim=1).mean()
        base_traj_fde = torch.linalg.norm(
            base_full_relpos[:, -1, :] - gt_full_relpos[:, -1, :], dim=-1
        ).mean()
        residual_anchor_mae = torch.mean(torch.abs(pred_residual_phys - residual_target))

        curriculum_ramp = self._compute_curriculum_ramp()
        curriculum_w_ade = curriculum_ramp * self.traj_ade_loss_weight_max
        curriculum_w_fde = curriculum_ramp * self.traj_fde_loss_weight_max

        total_loss = loss_diff
        if curriculum_ramp > 0.0:
            total_loss = (
                total_loss
                + curriculum_w_ade * loss_traj_ade
                + curriculum_w_fde * loss_traj_fde
            )

        self._loss_debug = {
            "loss_diff": float(loss_diff.detach().item()),
            "loss_traj_ade": float(loss_traj_ade.detach().item()),
            "loss_traj_fde": float(loss_traj_fde.detach().item()),
            "base_traj_ade": float(base_traj_ade.detach().item()),
            "base_traj_fde": float(base_traj_fde.detach().item()),
            "pred_vs_base_ade_gain": float(
                (base_traj_ade - loss_traj_ade).detach().item()
            ),
            "residual_anchor_mae": float(residual_anchor_mae.detach().item()),
            "curriculum_progress": float(self.training_progress),
            "curriculum_ramp": float(curriculum_ramp),
            "curriculum_w_ade": float(curriculum_w_ade),
            "curriculum_w_fde": float(curriculum_w_fde),
        }
        return total_loss
