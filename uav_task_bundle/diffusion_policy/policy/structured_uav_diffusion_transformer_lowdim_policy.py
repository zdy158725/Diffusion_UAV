from typing import Dict

import torch
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.diffusion.structured_uav_obs_encoder import (
    StructuredUAVObsEncoder,
)
from diffusion_policy.model.diffusion.transformer_for_diffusion import (
    TransformerForDiffusion,
)
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import (
    DiffusionTransformerLowdimPolicy,
)


STRUCTURED_BATCH_KEYS = (
    "agent_obs",
    "agent_team",
    "agent_valid",
    "agent_social_mask",
    "agent_role_id",
)


class StructuredUAVDiffusionTransformerLowdimPolicy(
    DiffusionTransformerLowdimPolicy
):
    def __init__(
        self,
        model: TransformerForDiffusion,
        noise_scheduler: DDPMScheduler,
        structured_obs_encoder: StructuredUAVObsEncoder,
        horizon,
        obs_dim,
        action_dim,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_cond=False,
        pred_action_steps_only=False,
        velocity_loss_weight: float = 0.0,
        short_horizon_loss_weight: float = 0.0,
        short_horizon_focus_steps: int = 4,
        short_horizon_focus_gain: float = 3.0,
        terminal_pos_loss_weight: float = 0.0,
        terminal_pos_loss_power: float = 1.0,
        relative_path_loss_weight: float = 0.0,
        action_target_type: str = "delta_position",
        **kwargs,
    ):
        super().__init__(
            model=model,
            noise_scheduler=noise_scheduler,
            horizon=horizon,
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            obs_as_cond=obs_as_cond,
            pred_action_steps_only=pred_action_steps_only,
            velocity_loss_weight=velocity_loss_weight,
            short_horizon_loss_weight=short_horizon_loss_weight,
            short_horizon_focus_steps=short_horizon_focus_steps,
            short_horizon_focus_gain=short_horizon_focus_gain,
            terminal_pos_loss_weight=terminal_pos_loss_weight,
            terminal_pos_loss_power=terminal_pos_loss_power,
            relative_path_loss_weight=relative_path_loss_weight,
            action_target_type=action_target_type,
            **kwargs,
        )
        if not self.obs_as_cond:
            raise ValueError("StructuredUAVDiffusionTransformerLowdimPolicy requires obs_as_cond=True")
        self.structured_obs_encoder = structured_obs_encoder

    def _has_structured_inputs(self, data: Dict[str, torch.Tensor]) -> bool:
        return all(key in data for key in STRUCTURED_BATCH_KEYS)

    def _encode_structured_cond(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "agent_obs" not in self.normalizer.params_dict:
            raise RuntimeError("Normalizer is missing 'agent_obs' stats for structured cond.")
        agent_obs = self.normalizer["agent_obs"].normalize(data["agent_obs"])
        return self.structured_obs_encoder(
            agent_obs=agent_obs,
            agent_team=data["agent_team"],
            agent_valid=data["agent_valid"],
            agent_social_mask=data["agent_social_mask"],
            agent_role_id=data["agent_role_id"],
        )

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        generator: torch.Generator = None,
    ) -> Dict[str, torch.Tensor]:
        if not self._has_structured_inputs(obs_dict):
            return super().predict_action(obs_dict=obs_dict, generator=generator)

        batch_size = obs_dict["agent_obs"].shape[0]
        device = self.device
        dtype = self.dtype
        cond = self._encode_structured_cond(obs_dict)

        shape = (batch_size, self.horizon, self.action_dim)
        if self.pred_action_steps_only:
            shape = (batch_size, self.n_action_steps, self.action_dim)
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond,
            generator=generator,
            token_offset=self.get_action_token_offset(),
            **self.kwargs,
        )

        naction_pred = nsample[..., : self.action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start, end = self.get_action_window_indices()
            action = action_pred[:, start:end]

        return {
            "action": action,
            "action_pred": action_pred,
        }

    def compute_loss(self, batch):
        if not self._has_structured_inputs(batch):
            return super().compute_loss(batch)

        action = self.normalizer["action"].normalize(batch["action"])
        cond = self._encode_structured_cond(batch)
        trajectory = action
        if self.pred_action_steps_only:
            start, end = self.get_action_window_indices()
            trajectory = action[:, start:end]

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
        total_loss = loss.mean()

        def get_eval_action_tensors(pred_tensor, target_tensor, mask_tensor=None):
            if self.pred_action_steps_only:
                return pred_tensor, target_tensor, mask_tensor

            total_steps = target_tensor.shape[1]
            start = min(max(self.n_obs_steps, 0), total_steps - 1)
            end = min(start + self.n_action_steps, total_steps)
            focus_pred = pred_tensor[:, start:end, :]
            focus_target = target_tensor[:, start:end, :]
            focus_mask = None
            if mask_tensor is not None:
                focus_mask = mask_tensor[:, start:end, :]
            return focus_pred, focus_target, focus_mask

        if self.short_horizon_loss_weight > 0.0 and trajectory.shape[1] > 0:
            total_steps = trajectory.shape[1]
            focus_start = 0 if self.pred_action_steps_only else min(max(self.n_obs_steps, 0), total_steps - 1)
            focus_steps = max(self.short_horizon_focus_steps, 1)
            focus_end = min(focus_start + focus_steps, total_steps)

            step_weights = torch.ones(
                (1, total_steps, 1), device=trajectory.device, dtype=trajectory.dtype
            )
            if focus_end > focus_start:
                step_weights[:, focus_start:focus_end, :] = self.short_horizon_focus_gain

            recon_err = (x0_pred - trajectory) ** 2
            recon_err = recon_err * loss_mask.type(recon_err.dtype) * step_weights
            recon_denom = (loss_mask.type(recon_err.dtype) * step_weights).sum().clamp(min=1.0)
            short_horizon_loss = recon_err.sum() / recon_denom
            total_loss = total_loss + self.short_horizon_loss_weight * short_horizon_loss

        if self.velocity_loss_weight > 0.0 and trajectory.shape[1] > 1:
            vel_pred = x0_pred[:, 1:, :] - x0_pred[:, :-1, :]
            vel_target = trajectory[:, 1:, :] - trajectory[:, :-1, :]
            vel_mask = loss_mask[:, 1:, :] & loss_mask[:, :-1, :]

            vel_err = (vel_pred - vel_target) ** 2
            vel_err = vel_err * vel_mask.type(vel_err.dtype)
            vel_denom = vel_mask.type(vel_err.dtype).sum().clamp(min=1.0)
            vel_loss = vel_err.sum() / vel_denom
            total_loss = total_loss + self.velocity_loss_weight * vel_loss

        if self.terminal_pos_loss_weight > 0.0 and trajectory.shape[1] > 0:
            focus_pred, focus_target, focus_mask = get_eval_action_tensors(
                x0_pred, trajectory, loss_mask
            )
            if focus_pred.shape[1] > 0:
                if self.action_target_type == "relative_future_position":
                    pred_pos = self.normalizer["action"].unnormalize(focus_pred)
                    target_pos = self.normalizer["action"].unnormalize(focus_target)
                else:
                    pred_pos = torch.cumsum(focus_pred, dim=1)
                    target_pos = torch.cumsum(focus_target, dim=1)
                final_sq_err = (pred_pos[:, -1, :] - target_pos[:, -1, :]) ** 2
                final_mask = focus_mask[:, -1, :].type(final_sq_err.dtype)
                denom = final_mask.sum().clamp(min=1.0)
                terminal_pos_loss = (final_sq_err * final_mask).sum() / denom
                if self.terminal_pos_loss_power != 1.0:
                    terminal_pos_loss = torch.pow(
                        terminal_pos_loss + 1e-8, self.terminal_pos_loss_power
                    )
                total_loss = total_loss + self.terminal_pos_loss_weight * terminal_pos_loss

        if self.relative_path_loss_weight > 0.0 and trajectory.shape[1] > 0:
            focus_pred, focus_target, _ = get_eval_action_tensors(x0_pred, trajectory)
            if focus_pred.shape[1] > 0:
                pred_action = self.normalizer["action"].unnormalize(focus_pred)
                target_action = self.normalizer["action"].unnormalize(focus_target)
                if self.action_target_type == "relative_future_position":
                    current_pos = batch["obs"][
                        :, self.get_action_anchor_obs_index(), : self.action_dim
                    ].to(device=pred_action.device, dtype=pred_action.dtype)
                    pred_pos = current_pos[:, None, :] + pred_action
                    target_pos = current_pos[:, None, :] + target_action
                    gt_step = torch.zeros_like(target_pos)
                    gt_step[:, 0, :] = target_pos[:, 0, :] - current_pos
                    if target_pos.shape[1] > 1:
                        gt_step[:, 1:, :] = target_pos[:, 1:, :] - target_pos[:, :-1, :]
                    gt_path_len = torch.linalg.norm(gt_step, dim=-1).sum(dim=1).clamp(min=1e-6)
                else:
                    pred_pos = torch.cumsum(pred_action, dim=1)
                    target_pos = torch.cumsum(target_action, dim=1)
                    gt_path_len = torch.linalg.norm(target_action, dim=-1).sum(dim=1).clamp(min=1e-6)
                step_err = torch.linalg.norm(pred_pos - target_pos, dim=-1)
                relative_path_loss = (step_err.sum(dim=1) / gt_path_len).mean()
                total_loss = total_loss + self.relative_path_loss_weight * relative_path_loss

        return total_loss
