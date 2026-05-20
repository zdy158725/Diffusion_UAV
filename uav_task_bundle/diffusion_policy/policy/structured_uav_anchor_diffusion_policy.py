from typing import Dict, Sequence

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
from diffusion_policy.policy.structured_uav_diffusion_transformer_lowdim_policy import (
    STRUCTURED_BATCH_KEYS,
    StructuredUAVDiffusionTransformerLowdimPolicy,
)


class StructuredUAVAnchorDiffusionPolicy(StructuredUAVDiffusionTransformerLowdimPolicy):
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
        full_future_steps: int,
        anchor_steps: Sequence[int] = (6, 12, 18, 24, 30),
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
        action_target_type: str = "relative_future_position",
        stage2_start_frac: float = 0.6,
        traj_ade_loss_weight_max: float = 5e-4,
        traj_fde_loss_weight_max: float = 1e-3,
        curriculum_mode: str = "linear_ramp",
        **kwargs,
    ):
        super().__init__(
            model=model,
            noise_scheduler=noise_scheduler,
            structured_obs_encoder=structured_obs_encoder,
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
        if self.action_target_type != "relative_future_position":
            raise ValueError(
                "StructuredUAVAnchorDiffusionPolicy only supports relative_future_position."
            )
        if not self.pred_action_steps_only:
            raise ValueError(
                "StructuredUAVAnchorDiffusionPolicy requires pred_action_steps_only=True."
            )
        self.anchor_horizon = int(n_action_steps)
        self.full_future_steps = int(full_future_steps)
        self.anchor_steps = tuple(int(step) for step in anchor_steps)
        if len(self.anchor_steps) != self.anchor_horizon:
            raise ValueError(
                "anchor_steps length must equal anchor_horizon="
                f"{self.anchor_horizon}, got {len(self.anchor_steps)}."
            )
        if tuple(sorted(self.anchor_steps)) != self.anchor_steps:
            raise ValueError("anchor_steps must be strictly increasing.")
        if self.anchor_steps[0] <= 0:
            raise ValueError("anchor_steps must be positive 1-indexed future steps.")
        if self.anchor_steps[-1] != self.full_future_steps:
            raise ValueError(
                "The last anchor step must equal full_future_steps="
                f"{self.full_future_steps}, got {self.anchor_steps[-1]}."
            )
        self.stage2_start_frac = float(stage2_start_frac)
        if not (0.0 <= self.stage2_start_frac < 1.0):
            raise ValueError(
                "stage2_start_frac must be in [0, 1), got "
                f"{self.stage2_start_frac}."
            )
        self.traj_ade_loss_weight_max = float(traj_ade_loss_weight_max)
        self.traj_fde_loss_weight_max = float(traj_fde_loss_weight_max)
        if self.traj_ade_loss_weight_max < 0.0 or self.traj_fde_loss_weight_max < 0.0:
            raise ValueError("Trajectory curriculum loss weights must be non-negative.")
        self.curriculum_mode = str(curriculum_mode)
        if self.curriculum_mode != "linear_ramp":
            raise ValueError(
                "StructuredUAVAnchorDiffusionPolicy only supports "
                f'curriculum_mode="linear_ramp", got {self.curriculum_mode}.'
            )
        self.training_progress = 0.0
        self._loss_debug: Dict[str, float] = {}

    def get_action_window_indices(self):
        start = self.n_obs_steps
        end = start + self.full_future_steps
        return start, end

    def get_action_token_offset(self) -> int:
        # Anchor diffusion models a compact 5-step coarse sequence, not the
        # original 30-step future window, so decoder positions start at 0.
        return 0

    def _has_structured_inputs(self, data: Dict[str, torch.Tensor]) -> bool:
        return all(key in data for key in STRUCTURED_BATCH_KEYS)

    def _build_piecewise_linear_path(self, anchor_relpos: torch.Tensor) -> torch.Tensor:
        batch_size = anchor_relpos.shape[0]
        future_relpos = torch.zeros(
            (batch_size, self.full_future_steps, self.action_dim),
            device=anchor_relpos.device,
            dtype=anchor_relpos.dtype,
        )
        prev_step = 0
        prev_point = torch.zeros(
            (batch_size, self.action_dim),
            device=anchor_relpos.device,
            dtype=anchor_relpos.dtype,
        )
        for anchor_idx, step in enumerate(self.anchor_steps):
            point = anchor_relpos[:, anchor_idx, :]
            segment_len = step - prev_step
            step_scale = torch.arange(
                1,
                segment_len + 1,
                device=anchor_relpos.device,
                dtype=anchor_relpos.dtype,
            ).view(1, segment_len, 1) / float(segment_len)
            segment = prev_point.unsqueeze(1) + step_scale * (
                point - prev_point
            ).unsqueeze(1)
            future_relpos[:, prev_step:step, :] = segment
            prev_step = step
            prev_point = point
        return future_relpos

    def set_training_progress(self, progress: float) -> None:
        self.training_progress = float(max(0.0, min(1.0, progress)))

    def get_loss_debug_dict(self) -> Dict[str, float]:
        return dict(self._loss_debug)

    def _compute_curriculum_ramp(self) -> float:
        if self.training_progress < self.stage2_start_frac:
            return 0.0
        denom = max(1.0 - self.stage2_start_frac, 1e-8)
        ramp = (self.training_progress - self.stage2_start_frac) / denom
        return float(max(0.0, min(1.0, ramp)))

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        generator: torch.Generator = None,
    ) -> Dict[str, torch.Tensor]:
        if not self._has_structured_inputs(obs_dict):
            raise ValueError(
                "StructuredUAVAnchorDiffusionPolicy requires structured observation fields."
            )
        if "waypoint_target" not in self.normalizer.params_dict:
            raise RuntimeError('Normalizer is missing "waypoint_target" stats.')

        batch_size = obs_dict["agent_obs"].shape[0]
        device = self.device
        dtype = self.dtype
        cond = self._encode_structured_cond(obs_dict)

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
        nanchor_pred = nsample[..., : self.action_dim]
        anchor_pred = self.normalizer["waypoint_target"].unnormalize(nanchor_pred)
        action_pred = self._build_piecewise_linear_path(anchor_pred)
        return {
            "action": action_pred,
            "action_pred": action_pred,
            "anchor_pred": anchor_pred,
            "anchor_steps": torch.as_tensor(
                self.anchor_steps, device=device, dtype=torch.long
            ),
        }

    def compute_loss(self, batch):
        if not self._has_structured_inputs(batch):
            raise ValueError(
                "StructuredUAVAnchorDiffusionPolicy requires structured batch fields."
            )
        if "waypoint_target" not in batch:
            raise ValueError("Batch is missing waypoint_target for anchor diffusion.")
        if "waypoint_target" not in self.normalizer.params_dict:
            raise RuntimeError('Normalizer is missing "waypoint_target" stats.')

        cond = self._encode_structured_cond(batch)
        trajectory = self.normalizer["waypoint_target"].normalize(batch["waypoint_target"])
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

        pred_anchor_phys = self.normalizer["waypoint_target"].unnormalize(x0_pred)
        pred_full_relpos = self._build_piecewise_linear_path(pred_anchor_phys)
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
            "curriculum_progress": float(self.training_progress),
            "curriculum_ramp": float(curriculum_ramp),
            "curriculum_w_ade": float(curriculum_w_ade),
            "curriculum_w_fde": float(curriculum_w_fde),
        }
        return total_loss
