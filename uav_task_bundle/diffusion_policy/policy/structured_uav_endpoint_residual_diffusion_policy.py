from contextlib import nullcontext
from typing import Dict, List

import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.structured_uav_endpoint_predictor import (
    StructuredUAVEndpointPredictor,
)
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


class StructuredUAVEndpointResidualDiffusionPolicy(
    StructuredUAVDiffusionTransformerLowdimPolicy
):
    def __init__(
        self,
        model: TransformerForDiffusion,
        noise_scheduler: DDPMScheduler,
        structured_obs_encoder: StructuredUAVObsEncoder,
        endpoint_predictor: StructuredUAVEndpointPredictor,
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
        action_target_type: str = "relative_future_position",
        freeze_endpoint_predictor: bool = True,
        endpoint_condition_dim: int = 256,
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
                "StructuredUAVEndpointResidualDiffusionPolicy only supports relative_future_position."
            )
        if not self.pred_action_steps_only:
            raise ValueError(
                "StructuredUAVEndpointResidualDiffusionPolicy requires pred_action_steps_only=True."
            )
        self.endpoint_predictor = endpoint_predictor
        self.freeze_endpoint_predictor = bool(freeze_endpoint_predictor)
        cond_hidden_dim = int(self.structured_obs_encoder.hidden_dim)
        endpoint_condition_dim = int(endpoint_condition_dim)
        self.endpoint_condition_mlp = nn.Sequential(
            nn.Linear(self.action_dim, endpoint_condition_dim),
            nn.Mish(),
            nn.Linear(endpoint_condition_dim, cond_hidden_dim),
        )
        self._set_endpoint_predictor_trainable(not self.freeze_endpoint_predictor)

    def _set_endpoint_predictor_trainable(self, trainable: bool):
        for param in self.endpoint_predictor.parameters():
            param.requires_grad = bool(trainable)
        if trainable:
            self.endpoint_predictor.train()
        else:
            self.endpoint_predictor.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_endpoint_predictor:
            self.endpoint_predictor.eval()
        return self

    def set_normalizer(self, normalizer: LinearNormalizer):
        super().set_normalizer(normalizer)
        self.endpoint_predictor.set_normalizer(normalizer)

    def get_optimizer(self, weight_decay: float, learning_rate: float, betas):
        params: List[torch.nn.Parameter] = []
        seen = set()
        modules = [self.model, self.structured_obs_encoder, self.endpoint_condition_mlp]
        if not self.freeze_endpoint_predictor:
            modules.append(self.endpoint_predictor)
        for module in modules:
            for param in module.parameters():
                if (not param.requires_grad) or (id(param) in seen):
                    continue
                seen.add(id(param))
                params.append(param)
        return torch.optim.AdamW(
            params=params,
            lr=learning_rate,
            betas=tuple(betas),
            weight_decay=weight_decay,
        )

    def _has_structured_inputs(self, data: Dict[str, torch.Tensor]) -> bool:
        return all(key in data for key in STRUCTURED_BATCH_KEYS)

    def _build_reference_path(self, endpoint_rel: torch.Tensor) -> torch.Tensor:
        step_scale = torch.arange(
            1,
            self.n_action_steps + 1,
            device=endpoint_rel.device,
            dtype=endpoint_rel.dtype,
        )
        step_scale = step_scale.view(1, self.n_action_steps, 1) / float(self.n_action_steps)
        return step_scale * endpoint_rel.unsqueeze(1)

    def _predict_endpoint(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        context = nullcontext()
        if self.freeze_endpoint_predictor:
            context = torch.no_grad()
        with context:
            endpoint_pred = self.endpoint_predictor.predict_endpoint(data)
        if self.freeze_endpoint_predictor:
            endpoint_pred = endpoint_pred.detach()
        return endpoint_pred

    def _encode_structured_cond_with_endpoint(self, data: Dict[str, torch.Tensor]):
        cond = self._encode_structured_cond(data)
        endpoint_pred = self._predict_endpoint(data).to(device=cond.device, dtype=cond.dtype)
        endpoint_emb = self.endpoint_condition_mlp(endpoint_pred)
        cond = cond + endpoint_emb.unsqueeze(1)
        reference_path = self._build_reference_path(endpoint_pred)
        return cond, endpoint_pred, reference_path

    def load_endpoint_predictor(self, checkpoint_path: str, strict: bool = True):
        payload = torch.load(checkpoint_path, map_location="cpu", pickle_module=dill)
        state_dict = payload
        if isinstance(payload, dict):
            if "state_dicts" in payload and isinstance(payload["state_dicts"], dict):
                if "model" in payload["state_dicts"]:
                    state_dict = payload["state_dicts"]["model"]
                elif "ema_model" in payload["state_dicts"]:
                    state_dict = payload["state_dicts"]["ema_model"]
            elif "model" in payload and isinstance(payload["model"], dict):
                state_dict = payload["model"]
        self.endpoint_predictor.load_state_dict(state_dict, strict=strict)
        self.endpoint_predictor.set_normalizer(self.normalizer)
        if self.freeze_endpoint_predictor:
            self._set_endpoint_predictor_trainable(False)
        return state_dict

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        generator: torch.Generator = None,
    ) -> Dict[str, torch.Tensor]:
        if not self._has_structured_inputs(obs_dict):
            raise ValueError(
                "StructuredUAVEndpointResidualDiffusionPolicy requires structured observation fields."
            )
        if "residual_action" not in self.normalizer.params_dict:
            raise RuntimeError("Normalizer is missing \"residual_action\" stats.")

        batch_size = obs_dict["agent_obs"].shape[0]
        device = self.device
        dtype = self.dtype
        cond, endpoint_pred, reference_path = self._encode_structured_cond_with_endpoint(obs_dict)

        cond_data = torch.zeros(
            size=(batch_size, self.n_action_steps, self.action_dim),
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
        residual_pred = self.normalizer["residual_action"].unnormalize(nresidual_pred)
        action_pred = reference_path + residual_pred
        return {
            "action": action_pred,
            "action_pred": action_pred,
            "endpoint_pred": endpoint_pred,
            "reference_path": reference_path,
            "residual_action": residual_pred,
        }

    def compute_loss(self, batch):
        if not self._has_structured_inputs(batch):
            raise ValueError(
                "StructuredUAVEndpointResidualDiffusionPolicy requires structured batch fields."
            )
        if "residual_action" not in self.normalizer.params_dict:
            raise RuntimeError("Normalizer is missing \"residual_action\" stats.")

        start, end = self.get_action_window_indices()
        future_relpos = batch["action"][:, start:end, :]
        cond, _endpoint_pred, reference_path = self._encode_structured_cond_with_endpoint(batch)
        residual_target = future_relpos - reference_path
        trajectory = self.normalizer["residual_action"].normalize(residual_target)

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
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        return loss.mean()
