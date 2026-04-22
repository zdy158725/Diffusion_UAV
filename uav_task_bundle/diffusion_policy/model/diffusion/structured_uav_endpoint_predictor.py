from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.structured_uav_obs_encoder import (
    StructuredUAVObsEncoder,
)
from diffusion_policy.policy.structured_uav_diffusion_transformer_lowdim_policy import (
    STRUCTURED_BATCH_KEYS,
)


class StructuredUAVEndpointPredictor(nn.Module):
    def __init__(
        self,
        structured_obs_encoder: StructuredUAVObsEncoder,
        endpoint_dim: int = 3,
        hidden_dim: int = 256,
        loss_type: str = "mse",
    ) -> None:
        super().__init__()
        self.structured_obs_encoder = structured_obs_encoder
        self.endpoint_dim = int(endpoint_dim)
        self.hidden_dim = int(hidden_dim)
        self.loss_type = str(loss_type)
        if self.loss_type not in {"mse", "smooth_l1"}:
            raise ValueError(f"Unsupported loss_type={self.loss_type}")

        self.endpoint_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Mish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Mish(),
            nn.Linear(self.hidden_dim, self.endpoint_dim),
        )
        self.normalizer = LinearNormalizer()

    def _has_structured_inputs(self, data: Dict[str, torch.Tensor]) -> bool:
        return all(key in data for key in STRUCTURED_BATCH_KEYS)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(self, weight_decay: float, learning_rate: float, betas):
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=learning_rate,
            betas=tuple(betas),
            weight_decay=weight_decay,
        )

    def _encode_structured_cond(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self._has_structured_inputs(data):
            raise ValueError("Structured endpoint predictor requires structured batch fields.")
        if "agent_obs" not in self.normalizer.params_dict:
            raise RuntimeError("Normalizer is missing \"agent_obs\" stats for endpoint predictor.")
        agent_obs = self.normalizer["agent_obs"].normalize(data["agent_obs"])
        return self.structured_obs_encoder(
            agent_obs=agent_obs,
            agent_team=data["agent_team"],
            agent_valid=data["agent_valid"],
            agent_social_mask=data["agent_social_mask"],
            agent_role_id=data["agent_role_id"],
        )

    def predict_endpoint_normalized(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        cond = self._encode_structured_cond(data)
        last_token = cond[:, -1, :]
        mean_token = cond.mean(dim=1)
        summary = torch.cat([last_token, mean_token], dim=-1)
        return self.endpoint_head(summary)

    def predict_endpoint(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        endpoint_pred = self.predict_endpoint_normalized(data)
        if "endpoint_target" not in self.normalizer.params_dict:
            raise RuntimeError(
                "Normalizer is missing \"endpoint_target\" stats for endpoint predictor."
            )
        return self.normalizer["endpoint_target"].unnormalize(endpoint_pred)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "endpoint_target" not in batch:
            raise ValueError("Batch is missing endpoint_target for endpoint training.")
        if "endpoint_target" not in self.normalizer.params_dict:
            raise RuntimeError(
                "Normalizer is missing \"endpoint_target\" stats for endpoint predictor."
            )
        endpoint_target = self.normalizer["endpoint_target"].normalize(
            batch["endpoint_target"]
        )
        endpoint_pred = self.predict_endpoint_normalized(batch)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(endpoint_pred, endpoint_target)
        return F.mse_loss(endpoint_pred, endpoint_target)
