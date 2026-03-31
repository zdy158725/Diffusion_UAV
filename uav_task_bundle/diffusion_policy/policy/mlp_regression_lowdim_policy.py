from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class MLPRegressionLowdimPolicy(BaseLowdimPolicy):
    def __init__(
        self,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.0,
        obs_as_cond: bool = True,
        pred_action_steps_only: bool = True,
    ):
        super().__init__()
        if not obs_as_cond:
            raise ValueError("MLPRegressionLowdimPolicy requires obs_as_cond=True")
        if not pred_action_steps_only:
            raise ValueError("MLPRegressionLowdimPolicy requires pred_action_steps_only=True")

        self.normalizer = LinearNormalizer()
        self.horizon = int(horizon)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.obs_as_cond = bool(obs_as_cond)
        self.pred_action_steps_only = bool(pred_action_steps_only)

        layers = []
        in_dim = self.n_obs_steps * self.obs_dim
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.n_action_steps * self.action_dim))
        self.network = nn.Sequential(*layers)

    def get_action_window_indices(self) -> Tuple[int, int]:
        start = self.n_obs_steps
        end = start + self.n_action_steps
        return start, end

    def get_action_anchor_obs_index(self) -> int:
        return self.n_obs_steps - 1

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
            betas=tuple(betas),
        )

    def _predict_normalized_action(self, nobs: torch.Tensor) -> torch.Tensor:
        cond = nobs[:, : self.n_obs_steps, :]
        flat = cond.reshape(cond.shape[0], -1)
        pred = self.network(flat)
        return pred.reshape(-1, self.n_action_steps, self.action_dim)

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        generator: torch.Generator = None,
    ) -> Dict[str, torch.Tensor]:
        assert "obs" in obs_dict
        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        naction = self._predict_normalized_action(nobs)
        action = self.normalizer["action"].unnormalize(naction)
        return {
            "action": action,
            "action_pred": action,
        }

    def compute_loss(self, batch):
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch["obs"]
        naction = nbatch["action"]
        start, end = self.get_action_window_indices()
        target = naction[:, start:end]
        pred = self._predict_normalized_action(nobs)
        return F.mse_loss(pred, target)
