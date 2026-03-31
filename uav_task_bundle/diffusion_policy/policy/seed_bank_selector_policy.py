from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer


class SeedBankSelectorPolicy(ModuleAttrMixin):
    def __init__(
        self,
        n_obs_steps: int,
        obs_dim: int,
        n_action_steps: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.normalizer = LinearNormalizer()
        self.n_obs_steps = int(n_obs_steps)
        self.obs_dim = int(obs_dim)
        self.n_action_steps = int(n_action_steps)
        self.action_dim = int(action_dim)

        in_dim = (
            self.n_obs_steps * self.obs_dim
            + 2 * self.n_action_steps * self.action_dim
        )
        layers = []
        current_dim = in_dim
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

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

    def _build_feature_tensor(
        self,
        obs_hist: torch.Tensor,
        cand_action: torch.Tensor,
        cand_rel_pos: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_candidates = cand_action.shape[:2]
        obs_hist = obs_hist[:, None, :, :].expand(-1, num_candidates, -1, -1)

        nobs_hist = self.normalizer["obs_hist"].normalize(obs_hist)
        ncand_action = self.normalizer["cand_action"].normalize(cand_action)
        ncand_rel_pos = self.normalizer["cand_rel_pos"].normalize(cand_rel_pos)

        return torch.cat(
            [
                nobs_hist.reshape(batch_size, num_candidates, -1),
                ncand_action.reshape(batch_size, num_candidates, -1),
                ncand_rel_pos.reshape(batch_size, num_candidates, -1),
            ],
            dim=-1,
        )

    def predict_logits(
        self,
        obs_hist: torch.Tensor,
        cand_action: torch.Tensor,
        cand_rel_pos: torch.Tensor,
    ) -> torch.Tensor:
        features = self._build_feature_tensor(obs_hist, cand_action, cand_rel_pos)
        logits = self.network(features.reshape(-1, features.shape[-1]))
        return logits.reshape(features.shape[0], features.shape[1])

    def predict_probs(
        self,
        obs_hist: torch.Tensor,
        cand_action: torch.Tensor,
        cand_rel_pos: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.predict_logits(obs_hist, cand_action, cand_rel_pos)
        return torch.softmax(logits, dim=1)

    def select_best(
        self,
        obs_hist: torch.Tensor,
        cand_action: torch.Tensor,
        cand_rel_pos: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits = self.predict_logits(obs_hist, cand_action, cand_rel_pos)
        probs = torch.softmax(logits, dim=1)
        best_idx = torch.argmax(probs, dim=1)
        return {
            "candidate_logits": logits,
            "candidate_probs": probs,
            "best_idx": best_idx,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.predict_logits(
            obs_hist=batch["obs_hist"],
            cand_action=batch["cand_action"],
            cand_rel_pos=batch["cand_rel_pos"],
        )
        target = batch["best_idx"]
        return F.cross_entropy(logits, target)
