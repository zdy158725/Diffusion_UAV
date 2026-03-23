import os
from typing import Dict

import torch
import zarr

from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class TrajectoryCandidateRerankerDataset(BaseLowdimDataset):
    def __init__(self, zarr_path: str):
        super().__init__()
        self.zarr_path = os.path.expanduser(zarr_path)
        self.root = zarr.open(self.zarr_path, "r")
        self.data_group = self.root["data"]

        self.obs_hist = self.data_group["obs_hist"]
        self.gt_action = self.data_group["gt_action"]
        self.cand_action = self.data_group["cand_action"]
        self.cand_rel_pos = self.data_group["cand_rel_pos"]
        self.cand_ade_m = self.data_group["cand_ade_m"]
        self.cand_fde_m = self.data_group["cand_fde_m"]
        self.cand_path_error_pct = self.data_group["cand_path_error_pct"]
        self.best_idx = self.data_group["best_idx"]

        self._length = int(self.obs_hist.shape[0])
        self.num_candidates = int(self.cand_action.shape[1])
        self.n_obs_steps = int(self.obs_hist.shape[1])
        self.obs_dim = int(self.obs_hist.shape[2])
        self.n_action_steps = int(self.cand_action.shape[2])
        self.action_dim = int(self.cand_action.shape[3])

    def get_normalizer(self, mode: str = "gaussian", **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={
                "obs_hist": self.obs_hist,
                "cand_action": self.cand_action,
                "cand_rel_pos": self.cand_rel_pos,
            },
            last_n_dims=2,
            mode=mode,
            **kwargs,
        )
        return normalizer

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "obs_hist": torch.from_numpy(self.obs_hist[idx]),
            "gt_action": torch.from_numpy(self.gt_action[idx]),
            "cand_action": torch.from_numpy(self.cand_action[idx]),
            "cand_rel_pos": torch.from_numpy(self.cand_rel_pos[idx]),
            "cand_ade_m": torch.from_numpy(self.cand_ade_m[idx]),
            "cand_fde_m": torch.from_numpy(self.cand_fde_m[idx]),
            "cand_path_error_pct": torch.from_numpy(self.cand_path_error_pct[idx]),
            "best_idx": torch.tensor(int(self.best_idx[idx]), dtype=torch.long),
        }
