from typing import Dict
import copy
import os
import numpy as np
import torch
import zarr
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from torch.utils.data._utils.collate import default_collate


class UAVCombatBatchCollator:
    enemy0_pos_slice = slice(21, 24)

    @classmethod
    def compute_action_from_obs(cls, obs: torch.Tensor) -> torch.Tensor:
        enemy0_pos = obs[..., cls.enemy0_pos_slice]
        action = torch.zeros_like(enemy0_pos)
        action[..., 1:, :] = enemy0_pos[..., 1:, :] - enemy0_pos[..., :-1, :]
        return action

    def __call__(self, batch):
        collated = default_collate(batch)
        collated["action"] = self.compute_action_from_obs(collated["obs"])
        return collated

class UAVCombatLowdimDataset(BaseLowdimDataset):
    enemy0_pos_slice = slice(21, 24)

    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='uav_observations', # 形状预期: (N, 6, 7)
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        print(f"DEBUG: 收到 zarr_path 为 -> '{zarr_path}'")
        super().__init__()
        self.zarr_path = os.path.expanduser(zarr_path)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key])
        zroot = zarr.open(self.zarr_path, "r")
        self.delta_max_abs = zroot.attrs.get("delta_max_abs", None)
        self.meters_per_unit = float(zroot.attrs.get("meters_per_unit", 1.0))

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
            )
        
        self.obs_key = obs_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    @classmethod
    def _compute_action_from_obs_np(cls, obs: np.ndarray, episode_ends=None) -> np.ndarray:
        enemy0_pos = obs[..., cls.enemy0_pos_slice]
        action = np.zeros_like(enemy0_pos, dtype=np.float32)
        if episode_ends is None:
            action[..., 1:, :] = enemy0_pos[..., 1:, :] - enemy0_pos[..., :-1, :]
            return action

        start = 0
        for end in np.asarray(episode_ends, dtype=np.int64).tolist():
            if end - start > 1:
                action[start + 1:end, :] = enemy0_pos[start + 1:end, :] - enemy0_pos[start:end - 1, :]
            start = end
        return action

    def get_collate_fn(self):
        return UAVCombatBatchCollator()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        obs = self._sample_to_obs(self.replay_buffer)
        action = self._compute_action_from_obs_np(
            obs,
            episode_ends=self.replay_buffer.episode_ends[:]
        )
        data = {
            "obs": obs,
            "action": action,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        if self.delta_max_abs is not None:
            max_abs = np.asarray(self.delta_max_abs, dtype=np.float32).reshape(-1)
            max_abs = np.maximum(max_abs, 1e-6)
            action = data["action"].astype(np.float32)
            normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
                scale=(1.0 / max_abs).astype(np.float32),
                offset=np.zeros_like(max_abs, dtype=np.float32),
                input_stats_dict={
                    "min": (-max_abs).astype(np.float32),
                    "max": max_abs.astype(np.float32),
                    "mean": np.mean(action, axis=0).astype(np.float32),
                    "std": np.std(action, axis=0).astype(np.float32),
                },
            )
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        obs = self._sample_to_obs(self.replay_buffer)
        action = self._compute_action_from_obs_np(
            obs,
            episode_ends=self.replay_buffer.episode_ends[:]
        )
        return torch.from_numpy(action)

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_obs(self, sample):
        obs_raw = sample[self.obs_key]
        obs = obs_raw.reshape(obs_raw.shape[0], -1)
        return obs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = {
            'obs': self._sample_to_obs(sample),  # 形状: (T, 42)
        }

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
