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

class UAVCombatLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='uav_observations', # 形状预期: (N, 6, 7)
            action_key='uav_actions',    # 形状预期: (N, 3) 敌机相对位移
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        print(f"DEBUG: 收到 zarr_path 为 -> '{zarr_path}'")
        super().__init__()
        self.zarr_path = os.path.expanduser(zarr_path)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, action_key])
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
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

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
        data = self._sample_to_data(self.replay_buffer)
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
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        obs_raw = sample[self.obs_key]
        obs = obs_raw.reshape(obs_raw.shape[0], -1)

        action_raw = sample[self.action_key]
        action = action_raw.reshape(action_raw.shape[0], -1)

        data = {
            'obs': obs,        # 形状: (T, 42)
            'action': action,  # 形状: (T, 3)
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
