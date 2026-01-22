from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

class UAVCombatLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='data/uav_observations',
            action_key='data/uav_actions',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        # 1. 使用更底层的方式打开 Zarr
        import zarr
        zarr_obj = zarr.open(zarr_path, mode='r')
        
        # 2. 手动创建 ReplayBuffer
        # 我们直接把整个 zarr 组传进去，它会自动寻找内部的数组
        self.replay_buffer = ReplayBuffer(zarr_obj)
        
        # 3. 验证关键的 episode_ends 是否存在
        # 根据你之前的 ls，它在 meta/episode_ends
        if 'meta/episode_ends' in zarr_obj:
            # 告诉 buffer 哪里是回合结束
            self.replay_buffer.episode_ends = zarr_obj['meta/episode_ends'][:]
        elif 'episode_ends' in zarr_obj:
            self.replay_buffer.episode_ends = zarr_obj['episode_ends'][:]
        else:
            raise KeyError("在 Zarr 中找不到 episode_ends，请检查路径！")

        # 4. 设置采样掩码
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
            'action': action,  # 形状: (T, 9)
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data