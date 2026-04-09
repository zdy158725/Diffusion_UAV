from typing import Dict
import copy
import os
import numpy as np
import torch
import zarr

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset


class ADSBAbsoluteLowdimDataset(BaseLowdimDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_key="uav_observations",
        action_key="uav_actions",
        obs_slice=(0, 9),
        n_obs_steps=11,
        action_target_type="relative_future_position",
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        super().__init__()
        self.zarr_path = os.path.expanduser(zarr_path)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=[obs_key, action_key],
        )
        zroot = zarr.open(self.zarr_path, "r")
        self.meters_per_unit = float(zroot.attrs.get("meters_per_unit", 1.0))
        self.position_slice = slice(0, 3)
        self.action_target = str(zroot.attrs.get("action_target", "absolute_position"))
        if self.action_target != "absolute_position":
            raise ValueError(
                f"ADSBAbsoluteLowdimDataset expects action_target=absolute_position, got {self.action_target}"
            )
        self.action_target_type = str(action_target_type)
        if self.action_target_type != "relative_future_position":
            raise ValueError(
                "ADSBAbsoluteLowdimDataset currently expects "
                f"action_target_type=relative_future_position, got {self.action_target_type}"
            )
        self.n_obs_steps = int(n_obs_steps)
        if self.n_obs_steps <= 0:
            raise ValueError(f"n_obs_steps must be positive, got {self.n_obs_steps}")

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=[obs_key, action_key],
            episode_mask=train_mask,
        )

        obs_slice = tuple(obs_slice)
        if len(obs_slice) != 2:
            raise ValueError(f"obs_slice should have length 2, got {obs_slice}")
        self.obs_slice = slice(int(obs_slice[0]), int(obs_slice[1]))
        self.obs_key = obs_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self._relative_action_cache = None

    def get_collate_fn(self):
        return None

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=[self.obs_key, self.action_key],
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        val_set._relative_action_cache = None
        return val_set

    def _sample_to_obs(self, sample):
        obs_raw = sample[self.obs_key]
        return obs_raw[..., self.obs_slice].astype(np.float32, copy=False)

    def _get_current_pos_from_obs(self, obs: np.ndarray) -> np.ndarray:
        if obs.shape[0] < self.n_obs_steps:
            raise ValueError(
                f"Sample length {obs.shape[0]} shorter than n_obs_steps={self.n_obs_steps}"
            )
        current_idx = self.n_obs_steps - 1
        return obs[current_idx, self.position_slice].astype(np.float32, copy=False)

    def _sample_to_action(self, sample, obs=None):
        if obs is None:
            obs = self._sample_to_obs(sample)
        abs_action = sample[self.action_key].astype(np.float32, copy=False)
        current_pos = self._get_current_pos_from_obs(obs)
        rel_action = abs_action - current_pos[None, :]
        return rel_action.astype(np.float32, copy=False)

    def _collect_relative_action_sequences(self) -> np.ndarray:
        if self._relative_action_cache is None:
            rel_actions = []
            for idx in range(len(self.sampler)):
                sample = self.sampler.sample_sequence(idx)
                obs = self._sample_to_obs(sample)
                rel_actions.append(self._sample_to_action(sample, obs=obs))
            if rel_actions:
                self._relative_action_cache = np.stack(rel_actions, axis=0).astype(np.float32, copy=False)
            else:
                self._relative_action_cache = np.empty((0, self.horizon, 3), dtype=np.float32)
        return self._relative_action_cache

    def get_normalizer(self, mode="limits", **kwargs):
        obs = self._sample_to_obs(self.replay_buffer)
        action = self._collect_relative_action_sequences()
        normalizer = LinearNormalizer()
        normalizer.fit(data={"obs": obs}, last_n_dims=1, mode="limits", **kwargs)
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            action,
            last_n_dims=1,
            mode="gaussian",
        )
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self._collect_relative_action_sequences())

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        obs = self._sample_to_obs(sample)
        data = {
            "obs": obs,
            "action": self._sample_to_action(sample, obs=obs),
        }
        return dict_apply(data, torch.from_numpy)
