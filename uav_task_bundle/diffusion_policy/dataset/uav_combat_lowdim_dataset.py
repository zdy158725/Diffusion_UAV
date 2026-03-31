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
    def __init__(self, position_slice=slice(0, 3)):
        self.position_slice = position_slice

    def compute_action_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        pos = obs[..., self.position_slice]
        action = torch.zeros_like(pos)
        action[..., 1:, :] = pos[..., 1:, :] - pos[..., :-1, :]
        return action

    def __call__(self, batch):
        collated = default_collate(batch)
        collated["action"] = self.compute_action_from_obs(collated["obs"])
        return collated


class UAVCombatLowdimDataset(BaseLowdimDataset):
    OBS_MODE_SLICE = 'slice'
    OBS_MODE_ENEMY0_OWN_REL24 = 'enemy0_abs_plus_own_rel24'
    OBS_MODE_ENEMY0_OWN_REL24_GEO39 = 'enemy0_abs_plus_own_rel24_geo39'

    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='uav_observations',
            obs_slice=(21, 27),
            obs_mode='slice',
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
        self.delta_max_abs_raw = zroot.attrs.get("delta_max_abs", None)
        self.delta_max_abs = zroot.attrs.get("delta_scale_abs", self.delta_max_abs_raw)
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

        self.obs_mode = str(obs_mode)
        valid_modes = {
            self.OBS_MODE_SLICE,
            self.OBS_MODE_ENEMY0_OWN_REL24,
            self.OBS_MODE_ENEMY0_OWN_REL24_GEO39,
        }
        if self.obs_mode not in valid_modes:
            raise ValueError(f"Unsupported obs_mode={self.obs_mode}. Expected one of {sorted(valid_modes)}")

        obs_slice = tuple(obs_slice)
        if len(obs_slice) != 2:
            raise ValueError(f"obs_slice should have length 2, got {obs_slice}")
        self.obs_slice = slice(int(obs_slice[0]), int(obs_slice[1]))
        # Output obs must keep enemy0 absolute xyz in the first 3 dims so that
        # action differencing and trajectory anchoring stay compatible.
        self.position_slice = slice(0, 3)
        self.obs_dim = self._infer_obs_dim()

        self.obs_key = obs_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def _infer_obs_dim(self) -> int:
        if self.obs_mode == self.OBS_MODE_SLICE:
            return int(self.obs_slice.stop - self.obs_slice.start)
        if self.obs_mode == self.OBS_MODE_ENEMY0_OWN_REL24:
            return 24
        if self.obs_mode == self.OBS_MODE_ENEMY0_OWN_REL24_GEO39:
            return 39
        raise RuntimeError(f"Unexpected obs_mode={self.obs_mode}")

    def _build_enemy0_own_rel24_parts(self, obs: np.ndarray):
        if obs.shape[-1] < 42:
            raise ValueError(
                "enemy0_abs_plus_own_rel24 expects raw obs dim >= 42, "
                f"got {obs.shape[-1]}"
            )

        enemy0_abs = obs[..., 21:27]
        enemy0_pos = enemy0_abs[..., 0:3]
        enemy0_vel = enemy0_abs[..., 3:6]

        own_rel_parts = []
        for start in (0, 7, 14):
            own_abs = obs[..., start:start + 6]
            own_pos = own_abs[..., 0:3]
            own_vel = own_abs[..., 3:6]
            own_rel = np.concatenate([
                own_pos - enemy0_pos,
                own_vel - enemy0_vel,
            ], axis=-1)
            own_rel_parts.append(own_rel.astype(np.float32, copy=False))
        return enemy0_abs.astype(np.float32, copy=False), own_rel_parts

    def _build_xy_geometry_features(self, own_rel_parts) -> np.ndarray:
        eps = np.float32(1e-6)
        geo_parts = []
        for own_rel in own_rel_parts:
            dx = own_rel[..., 0]
            dy = own_rel[..., 1]
            dvx = own_rel[..., 3]
            dvy = own_rel[..., 4]

            r_xy = np.sqrt(dx * dx + dy * dy).astype(np.float32, copy=False)
            valid = r_xy >= eps
            denom = np.where(valid, r_xy, np.float32(1.0)).astype(np.float32, copy=False)

            sin_theta = np.where(valid, dy / denom, np.float32(0.0)).astype(np.float32, copy=False)
            cos_theta = np.where(valid, dx / denom, np.float32(0.0)).astype(np.float32, copy=False)
            v_radial = np.where(
                valid,
                (dx * dvx + dy * dvy) / denom,
                np.float32(0.0),
            ).astype(np.float32, copy=False)
            v_tangential = np.where(
                valid,
                (dx * dvy - dy * dvx) / denom,
                np.float32(0.0),
            ).astype(np.float32, copy=False)

            geo_parts.append(np.stack(
                [r_xy, sin_theta, cos_theta, v_radial, v_tangential],
                axis=-1
            ).astype(np.float32, copy=False))
        return np.concatenate(geo_parts, axis=-1).astype(np.float32, copy=False)

    def _build_obs_features(self, obs_raw: np.ndarray) -> np.ndarray:
        obs = obs_raw.reshape(obs_raw.shape[0], -1).astype(np.float32, copy=False)
        if self.obs_mode == self.OBS_MODE_SLICE:
            return obs[..., self.obs_slice]

        if self.obs_mode not in {
            self.OBS_MODE_ENEMY0_OWN_REL24,
            self.OBS_MODE_ENEMY0_OWN_REL24_GEO39,
        }:
            raise RuntimeError(f"Unexpected obs_mode={self.obs_mode}")

        enemy0_abs, own_rel_parts = self._build_enemy0_own_rel24_parts(obs)
        base_features = np.concatenate([enemy0_abs] + own_rel_parts, axis=-1).astype(np.float32, copy=False)
        if self.obs_mode == self.OBS_MODE_ENEMY0_OWN_REL24:
            return base_features

        geometry_features = self._build_xy_geometry_features(own_rel_parts)
        return np.concatenate([base_features, geometry_features], axis=-1).astype(np.float32, copy=False)

    def _compute_action_from_obs_np(self, obs: np.ndarray, episode_ends=None) -> np.ndarray:
        pos = obs[..., self.position_slice]
        action = np.zeros_like(pos, dtype=np.float32)
        if episode_ends is None:
            action[..., 1:, :] = pos[..., 1:, :] - pos[..., :-1, :]
            return action

        start = 0
        for end in np.asarray(episode_ends, dtype=np.int64).tolist():
            if end - start > 1:
                action[start + 1:end, :] = pos[start + 1:end, :] - pos[start:end - 1, :]
            start = end
        return action

    def get_collate_fn(self):
        return UAVCombatBatchCollator(position_slice=self.position_slice)

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
        return self._build_obs_features(obs_raw)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = {
            'obs': self._sample_to_obs(sample),
        }

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
