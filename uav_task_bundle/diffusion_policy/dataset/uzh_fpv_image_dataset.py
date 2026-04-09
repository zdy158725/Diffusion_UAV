from typing import Dict
import copy
import os

import numpy as np
import torch
import zarr

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def get_image_range_normalizer():
    scale = np.array([2], dtype=np.float32)
    offset = np.array([-1], dtype=np.float32)
    stat = {
        "min": np.array([0], dtype=np.float32),
        "max": np.array([1], dtype=np.float32),
        "mean": np.array([0.5], dtype=np.float32),
        "std": np.array([np.sqrt(1.0 / 12.0)], dtype=np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat,
    )


class UZHFPVImageDataset(BaseImageDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        super().__init__()
        self.zarr_path = os.path.expanduser(zarr_path)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=["img", "state", "action"],
        )
        zroot = zarr.open(self.zarr_path, "r")
        self.delta_max_abs_raw = zroot.attrs.get("delta_max_abs", None)
        self.delta_max_abs = zroot.attrs.get("delta_scale_abs", self.delta_max_abs_raw)
        raw_position_max_abs = zroot.attrs.get("raw_position_max_abs", None)
        self.raw_position_max_abs = (
            None if raw_position_max_abs is None
            else np.maximum(np.asarray(raw_position_max_abs, dtype=np.float32).reshape(-1), 1e-6)
        )
        self.meters_per_unit = float(zroot.attrs.get("meters_per_unit", 1.0))
        self.position_slice = slice(0, 3)

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
            keys=["img", "state", "action"],
            episode_mask=train_mask,
        )
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
            keys=["img", "state", "action"],
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, state_mode="limits", **kwargs):
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"state": self.replay_buffer["state"]},
            last_n_dims=1,
            mode=state_mode,
            **kwargs,
        )
        normalizer["image"] = get_image_range_normalizer()

        if self.delta_max_abs is not None:
            max_abs = np.asarray(self.delta_max_abs, dtype=np.float32).reshape(-1)
            max_abs = np.maximum(max_abs, 1e-6)
            action = self.replay_buffer["action"][:].astype(np.float32)
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
        else:
            normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer["action"][:].astype(np.float32),
                last_n_dims=1,
                mode="limits",
                fit_offset=False,
            )
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"][:].astype(np.float32))

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        image = np.moveaxis(sample["img"], -1, 1).astype(np.float32) / 255.0
        state = sample["state"].astype(np.float32)
        action = sample["action"].astype(np.float32)
        return {
            "obs": {
                "image": image,
                "state": state,
            },
            "action": action,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)
