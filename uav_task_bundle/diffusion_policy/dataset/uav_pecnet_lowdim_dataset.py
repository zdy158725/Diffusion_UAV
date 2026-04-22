from typing import Dict, Optional, Sequence
import copy
import os
import pathlib
import sys

import numpy as np
import torch

from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PECNET_UTILS_DIR = PROJECT_ROOT / "baseline" / "PECNet" / "PECNet-master" / "utils"
if str(PECNET_UTILS_DIR) not in sys.path:
    sys.path.append(str(PECNET_UTILS_DIR))

from uav_data import (  # noqa: E402
    DRONE_ORDER,
    TEAM_TO_ONEHOT,
    build_sample_indices,
    load_episode_split,
    load_uav_episodes,
)

ROLE_SELF = 0
ROLE_ALLY = 1
ROLE_ENEMY = 2
ROLE_EMPTY = 3


class UAVPECNetLowdimDataset(BaseLowdimDataset):
    """
    Adapt the PECNet UAV-3D CSV dataset into diffusion_policy's lowdim format.

    The observation keeps the PECNet information content:
    - per-agent past features over the observed history
    - current context features (broadcast across obs steps)
    - team one-hot
    - valid-agent mask
    - social mask

    To remain compatible with the existing lowdim offline runner, the first
    three observation dims store the target absolute xyz at each observed step.
    """

    def __init__(
        self,
        csv_path,
        split_stats_path=None,
        past_length=5,
        future_length=6,
        train_ratio=0.8,
        split_seed=123,
        max_train_samples=None,
        max_val_samples=None,
        stats_sample_limit=50000,
        include_target_abs_pos=True,
        include_context=True,
        include_team_onehot=True,
        include_valid_mask=True,
        include_social_mask=True,
        include_structured_fields=False,
        action_target_type="delta_position",
        obs_normalizer_mode="gaussian",
        action_normalizer_mode="gaussian",
    ):
        super().__init__()

        self.csv_path = os.path.expanduser(str(csv_path))
        self.split_stats_path = (
            None if split_stats_path is None else os.path.expanduser(str(split_stats_path))
        )
        self.past_length = int(past_length)
        self.future_length = int(future_length)
        self.horizon = self.past_length + self.future_length
        self.max_agents = 6
        self.train_ratio = float(train_ratio)
        self.split_seed = int(split_seed)
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.stats_sample_limit = (
            None if stats_sample_limit is None else int(stats_sample_limit)
        )

        self.include_target_abs_pos = bool(include_target_abs_pos)
        self.include_context = bool(include_context)
        self.include_team_onehot = bool(include_team_onehot)
        self.include_valid_mask = bool(include_valid_mask)
        self.include_social_mask = bool(include_social_mask)
        self.include_structured_fields = bool(include_structured_fields)
        self.action_target_type = str(action_target_type)
        if self.action_target_type not in {"delta_position", "relative_future_position"}:
            raise ValueError(
                "Unsupported action_target_type="
                f"{self.action_target_type}. Expected delta_position or "
                "relative_future_position."
            )
        self.obs_normalizer_mode = str(obs_normalizer_mode)
        self.action_normalizer_mode = str(action_normalizer_mode)

        self.meters_per_unit = 1.0
        if self.include_target_abs_pos:
            self.position_slice = slice(0, 3)
        else:
            self.position_slice = slice(0, 0)

        self.episodes = load_uav_episodes(pathlib.Path(self.csv_path))
        all_episode_ids = [episode.episode_id for episode in self.episodes]
        train_episode_ids, val_episode_ids = load_episode_split(
            episode_ids=all_episode_ids,
            split_stats_path=(
                None if self.split_stats_path is None else pathlib.Path(self.split_stats_path)
            ),
            train_ratio=self.train_ratio,
            split_seed=self.split_seed,
        )

        self.train_episode_ids = list(train_episode_ids)
        self.val_episode_ids = list(val_episode_ids)
        self.train_sample_indices = build_sample_indices(
            episodes=self.episodes,
            selected_episode_ids=self.train_episode_ids,
            past_length=self.past_length,
            future_length=self.future_length,
            sample_limit=self.max_train_samples,
        )
        self.val_sample_indices = build_sample_indices(
            episodes=self.episodes,
            selected_episode_ids=self.val_episode_ids,
            past_length=self.past_length,
            future_length=self.future_length,
            sample_limit=self.max_val_samples,
        )

        self.sample_indices = self.train_sample_indices
        self.split = "train"
        self.obs_dim = self._infer_obs_dim()
        self.action_dim = 3
        self._normalizer_cache: Optional[LinearNormalizer] = None
        self._all_actions_cache: Optional[np.ndarray] = None

    def _infer_obs_dim(self) -> int:
        obs_dim = 0
        if self.include_target_abs_pos:
            obs_dim += 3
        obs_dim += self.max_agents * 7
        if self.include_context:
            obs_dim += self.max_agents * 7
        if self.include_team_onehot:
            obs_dim += self.max_agents * 2
        if self.include_valid_mask:
            obs_dim += self.max_agents
        if self.include_social_mask:
            obs_dim += self.max_agents * self.max_agents
        return int(obs_dim)

    def get_action_window_indices(self):
        start = self.past_length
        end = start + self.future_length
        return start, end

    def get_collate_fn(self):
        return None

    def _set_split(self, split: str):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split={split}")
        self.split = split
        self.sample_indices = (
            self.train_sample_indices if split == "train" else self.val_sample_indices
        )
        self._normalizer_cache = None
        self._all_actions_cache = None

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set._set_split("val")
        return val_set

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    def _get_active_track(self, episode, slot: int, anchor_t: int):
        track = episode.tracks[slot]
        if track is None:
            return None
        if anchor_t >= track.length or anchor_t < (self.past_length - 1):
            return None
        return track

    def _build_structured_slot_order(self, episode, target_slot: int, anchor_t: int):
        target_track = episode.tracks[target_slot]
        if target_track is None:
            raise RuntimeError(f"Target track missing for slot={target_slot}")

        same_team_slots = []
        other_team_slots = []
        for slot, _ in enumerate(DRONE_ORDER):
            if slot == target_slot:
                continue
            track = self._get_active_track(episode, slot=slot, anchor_t=anchor_t)
            if track is None:
                continue
            if track.team == target_track.team:
                same_team_slots.append(slot)
            else:
                other_team_slots.append(slot)
        return [target_slot] + same_team_slots + other_team_slots

    def _build_structured_agent_tensors(
        self,
        episode,
        target_slot: int,
        anchor_t: int,
        anchor_abs: np.ndarray,
        start: int,
        stop: int,
    ):
        target_track = episode.tracks[target_slot]
        if target_track is None:
            raise RuntimeError(f"Target track missing for slot={target_slot}")

        agent_obs = np.zeros((self.past_length, self.max_agents, 7), dtype=np.float32)
        agent_team = np.zeros((self.max_agents, 2), dtype=np.float32)
        agent_valid = np.zeros((self.max_agents,), dtype=np.float32)
        agent_role_id = np.full((self.max_agents,), ROLE_EMPTY, dtype=np.int64)

        canonical_slots = self._build_structured_slot_order(
            episode=episode, target_slot=target_slot, anchor_t=anchor_t
        )
        for canonical_slot, source_slot in enumerate(canonical_slots[: self.max_agents]):
            track = self._get_active_track(episode, slot=source_slot, anchor_t=anchor_t)
            if track is None:
                continue

            rel_position = track.position[start:stop] - anchor_abs[None, :]
            velocity = track.velocity[start:stop]
            hp = track.hp[start:stop, None]
            agent_obs[:, canonical_slot] = np.concatenate(
                [rel_position, velocity, hp], axis=-1
            ).astype(np.float32, copy=False)
            agent_team[canonical_slot] = TEAM_TO_ONEHOT[track.team]
            agent_valid[canonical_slot] = 1.0

            if canonical_slot == 0:
                agent_role_id[canonical_slot] = ROLE_SELF
            elif track.team == target_track.team:
                agent_role_id[canonical_slot] = ROLE_ALLY
            else:
                agent_role_id[canonical_slot] = ROLE_ENEMY

        agent_social_mask = np.outer(agent_valid, agent_valid).astype(np.float32, copy=False)
        return {
            "agent_obs": agent_obs,
            "agent_team": agent_team,
            "agent_valid": agent_valid,
            "agent_social_mask": agent_social_mask,
            "agent_role_id": agent_role_id,
        }

    def _build_agent_tensors(self, sample_index: int):
        episode_index, target_slot, anchor_t = self.sample_indices[sample_index].tolist()
        episode = self.episodes[episode_index]
        target_track = episode.tracks[target_slot]
        if target_track is None:
            raise RuntimeError(f"Target track missing for sample_index={sample_index}")

        anchor_t = int(anchor_t)
        start = anchor_t - self.past_length + 1
        stop = anchor_t + 1
        anchor_abs = target_track.position[anchor_t].astype(np.float32, copy=False)
        target_abs_hist = target_track.position[start:stop].astype(np.float32, copy=False)

        past = np.zeros((self.max_agents, self.past_length, 7), dtype=np.float32)
        context = np.zeros((self.max_agents, 7), dtype=np.float32)
        team = np.zeros((self.max_agents, 2), dtype=np.float32)
        valid = np.zeros((self.max_agents,), dtype=np.float32)

        for slot, track in enumerate(episode.tracks):
            if track is None:
                continue
            if anchor_t >= track.length or anchor_t < (self.past_length - 1):
                continue

            rel_position = track.position[start:stop] - anchor_abs[None, :]
            velocity = track.velocity[start:stop]
            hp = track.hp[start:stop, None]
            past[slot] = np.concatenate([rel_position, velocity, hp], axis=-1).astype(
                np.float32,
                copy=False,
            )
            context[slot] = np.concatenate(
                [
                    track.position[anchor_t] - anchor_abs,
                    track.velocity[anchor_t],
                    np.asarray([track.hp[anchor_t]], dtype=np.float32),
                ],
                axis=0,
            ).astype(np.float32, copy=False)
            team[slot] = TEAM_TO_ONEHOT[track.team]
            valid[slot] = 1.0

        social_mask = np.outer(valid, valid).astype(np.float32, copy=False)

        future_abs = target_track.position[
            anchor_t + 1 : anchor_t + self.future_length + 1
        ].astype(np.float32, copy=False)
        future_relpos = (future_abs - anchor_abs[None, :]).astype(np.float32, copy=False)
        endpoint_target = future_relpos[-1].astype(np.float32, copy=False)
        interp_scale = (
            np.arange(1, self.future_length + 1, dtype=np.float32) / float(self.future_length)
        )[:, None]
        reference_path_gt = (interp_scale * endpoint_target[None, :]).astype(
            np.float32,
            copy=False,
        )
        residual_target_gt = (future_relpos - reference_path_gt).astype(
            np.float32,
            copy=False,
        )
        action = np.zeros((self.horizon, 3), dtype=np.float32)
        if self.action_target_type == "relative_future_position":
            start_idx, end_idx = self.get_action_window_indices()
            action[start_idx:end_idx] = future_relpos
        else:
            full_pos = np.concatenate([target_abs_hist, future_abs], axis=0).astype(
                np.float32,
                copy=False,
            )
            action[1:] = full_pos[1:] - full_pos[:-1]

        tensors = {
            "target_abs_hist": target_abs_hist,
            "past": past,
            "context": context,
            "team": team,
            "valid": valid,
            "social_mask": social_mask,
            "action": action,
            "endpoint_target": endpoint_target,
            "reference_path_gt": reference_path_gt,
            "residual_target_gt": residual_target_gt,
        }
        if self.include_structured_fields:
            tensors.update(
                self._build_structured_agent_tensors(
                    episode=episode,
                    target_slot=target_slot,
                    anchor_t=anchor_t,
                    anchor_abs=anchor_abs,
                    start=start,
                    stop=stop,
                )
            )
        return tensors

    def _assemble_obs(self, tensors: Dict[str, np.ndarray]) -> np.ndarray:
        target_abs_hist = tensors["target_abs_hist"]
        past = tensors["past"]
        context = tensors["context"]
        team = tensors["team"]
        valid = tensors["valid"]
        social_mask = tensors["social_mask"]

        past_flat = past.transpose(1, 0, 2).reshape(self.past_length, -1)
        context_flat = context.reshape(-1)
        team_flat = team.reshape(-1)
        valid_flat = valid.reshape(-1)
        social_flat = social_mask.reshape(-1)

        obs_steps = []
        for t in range(self.past_length):
            parts = []
            if self.include_target_abs_pos:
                parts.append(target_abs_hist[t])
            parts.append(past_flat[t])
            if self.include_context:
                parts.append(context_flat)
            if self.include_team_onehot:
                parts.append(team_flat)
            if self.include_valid_mask:
                parts.append(valid_flat)
            if self.include_social_mask:
                parts.append(social_flat)
            obs_steps.append(np.concatenate(parts, axis=0).astype(np.float32, copy=False))
        obs = np.stack(obs_steps, axis=0).astype(np.float32, copy=False)
        if obs.shape[1] != self.obs_dim:
            raise RuntimeError(
                f"Obs dim mismatch: expected {self.obs_dim}, got {obs.shape[1]}"
            )
        return obs

    def _build_item_arrays(self, sample_index: int):
        tensors = self._build_agent_tensors(sample_index)
        obs = self._assemble_obs(tensors)
        action = tensors["action"]
        return obs, action

    def _collect_obs_subset(self, sample_indices: Sequence[int]) -> np.ndarray:
        obs_list = []
        for sample_index in sample_indices:
            obs, _ = self._build_item_arrays(int(sample_index))
            obs_list.append(obs)
        if not obs_list:
            return np.empty((0, self.past_length, self.obs_dim), dtype=np.float32)
        return np.stack(obs_list, axis=0).astype(np.float32, copy=False)

    def _collect_action_subset(self, sample_indices: Sequence[int]) -> np.ndarray:
        action_list = []
        for sample_index in sample_indices:
            _, action = self._build_item_arrays(int(sample_index))
            action_list.append(action)
        if not action_list:
            return np.empty((0, self.horizon, self.action_dim), dtype=np.float32)
        return np.stack(action_list, axis=0).astype(np.float32, copy=False)

    def _collect_endpoint_subset(self, sample_indices: Sequence[int]) -> np.ndarray:
        endpoint_list = []
        for sample_index in sample_indices:
            tensors = self._build_agent_tensors(int(sample_index))
            endpoint_list.append(tensors["endpoint_target"])
        if not endpoint_list:
            return np.empty((0, self.action_dim), dtype=np.float32)
        return np.stack(endpoint_list, axis=0).astype(np.float32, copy=False)

    def _collect_residual_subset(self, sample_indices: Sequence[int]) -> np.ndarray:
        residual_list = []
        for sample_index in sample_indices:
            tensors = self._build_agent_tensors(int(sample_index))
            residual_list.append(tensors["residual_target_gt"])
        if not residual_list:
            return np.empty((0, self.future_length, self.action_dim), dtype=np.float32)
        return np.stack(residual_list, axis=0).astype(np.float32, copy=False)

    def _collect_agent_obs_subset(self, sample_indices: Sequence[int]) -> np.ndarray:
        agent_obs_list = []
        for sample_index in sample_indices:
            tensors = self._build_agent_tensors(int(sample_index))
            agent_obs_list.append(tensors["agent_obs"])
        if not agent_obs_list:
            return np.empty((0, self.past_length, self.max_agents, 7), dtype=np.float32)
        return np.stack(agent_obs_list, axis=0).astype(np.float32, copy=False)

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        if self._normalizer_cache is not None:
            return self._normalizer_cache

        sample_indices = np.arange(len(self), dtype=np.int64)
        if self.stats_sample_limit is not None:
            sample_indices = sample_indices[: self.stats_sample_limit]

        obs = self._collect_obs_subset(sample_indices)
        action = self._collect_action_subset(sample_indices)
        action_stats = action
        if self.action_target_type == "relative_future_position":
            start, end = self.get_action_window_indices()
            action_stats = action[:, start:end, :]

        normalizer = LinearNormalizer()
        normalizer["obs"] = SingleFieldLinearNormalizer.create_fit(
            obs,
            last_n_dims=1,
            mode=self.obs_normalizer_mode,
        )
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            action_stats,
            last_n_dims=1,
            mode=self.action_normalizer_mode,
        )
        if self.action_target_type == "relative_future_position":
            endpoint_target = self._collect_endpoint_subset(sample_indices)
            residual_action = self._collect_residual_subset(sample_indices)
            normalizer["endpoint_target"] = SingleFieldLinearNormalizer.create_fit(
                endpoint_target,
                last_n_dims=1,
                mode=self.action_normalizer_mode,
            )
            normalizer["residual_action"] = SingleFieldLinearNormalizer.create_fit(
                residual_action,
                last_n_dims=1,
                mode=self.action_normalizer_mode,
            )
        if self.include_structured_fields:
            agent_obs = self._collect_agent_obs_subset(sample_indices)
            normalizer["agent_obs"] = SingleFieldLinearNormalizer.create_fit(
                agent_obs,
                last_n_dims=1,
                mode=self.obs_normalizer_mode,
            )
        self._normalizer_cache = normalizer
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        if self._all_actions_cache is None:
            sample_indices = np.arange(len(self), dtype=np.int64)
            self._all_actions_cache = self._collect_action_subset(sample_indices)
        return torch.from_numpy(self._all_actions_cache)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tensors = self._build_agent_tensors(int(idx))
        obs = self._assemble_obs(tensors)
        action = tensors["action"]
        item = {
            "obs": torch.from_numpy(obs),
            "action": torch.from_numpy(action),
        }
        if self.action_target_type == "relative_future_position":
            item.update(
                {
                    "endpoint_target": torch.from_numpy(tensors["endpoint_target"]),
                    "reference_path_gt": torch.from_numpy(tensors["reference_path_gt"]),
                    "residual_target_gt": torch.from_numpy(tensors["residual_target_gt"]),
                }
            )
        if self.include_structured_fields:
            item.update(
                {
                    "agent_obs": torch.from_numpy(tensors["agent_obs"]),
                    "agent_team": torch.from_numpy(tensors["agent_team"]),
                    "agent_valid": torch.from_numpy(tensors["agent_valid"]),
                    "agent_social_mask": torch.from_numpy(tensors["agent_social_mask"]),
                    "agent_role_id": torch.from_numpy(tensors["agent_role_id"]),
                }
            )
        return item
