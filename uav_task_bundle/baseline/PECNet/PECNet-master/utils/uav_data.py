import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


DRONE_ORDER = (
    "blue_0",
    "blue_1",
    "blue_2",
    "red_0",
    "red_1",
    "red_2",
)
DRONE_TO_SLOT = {drone_id: idx for idx, drone_id in enumerate(DRONE_ORDER)}
TEAM_TO_ONEHOT = {
    "BLUE": np.asarray([1.0, 0.0], dtype=np.float32),
    "RED": np.asarray([0.0, 1.0], dtype=np.float32),
}


@dataclass
class TrackData:
    drone_id: str
    team: str
    position: np.ndarray
    velocity: np.ndarray
    hp: np.ndarray

    @property
    def length(self) -> int:
        return int(self.position.shape[0])


@dataclass
class EpisodeData:
    episode_id: str
    tracks: List[Optional[TrackData]]


@dataclass
class UAVFeatureNormalizer:
    past_mean: np.ndarray
    past_std: np.ndarray
    context_mean: np.ndarray
    context_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray

    @staticmethod
    def _normalize(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return (array - mean) / std

    @staticmethod
    def _unnormalize(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return array * std + mean

    def normalize_past(self, array: np.ndarray) -> np.ndarray:
        return self._normalize(array, self.past_mean, self.past_std).astype(np.float32, copy=False)

    def normalize_context(self, array: np.ndarray) -> np.ndarray:
        return self._normalize(array, self.context_mean, self.context_std).astype(np.float32, copy=False)

    def normalize_output(self, array: np.ndarray) -> np.ndarray:
        return self._normalize(array, self.output_mean, self.output_std).astype(np.float32, copy=False)

    def unnormalize_output(self, array: np.ndarray) -> np.ndarray:
        return self._unnormalize(array, self.output_mean, self.output_std).astype(np.float32, copy=False)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "past_mean": self.past_mean.astype(np.float32).tolist(),
            "past_std": self.past_std.astype(np.float32).tolist(),
            "context_mean": self.context_mean.astype(np.float32).tolist(),
            "context_std": self.context_std.astype(np.float32).tolist(),
            "output_mean": self.output_mean.astype(np.float32).tolist(),
            "output_std": self.output_std.astype(np.float32).tolist(),
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Sequence[float]]) -> "UAVFeatureNormalizer":
        return cls(
            past_mean=np.asarray(state["past_mean"], dtype=np.float32),
            past_std=np.asarray(state["past_std"], dtype=np.float32),
            context_mean=np.asarray(state["context_mean"], dtype=np.float32),
            context_std=np.asarray(state["context_std"], dtype=np.float32),
            output_mean=np.asarray(state["output_mean"], dtype=np.float32),
            output_std=np.asarray(state["output_std"], dtype=np.float32),
        )


def _ensure_min_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.maximum(std, eps).astype(np.float32)


def load_uav_episodes(csv_path: Path) -> List[EpisodeData]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    raw_episodes: Dict[str, Dict[str, object]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episode_id = str(row["episode_id"])
            drone_id = str(row["drone_id"]).lower()
            team = str(row["team"]).upper()
            timestep = int(row["timestep"])

            if drone_id not in DRONE_TO_SLOT:
                raise ValueError(f"Unsupported drone_id={drone_id} in {csv_path}")

            episode_state = raw_episodes.setdefault(
                episode_id,
                {"tracks": {}},
            )
            track_map: Dict[str, Dict[str, object]] = episode_state["tracks"]  # type: ignore[assignment]
            track_state = track_map.setdefault(
                drone_id,
                {
                    "team": team,
                    "timesteps": [],
                    "position": [],
                    "velocity": [],
                    "hp": [],
                },
            )

            if str(track_state["team"]).upper() != team:
                raise ValueError(
                    f"Inconsistent team for {episode_id}/{drone_id}: "
                    f"{track_state['team']} vs {team}"
                )

            track_state["timesteps"].append(timestep)
            track_state["position"].append(
                [float(row["x"]), float(row["y"]), float(row["z"])]
            )
            track_state["velocity"].append(
                [float(row["vx"]), float(row["vy"]), float(row["vz"])]
            )
            track_state["hp"].append(float(row["hp"]))

    episodes: List[EpisodeData] = []
    for episode_id in sorted(raw_episodes.keys()):
        tracks: List[Optional[TrackData]] = [None] * len(DRONE_ORDER)
        track_map = raw_episodes[episode_id]["tracks"]  # type: ignore[index]
        for drone_id, track_state in track_map.items():
            timesteps = np.asarray(track_state["timesteps"], dtype=np.int64)
            if len(timesteps) == 0:
                continue
            expected = np.arange(len(timesteps), dtype=np.int64)
            if not np.array_equal(timesteps, expected):
                raise ValueError(
                    f"Track {episode_id}/{drone_id} is expected to start at 0 "
                    f"and be gap-free, got timesteps {timesteps[:5]}..."
                )

            tracks[DRONE_TO_SLOT[drone_id]] = TrackData(
                drone_id=drone_id,
                team=str(track_state["team"]).upper(),
                position=np.asarray(track_state["position"], dtype=np.float32),
                velocity=np.asarray(track_state["velocity"], dtype=np.float32),
                hp=np.asarray(track_state["hp"], dtype=np.float32),
            )

        episodes.append(EpisodeData(episode_id=episode_id, tracks=tracks))

    return episodes


def load_episode_split(
    episode_ids: Sequence[str],
    split_stats_path: Optional[Path] = None,
    train_ratio: float = 0.8,
    split_seed: int = 123,
) -> Tuple[List[str], List[str]]:
    episode_ids = list(episode_ids)
    if split_stats_path is not None:
        split_stats_path = Path(split_stats_path)
        if split_stats_path.exists():
            with split_stats_path.open("r") as f:
                split_data = json.load(f)
            train_ids = list(split_data.get("train_episode_ids", []))
            val_ids = list(split_data.get("val_episode_ids", []))
            missing = sorted(set(train_ids + val_ids) - set(episode_ids))
            if missing:
                raise ValueError(
                    "Split stats contains episode ids absent from CSV: "
                    f"{missing[:5]}"
                )
            return train_ids, val_ids

    rng = np.random.RandomState(split_seed)
    shuffled = list(episode_ids)
    rng.shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled, []

    train_count = int(round(len(shuffled) * train_ratio))
    train_count = max(1, min(len(shuffled) - 1, train_count))
    return shuffled[:train_count], shuffled[train_count:]


def build_sample_indices(
    episodes: Sequence[EpisodeData],
    selected_episode_ids: Sequence[str],
    past_length: int,
    future_length: int,
    sample_limit: Optional[int] = None,
) -> np.ndarray:
    episode_id_to_index = {episode.episode_id: idx for idx, episode in enumerate(episodes)}
    blocks: List[np.ndarray] = []
    total = 0

    for episode_id in selected_episode_ids:
        episode_index = episode_id_to_index[episode_id]
        episode = episodes[episode_index]
        for slot, track in enumerate(episode.tracks):
            if track is None:
                continue

            min_anchor = past_length - 1
            max_anchor = track.length - future_length - 1
            if max_anchor < min_anchor:
                continue

            anchors = np.arange(min_anchor, max_anchor + 1, dtype=np.int32)
            block = np.column_stack(
                [
                    np.full_like(anchors, episode_index),
                    np.full_like(anchors, slot),
                    anchors,
                ]
            )
            blocks.append(block)
            total += block.shape[0]

            if sample_limit is not None and total >= sample_limit:
                concatenated = np.concatenate(blocks, axis=0)
                return concatenated[:sample_limit]

    if not blocks:
        return np.zeros((0, 3), dtype=np.int32)

    concatenated = np.concatenate(blocks, axis=0)
    if sample_limit is not None:
        concatenated = concatenated[:sample_limit]
    return concatenated


def compute_normalizer(
    episodes: Sequence[EpisodeData],
    sample_indices: np.ndarray,
    past_length: int,
    future_length: int,
) -> UAVFeatureNormalizer:
    if sample_indices.shape[0] == 0:
        raise ValueError("Cannot compute normalizer with empty sample indices.")

    past_sum = np.zeros((7,), dtype=np.float64)
    past_sq_sum = np.zeros((7,), dtype=np.float64)
    context_sum = np.zeros((7,), dtype=np.float64)
    context_sq_sum = np.zeros((7,), dtype=np.float64)
    output_sum = np.zeros((3,), dtype=np.float64)
    output_sq_sum = np.zeros((3,), dtype=np.float64)

    past_count = 0
    context_count = 0
    output_count = 0

    for sample_idx, (episode_index, target_slot, anchor_t) in enumerate(sample_indices.tolist()):
        episode = episodes[episode_index]
        target_track = episode.tracks[target_slot]
        if target_track is None:
            raise RuntimeError("Target track unexpectedly missing while computing normalizer.")

        target_anchor_pos = target_track.position[anchor_t]

        for slot, track in enumerate(episode.tracks):
            if track is None:
                continue
            if anchor_t >= track.length:
                continue
            if anchor_t < (past_length - 1):
                continue

            start = anchor_t - past_length + 1
            stop = anchor_t + 1
            rel_position = track.position[start:stop] - target_anchor_pos[None, :]
            velocity = track.velocity[start:stop]
            hp = track.hp[start:stop, None]
            past_feature = np.concatenate([rel_position, velocity, hp], axis=-1)
            past_sum += np.sum(past_feature, axis=0)
            past_sq_sum += np.sum(np.square(past_feature), axis=0)
            past_count += past_feature.shape[0]

            current_context = np.concatenate(
                [
                    track.position[anchor_t] - target_anchor_pos,
                    track.velocity[anchor_t],
                    np.asarray([track.hp[anchor_t]], dtype=np.float32),
                ],
                axis=0,
            )
            context_sum += current_context
            context_sq_sum += np.square(current_context)
            context_count += 1

        future = target_track.position[anchor_t + 1 : anchor_t + future_length + 1] - target_anchor_pos[None, :]
        output_sum += np.sum(future, axis=0)
        output_sq_sum += np.sum(np.square(future), axis=0)
        output_count += future.shape[0]

        if (sample_idx + 1) % 200000 == 0:
            print(
                "Normalizer progress: {}/{} samples".format(
                    sample_idx + 1, sample_indices.shape[0]
                )
            )

    past_mean = past_sum / max(past_count, 1)
    context_mean = context_sum / max(context_count, 1)
    output_mean = output_sum / max(output_count, 1)

    past_var = past_sq_sum / max(past_count, 1) - np.square(past_mean)
    context_var = context_sq_sum / max(context_count, 1) - np.square(context_mean)
    output_var = output_sq_sum / max(output_count, 1) - np.square(output_mean)

    return UAVFeatureNormalizer(
        past_mean=past_mean.astype(np.float32),
        past_std=_ensure_min_std(np.sqrt(np.maximum(past_var, 0.0))),
        context_mean=context_mean.astype(np.float32),
        context_std=_ensure_min_std(np.sqrt(np.maximum(context_var, 0.0))),
        output_mean=output_mean.astype(np.float32),
        output_std=_ensure_min_std(np.sqrt(np.maximum(output_var, 0.0))),
    )


class UAVPECNetDataset(Dataset):
    def __init__(
        self,
        episodes: Sequence[EpisodeData],
        sample_indices: np.ndarray,
        past_length: int,
        future_length: int,
        normalizer: UAVFeatureNormalizer,
    ):
        self.episodes = list(episodes)
        self.sample_indices = np.asarray(sample_indices, dtype=np.int32)
        self.past_length = int(past_length)
        self.future_length = int(future_length)
        self.normalizer = normalizer

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode_index, target_slot, anchor_t = self.sample_indices[index].tolist()
        episode = self.episodes[episode_index]
        target_track = episode.tracks[target_slot]
        if target_track is None:
            raise RuntimeError("Target track unexpectedly missing.")

        target_anchor_pos = target_track.position[anchor_t]
        agent_slots = [target_slot]
        for slot, track in enumerate(episode.tracks):
            if slot == target_slot or track is None:
                continue
            if anchor_t < track.length and anchor_t >= (self.past_length - 1):
                agent_slots.append(slot)

        past = np.zeros((len(agent_slots), self.past_length, 7), dtype=np.float32)
        context_cont = np.zeros((len(agent_slots), 7), dtype=np.float32)
        team = np.zeros((len(agent_slots), 2), dtype=np.float32)

        for out_idx, slot in enumerate(agent_slots):
            track = episode.tracks[slot]
            if track is None:
                raise RuntimeError("Active agent slot unexpectedly missing track.")

            start = anchor_t - self.past_length + 1
            stop = anchor_t + 1
            rel_position = track.position[start:stop] - target_anchor_pos[None, :]
            velocity = track.velocity[start:stop]
            hp = track.hp[start:stop, None]
            past_feature = np.concatenate([rel_position, velocity, hp], axis=-1)
            past[out_idx] = self.normalizer.normalize_past(past_feature)

            current_context = np.concatenate(
                [
                    track.position[anchor_t] - target_anchor_pos,
                    track.velocity[anchor_t],
                    np.asarray([track.hp[anchor_t]], dtype=np.float32),
                ],
                axis=0,
            )
            context_cont[out_idx] = self.normalizer.normalize_context(current_context)
            team[out_idx] = TEAM_TO_ONEHOT[track.team]

        future_raw = (
            target_track.position[anchor_t + 1 : anchor_t + self.future_length + 1]
            - target_anchor_pos[None, :]
        ).astype(np.float32, copy=False)
        future = self.normalizer.normalize_output(future_raw)

        return {
            "past": torch.from_numpy(past),
            "context_cont": torch.from_numpy(context_cont),
            "team": torch.from_numpy(team),
            "future": torch.from_numpy(future),
            "future_raw": torch.from_numpy(future_raw),
            "dest": torch.from_numpy(future[-1]),
            "dest_raw": torch.from_numpy(future_raw[-1]),
            "anchor_abs": torch.from_numpy(target_anchor_pos.astype(np.float32)),
            "target_team": torch.tensor(0 if target_track.team == "BLUE" else 1, dtype=torch.long),
        }


def uav_pecnet_collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_agents = max(item["past"].shape[0] for item in batch)
    past_length = batch[0]["past"].shape[1]
    future_length = batch[0]["future"].shape[0]

    past = torch.zeros((batch_size, max_agents, past_length, 7), dtype=torch.float32)
    context_cont = torch.zeros((batch_size, max_agents, 7), dtype=torch.float32)
    team = torch.zeros((batch_size, max_agents, 2), dtype=torch.float32)
    valid_agent_mask = torch.zeros((batch_size, max_agents), dtype=torch.bool)

    future = torch.zeros((batch_size, future_length, 3), dtype=torch.float32)
    future_raw = torch.zeros((batch_size, future_length, 3), dtype=torch.float32)
    dest = torch.zeros((batch_size, 3), dtype=torch.float32)
    dest_raw = torch.zeros((batch_size, 3), dtype=torch.float32)
    anchor_abs = torch.zeros((batch_size, 3), dtype=torch.float32)
    target_team = torch.zeros((batch_size,), dtype=torch.long)

    for batch_index, item in enumerate(batch):
        agent_count = item["past"].shape[0]
        past[batch_index, :agent_count] = item["past"]
        context_cont[batch_index, :agent_count] = item["context_cont"]
        team[batch_index, :agent_count] = item["team"]
        valid_agent_mask[batch_index, :agent_count] = True

        future[batch_index] = item["future"]
        future_raw[batch_index] = item["future_raw"]
        dest[batch_index] = item["dest"]
        dest_raw[batch_index] = item["dest_raw"]
        anchor_abs[batch_index] = item["anchor_abs"]
        target_team[batch_index] = item["target_team"]

    social_mask = (
        valid_agent_mask[:, :, None] & valid_agent_mask[:, None, :]
    ).to(dtype=torch.float32)

    return {
        "past": past,
        "context_cont": context_cont,
        "team": team,
        "valid_agent_mask": valid_agent_mask,
        "social_mask": social_mask,
        "future": future,
        "future_raw": future_raw,
        "dest": dest,
        "dest_raw": dest_raw,
        "anchor_abs": anchor_abs,
        "target_team": target_team,
    }


def build_uav_datasets(
    csv_path: Path,
    split_stats_path: Optional[Path],
    past_length: int,
    future_length: int,
    train_ratio: float = 0.8,
    split_seed: int = 123,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    stats_sample_limit: Optional[int] = None,
) -> Tuple[UAVPECNetDataset, UAVPECNetDataset, UAVFeatureNormalizer, Dict[str, object]]:
    episodes = load_uav_episodes(csv_path)
    all_episode_ids = [episode.episode_id for episode in episodes]
    train_episode_ids, val_episode_ids = load_episode_split(
        episode_ids=all_episode_ids,
        split_stats_path=split_stats_path,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )

    train_sample_indices = build_sample_indices(
        episodes=episodes,
        selected_episode_ids=train_episode_ids,
        past_length=past_length,
        future_length=future_length,
        sample_limit=max_train_samples,
    )
    val_sample_indices = build_sample_indices(
        episodes=episodes,
        selected_episode_ids=val_episode_ids,
        past_length=past_length,
        future_length=future_length,
        sample_limit=max_val_samples,
    )

    normalizer_indices = train_sample_indices
    if stats_sample_limit is not None:
        normalizer_indices = normalizer_indices[:stats_sample_limit]

    print(f"Loaded {len(episodes)} episodes from {csv_path}")
    print(
        "Built PECNet UAV samples: train={}, val={}".format(
            train_sample_indices.shape[0], val_sample_indices.shape[0]
        )
    )
    print("Computing feature normalizer...")
    normalizer = compute_normalizer(
        episodes=episodes,
        sample_indices=normalizer_indices,
        past_length=past_length,
        future_length=future_length,
    )

    train_dataset = UAVPECNetDataset(
        episodes=episodes,
        sample_indices=train_sample_indices,
        past_length=past_length,
        future_length=future_length,
        normalizer=normalizer,
    )
    val_dataset = UAVPECNetDataset(
        episodes=episodes,
        sample_indices=val_sample_indices,
        past_length=past_length,
        future_length=future_length,
        normalizer=normalizer,
    )

    metadata = {
        "num_episodes": len(episodes),
        "train_episode_ids": train_episode_ids,
        "val_episode_ids": val_episode_ids,
        "train_num_samples": int(train_sample_indices.shape[0]),
        "val_num_samples": int(val_sample_indices.shape[0]),
    }
    return train_dataset, val_dataset, normalizer, metadata


def build_uav_val_dataset(
    csv_path: Path,
    split_stats_path: Optional[Path],
    past_length: int,
    future_length: int,
    normalizer: UAVFeatureNormalizer,
    train_ratio: float = 0.8,
    split_seed: int = 123,
    max_val_samples: Optional[int] = None,
) -> Tuple[UAVPECNetDataset, Dict[str, object]]:
    episodes = load_uav_episodes(csv_path)
    all_episode_ids = [episode.episode_id for episode in episodes]
    _, val_episode_ids = load_episode_split(
        episode_ids=all_episode_ids,
        split_stats_path=split_stats_path,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )
    val_sample_indices = build_sample_indices(
        episodes=episodes,
        selected_episode_ids=val_episode_ids,
        past_length=past_length,
        future_length=future_length,
        sample_limit=max_val_samples,
    )
    val_dataset = UAVPECNetDataset(
        episodes=episodes,
        sample_indices=val_sample_indices,
        past_length=past_length,
        future_length=future_length,
        normalizer=normalizer,
    )
    metadata = {
        "num_episodes": len(episodes),
        "val_episode_ids": val_episode_ids,
        "val_num_samples": int(val_sample_indices.shape[0]),
    }
    return val_dataset, metadata
