import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import zarr


OBS_DIM = 6
ACTION_DIM = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert UZH_FPV groundtruth trajectories into diffusion-policy "
            "zarr format with obs=[x,y,z,vx,vy,vz] and action=delta_position."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/dfc/Documents/zdy_documents/diffusion_policy/uav_task_bundle/UZH_FPV",
        help="Root directory containing UZH_FPV sequence folders.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/dfc/Documents/zdy_documents/diffusion_policy/uav_task_bundle/data/uzh_fpv/uzh_fpv_25hz_delta.zarr",
        help="Output zarr directory path.",
    )
    parser.add_argument(
        "--sequence_glob",
        type=str,
        default="*",
        help="Glob pattern relative to input_dir for locating sequence folders.",
    )
    parser.add_argument(
        "--resample_dt_sec",
        type=float,
        default=0.04,
        help="Uniform resampling interval in seconds. 0.04 corresponds to 25Hz.",
    )
    parser.add_argument(
        "--min_episode_len",
        type=int,
        default=100,
        help="Minimum number of resampled frames required to keep one episode.",
    )
    parser.add_argument(
        "--delta_scale_quantile",
        type=float,
        default=0.999,
        help="Per-axis quantile of |delta_position| used for action normalization scale.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output zarr if it already exists.",
    )
    return parser.parse_args()


def find_sequence_dirs(input_dir: Path, sequence_glob: str) -> List[Path]:
    sequence_dirs = sorted(
        p for p in input_dir.glob(sequence_glob)
        if p.is_dir() and (p / "groundtruth.txt").is_file()
    )
    if not sequence_dirs:
        raise FileNotFoundError(
            f"No sequence directories with groundtruth.txt matched {sequence_glob} under {input_dir}"
        )
    return sequence_dirs


def load_groundtruth(sequence_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    gt_path = sequence_dir / "groundtruth.txt"
    data = np.loadtxt(gt_path, comments="#", dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 4:
        raise ValueError(f"{gt_path} should have at least 4 columns, got shape {data.shape}")

    timestamp = data[:, 0]
    position = data[:, 1:4]

    order = np.argsort(timestamp)
    timestamp = timestamp[order]
    position = position[order]

    keep = np.ones(len(timestamp), dtype=bool)
    keep[1:] = np.diff(timestamp) > 0
    timestamp = timestamp[keep]
    position = position[keep]

    if len(timestamp) < 2:
        raise ValueError(f"{gt_path} has fewer than 2 valid timestamps after deduplication.")
    return timestamp, position


def resample_sequence(timestamp: np.ndarray, position: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    start_t = float(timestamp[0])
    end_t = float(timestamp[-1])
    if end_t <= start_t:
        raise ValueError("Non-positive sequence duration after sorting.")

    num_steps = int(np.floor((end_t - start_t) / dt)) + 1
    resampled_t = start_t + np.arange(num_steps, dtype=np.float64) * dt
    if resampled_t.shape[0] < 2:
        raise ValueError("Resampled sequence is too short.")

    resampled_pos = np.stack(
        [np.interp(resampled_t, timestamp, position[:, axis]) for axis in range(3)],
        axis=-1,
    )
    return resampled_t, resampled_pos


def build_obs_and_action(resampled_pos: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    local_pos = (resampled_pos - resampled_pos[0]).astype(np.float32, copy=False)
    edge_order = 2 if local_pos.shape[0] >= 3 else 1
    velocity = np.gradient(local_pos, dt, axis=0, edge_order=edge_order).astype(np.float32, copy=False)

    obs = np.concatenate([local_pos, velocity], axis=-1).astype(np.float32, copy=False)
    action = np.zeros_like(local_pos, dtype=np.float32)
    action[1:] = local_pos[1:] - local_pos[:-1]
    return obs, action


def convert_uzh_fpv_to_zarr(
    input_dir: Path,
    output_path: Path,
    sequence_glob: str,
    resample_dt_sec: float,
    min_episode_len: int,
    delta_scale_quantile: float,
    overwrite: bool,
) -> None:
    sequence_dirs = find_sequence_dirs(input_dir, sequence_glob)

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_obs: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    episode_ends: List[int] = []
    episode_names: List[str] = []
    skipped_sequences = 0

    current_idx = 0
    for seq_idx, seq_dir in enumerate(sequence_dirs, start=1):
        print(f"[{seq_idx}/{len(sequence_dirs)}] processing {seq_dir.name}")
        try:
            timestamp, position = load_groundtruth(seq_dir)
            _, resampled_pos = resample_sequence(timestamp, position, resample_dt_sec)
            obs, action = build_obs_and_action(resampled_pos, resample_dt_sec)
        except Exception as exc:
            skipped_sequences += 1
            print(f"  skipped: {exc}")
            continue

        if obs.shape[0] < int(min_episode_len):
            skipped_sequences += 1
            print(f"  skipped: resampled length {obs.shape[0]} < min_episode_len {min_episode_len}")
            continue

        all_obs.append(obs)
        all_actions.append(action)
        current_idx += obs.shape[0]
        episode_ends.append(current_idx)
        episode_names.append(seq_dir.name)

    if not all_obs:
        raise RuntimeError("No valid UZH_FPV episodes were kept after conversion.")

    full_obs = np.concatenate(all_obs, axis=0).astype(np.float32, copy=False)
    full_actions = np.concatenate(all_actions, axis=0).astype(np.float32, copy=False)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)

    delta_max_abs = np.max(np.abs(full_actions), axis=0)
    scale_q = float(delta_scale_quantile)
    if scale_q >= 1.0:
        delta_scale_abs = delta_max_abs.copy()
    else:
        delta_scale_abs = np.quantile(np.abs(full_actions), scale_q, axis=0).astype(np.float32)
    delta_scale_abs = np.maximum(delta_scale_abs, 1e-6).astype(np.float32)

    root = zarr.open(str(output_path), mode="w")
    root.attrs["source_dataset"] = "UZH_FPV"
    root.attrs["obs_dim"] = OBS_DIM
    root.attrs["action_dim"] = ACTION_DIM
    root.attrs["obs_description"] = ["x", "y", "z", "vx", "vy", "vz"]
    root.attrs["action_description"] = ["dx", "dy", "dz"]
    root.attrs["coordinate_frame"] = "episode_local_xyz_m"
    root.attrs["meters_per_unit"] = 1.0
    root.attrs["resample_dt_sec"] = float(resample_dt_sec)
    root.attrs["delta_max_abs"] = delta_max_abs.astype(np.float32).tolist()
    root.attrs["delta_scale_abs"] = delta_scale_abs.astype(np.float32).tolist()
    root.attrs["delta_scale_quantile"] = scale_q
    root.attrs["trajectory_count"] = int(len(episode_ends))
    episode_lengths = np.diff(np.concatenate([[0], episode_ends_arr]))
    root.attrs["episode_len_min"] = int(episode_lengths.min())
    root.attrs["episode_len_mean"] = float(np.mean(episode_lengths))
    root.attrs["episode_len_max"] = int(episode_lengths.max())
    root.attrs["action_target"] = "delta_position"

    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    data_group.create_dataset("uav_observations", data=full_obs, chunks=(1024, OBS_DIM))
    data_group.create_dataset("uav_actions", data=full_actions, chunks=(1024, ACTION_DIM))
    meta_group.create_dataset("episode_ends", data=episode_ends_arr)
    meta_group.create_dataset("episode_names", data=np.asarray(episode_names, dtype=object), object_codec=zarr.codecs.VLenUTF8())

    print(f"Saved: {output_path}")
    print(f"Sequence dirs processed: {len(sequence_dirs)}")
    print(f"Episodes kept: {len(episode_ends)}")
    print(f"Total samples: {full_obs.shape[0]}")
    print(f"Observation shape: {full_obs.shape}")
    print(f"Action shape: {full_actions.shape}")
    print(
        "Episode length min/mean/max: "
        f"{root.attrs['episode_len_min']}/{root.attrs['episode_len_mean']:.2f}/{root.attrs['episode_len_max']}"
    )
    print(f"Skipped sequences: {skipped_sequences}")
    print(f"Raw delta max abs: {delta_max_abs.astype(np.float32).tolist()}")
    print(f"Action scale abs (q={scale_q}): {delta_scale_abs.astype(np.float32).tolist()}")


if __name__ == "__main__":
    args = parse_args()
    convert_uzh_fpv_to_zarr(
        input_dir=Path(os.path.expanduser(args.input_dir)),
        output_path=Path(os.path.expanduser(args.output_path)),
        sequence_glob=args.sequence_glob,
        resample_dt_sec=float(args.resample_dt_sec),
        min_episode_len=int(args.min_episode_len),
        delta_scale_quantile=float(args.delta_scale_quantile),
        overwrite=bool(args.overwrite),
    )
