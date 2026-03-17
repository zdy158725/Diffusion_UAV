import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import zarr


OBS_DIM = 6
ACTION_DIM = 3


def load_single_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps: List[float] = []
    positions: List[List[float]] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", "tx", "ty", "tz"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{csv_path} missing required columns {required}")
        for row in reader:
            timestamps.append(float(row["timestamp"]))
            positions.append([
                float(row["tx"]),
                float(row["ty"]),
                float(row["tz"]),
            ])

    ts = np.asarray(timestamps, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"{csv_path} produced invalid position shape {pos.shape}")

    vel = np.zeros_like(pos, dtype=np.float32)
    action = np.zeros_like(pos, dtype=np.float32)
    if len(pos) > 1:
        dt = np.diff(ts)
        dt = np.maximum(dt, 1e-6)
        delta = pos[1:] - pos[:-1]
        vel[1:] = delta / dt[:, None]
        vel[0] = vel[1]
        action[1:] = delta

    obs = np.concatenate([pos, vel], axis=-1).astype(np.float32)
    return obs, action, ts


def convert_uav_synthetic_csv_dir_to_zarr(
    input_dir: str,
    output_path: str,
    glob_pattern: str = '*.csv',
    min_episode_len: int = 15,
    max_files: int = None,
    meters_per_unit: float = 1.0,
) -> None:
    input_root = Path(input_dir).expanduser().resolve()
    output_root = Path(output_path).expanduser().resolve()
    os.makedirs(output_root.parent, exist_ok=True)

    csv_files = sorted(input_root.glob(glob_pattern))
    if max_files is not None:
        csv_files = csv_files[: int(max_files)]
    if not csv_files:
        raise FileNotFoundError(f'No files matched {glob_pattern} in {input_root}')

    all_obs: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    episode_ends: List[int] = []
    kept_files: List[str] = []
    lengths: List[int] = []
    dt_values: List[float] = []
    skipped_short = 0
    current_idx = 0

    for csv_path in csv_files:
        obs, action, ts = load_single_csv(csv_path)
        if len(obs) < int(min_episode_len):
            skipped_short += 1
            continue
        if len(ts) > 1:
            dt_values.extend(np.diff(ts).tolist())
        all_obs.append(obs)
        all_actions.append(action)
        current_idx += len(obs)
        episode_ends.append(current_idx)
        kept_files.append(csv_path.name)
        lengths.append(len(obs))

    if not all_obs:
        raise RuntimeError('No usable trajectories found after filtering.')

    full_obs = np.concatenate(all_obs, axis=0).astype(np.float32)
    full_actions = np.concatenate(all_actions, axis=0).astype(np.float32)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)

    max_abs_delta = np.max(np.abs(full_actions), axis=0).astype(np.float32)
    max_abs_delta = np.maximum(max_abs_delta, 1e-6)
    median_dt = float(np.median(np.asarray(dt_values, dtype=np.float64))) if dt_values else None

    store = zarr.DirectoryStore(str(output_root))
    root = zarr.group(store=store, overwrite=True)
    root.attrs['author'] = 'OpenAI Codex'
    root.attrs['version'] = '1.0-uav-synthetic-csv'
    root.attrs['description'] = 'Single-UAV trajectory dataset converted from Synthetic-UAV-Flight-Trajectories CSV files'
    root.attrs['source_dataset'] = 'riotu-lab/Synthetic-UAV-Flight-Trajectories'
    root.attrs['obs_description'] = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    root.attrs['action_target'] = 'delta_position'
    root.attrs['delta_max_abs'] = max_abs_delta.tolist()
    root.attrs['meters_per_unit'] = float(meters_per_unit)
    root.attrs['input_glob'] = glob_pattern
    root.attrs['trajectory_count'] = len(kept_files)
    root.attrs['skipped_short_count'] = int(skipped_short)
    root.attrs['median_dt_sec'] = median_dt
    root.attrs['episode_len_min'] = int(np.min(lengths))
    root.attrs['episode_len_max'] = int(np.max(lengths))
    root.attrs['episode_len_mean'] = float(np.mean(lengths))

    data_group = root.create_group('data')
    data_group.create_dataset('uav_observations', data=full_obs, chunks=(1024, OBS_DIM))
    data_group.create_dataset('uav_actions', data=full_actions, chunks=(1024, ACTION_DIM))

    meta_group = root.create_group('meta')
    meta_group.create_dataset('episode_ends', data=episode_ends_arr)
    meta_group.create_dataset('episode_file_names', data=np.asarray(kept_files, dtype=object), object_codec=zarr.codecs.VLenUTF8())

    print(f'Saved: {output_root}')
    print(f'Trajectories kept: {len(kept_files)}')
    print(f'Total samples: {full_obs.shape[0]}')
    print(f'Observation shape: {full_obs.shape}')
    print(f'Action shape: {full_actions.shape}')
    print(f'Episode length min/mean/max: {int(np.min(lengths))}/{float(np.mean(lengths)):.2f}/{int(np.max(lengths))}')
    if median_dt is not None:
        print(f'Median dt (s): {median_dt:.6f}')
    print(f'Delta max abs: {max_abs_delta.tolist()}')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert UAV synthetic trajectory CSV files into diffusion-policy zarr format.')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/home/dfc/Documents/zdy_documents/diffusion_policy/uav_task_bundle/uav_synthetic_data',
        help='Directory containing per-trajectory CSV files.',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='/home/dfc/Documents/zdy_documents/diffusion_policy/uav_task_bundle/data/uav_synthetic/uav_synthetic_5k.zarr',
        help='Output zarr directory path.',
    )
    parser.add_argument('--glob_pattern', type=str, default='*.csv')
    parser.add_argument('--min_episode_len', type=int, default=15)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--meters_per_unit', type=float, default=1.0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert_uav_synthetic_csv_dir_to_zarr(
        input_dir=args.input_dir,
        output_path=args.output_path,
        glob_pattern=args.glob_pattern,
        min_episode_len=args.min_episode_len,
        max_files=args.max_files,
        meters_per_unit=args.meters_per_unit,
    )
