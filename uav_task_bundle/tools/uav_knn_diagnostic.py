#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import create_indices, get_val_mask


POS_SLICE = slice(0, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="kNN ambiguity diagnostic for UAV synthetic trajectory prediction."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="diffusion_policy/config/train_uav_synthetic_6step_big.yaml",
        help="Training config used to infer horizon, split, and dataset path.",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default=None,
        help="Optional override for the dataset zarr path.",
    )
    parser.add_argument(
        "--obs_key",
        type=str,
        default="uav_observations",
        help="Replay buffer key for observations.",
    )
    parser.add_argument(
        "--feature_modes",
        type=str,
        nargs="+",
        default=["relative_obs", "aligned_obs"],
        choices=["global_obs", "relative_obs", "aligned_obs"],
        help="Context feature constructions to evaluate.",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 5, 20],
        help="Neighbor counts to evaluate.",
    )
    parser.add_argument(
        "--max_train_windows",
        type=int,
        default=20000,
        help="Maximum number of train windows to index for kNN.",
    )
    parser.add_argument(
        "--max_val_windows",
        type=int,
        default=2000,
        help="Maximum number of validation windows to score.",
    )
    parser.add_argument(
        "--distance_batch_size",
        type=int,
        default=256,
        help="Number of validation queries per batched distance computation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=32,
        help="Sampling seed for train/val window subsets.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to write the diagnostic summary as JSON.",
    )
    return parser.parse_args()


def load_experiment_spec(config_path: Path) -> dict[str, Any]:
    cfg = OmegaConf.load(config_path)

    task_name = None
    for item in cfg.get("defaults", []):
        if isinstance(item, DictConfig) and "task" in item:
            task_name = str(item["task"])
            break
        if isinstance(item, dict) and "task" in item:
            task_name = str(item["task"])
            break
    if task_name is None:
        raise ValueError(f"Could not resolve task default from {config_path}")

    task_cfg_path = config_path.parent / "task" / f"{task_name}.yaml"
    task_cfg = OmegaConf.load(task_cfg_path)
    task_cfg_raw = OmegaConf.to_container(task_cfg, resolve=False)
    dataset_cfg = task_cfg_raw["dataset"]

    zarr_path = str(dataset_cfg["zarr_path"])
    zarr_path = zarr_path.replace("${hydra:runtime.cwd}", str(Path.cwd()))

    obs_slice = tuple(int(x) for x in dataset_cfg["obs_slice"])

    return {
        "task_name": task_name,
        "config_path": str(config_path),
        "task_config_path": str(task_cfg_path),
        "zarr_path": zarr_path,
        "horizon": int(cfg.horizon),
        "n_obs_steps": int(cfg.n_obs_steps),
        "n_action_steps": int(cfg.n_action_steps),
        "obs_slice": obs_slice,
        "val_ratio": float(dataset_cfg["val_ratio"]),
        "split_seed": int(dataset_cfg["seed"]),
    }


def compute_action_from_obs(obs_seq: np.ndarray) -> np.ndarray:
    pos = obs_seq[..., POS_SLICE]
    action = np.zeros_like(pos, dtype=np.float32)
    action[1:] = pos[1:] - pos[:-1]
    return action


def sample_windows(
    obs: np.ndarray,
    indices: np.ndarray,
    n_obs_steps: int,
    n_action_steps: int,
) -> dict[str, np.ndarray]:
    n = len(indices)
    obs_dim = obs.shape[-1]
    past_obs = np.empty((n, n_obs_steps, obs_dim), dtype=np.float32)
    future_action = np.empty((n, n_action_steps, 3), dtype=np.float32)
    start_pos = np.empty((n, 3), dtype=np.float32)

    for out_idx, (buffer_start, buffer_end, sample_start, sample_end) in enumerate(indices):
        if sample_start != 0 or sample_end != (buffer_end - buffer_start):
            raise ValueError("This diagnostic expects unpadded contiguous windows.")
        seq = obs[buffer_start:buffer_end]
        action = compute_action_from_obs(seq)
        past_obs[out_idx] = seq[:n_obs_steps]
        future_action[out_idx] = action[n_obs_steps:n_obs_steps + n_action_steps]
        start_pos[out_idx] = seq[n_obs_steps - 1, POS_SLICE]

    return {
        "past_obs": past_obs,
        "future_action": future_action,
        "start_pos": start_pos,
    }


def rotate_xy(arr: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    x = arr[..., 0]
    y = arr[..., 1]
    xr = cos * x + sin * y
    yr = -sin * x + cos * y
    return np.stack([xr, yr], axis=-1).astype(np.float32)


def canonicalize_local_frame(
    past_obs: np.ndarray,
    future_action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    canon_obs = np.array(past_obs, copy=True)
    canon_obs[:, :, POS_SLICE] -= canon_obs[:, -1:, POS_SLICE]

    vel_xy = past_obs[:, -1, 3:5]
    fallback_xy = past_obs[:, -1, 0:2] - past_obs[:, -2, 0:2]
    vel_norm = np.linalg.norm(vel_xy, axis=1, keepdims=True)
    heading_xy = np.where(vel_norm > 1e-6, vel_xy, fallback_xy)
    heading_norm = np.linalg.norm(heading_xy, axis=1, keepdims=True)
    heading_norm = np.clip(heading_norm, 1e-6, None)
    cos = (heading_xy[:, 0:1] / heading_norm).astype(np.float32)
    sin = (heading_xy[:, 1:2] / heading_norm).astype(np.float32)

    canon_obs[:, :, 0:2] = rotate_xy(canon_obs[:, :, 0:2], cos, sin)
    canon_obs[:, :, 3:5] = rotate_xy(canon_obs[:, :, 3:5], cos, sin)

    canon_future = np.array(future_action, copy=True)
    canon_future[:, :, 0:2] = rotate_xy(canon_future[:, :, 0:2], cos, sin)
    return canon_obs, canon_future


def prepare_mode_tensors(
    data: dict[str, np.ndarray],
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "global_obs":
        mode_obs = np.array(data["past_obs"], copy=True)
        mode_future = np.array(data["future_action"], copy=True)
    elif mode == "relative_obs":
        mode_obs = np.array(data["past_obs"], copy=True)
        mode_obs[:, :, POS_SLICE] -= mode_obs[:, -1:, POS_SLICE]
        mode_future = np.array(data["future_action"], copy=True)
    elif mode == "aligned_obs":
        mode_obs, mode_future = canonicalize_local_frame(
            past_obs=data["past_obs"],
            future_action=data["future_action"],
        )
    else:
        raise ValueError(f"Unsupported feature mode: {mode}")
    return mode_obs, mode_future


def build_features(past_obs: np.ndarray) -> np.ndarray:
    return past_obs.reshape(len(past_obs), -1).astype(np.float32)


def standardize_features(
    train_feat: np.ndarray,
    val_feat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_feat.mean(axis=0, keepdims=True)
    std = train_feat.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_feat - mean) / std, (val_feat - mean) / std


def knn_search(
    train_feat: np.ndarray,
    query_feat: np.ndarray,
    k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_feat = np.ascontiguousarray(train_feat, dtype=np.float32)
    query_feat = np.ascontiguousarray(query_feat, dtype=np.float32)
    train_sq = np.sum(train_feat * train_feat, axis=1, dtype=np.float32)

    all_indices = []
    all_distances = []
    train_feat_t = train_feat.T

    for start in range(0, len(query_feat), batch_size):
        end = min(start + batch_size, len(query_feat))
        batch = query_feat[start:end]
        batch_sq = np.sum(batch * batch, axis=1, dtype=np.float32, keepdims=True)
        dists = batch_sq + train_sq[None, :] - (2.0 * (batch @ train_feat_t))
        dists = np.maximum(dists, 0.0)

        topk_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        topk_dist = np.take_along_axis(dists, topk_idx, axis=1)
        order = np.argsort(topk_dist, axis=1)
        topk_idx = np.take_along_axis(topk_idx, order, axis=1)
        topk_dist = np.take_along_axis(topk_dist, order, axis=1)

        all_indices.append(topk_idx.astype(np.int64))
        all_distances.append(topk_dist.astype(np.float32))

    return np.concatenate(all_indices, axis=0), np.concatenate(all_distances, axis=0)


def trajectory_metrics(
    pred_action: np.ndarray,
    gt_action: np.ndarray,
    start_pos: np.ndarray,
) -> dict[str, np.ndarray]:
    pred_xyz = start_pos[:, None, :] + np.cumsum(pred_action, axis=1)
    gt_xyz = start_pos[:, None, :] + np.cumsum(gt_action, axis=1)
    dist = np.linalg.norm(pred_xyz - gt_xyz, axis=-1)

    gt_path_len = np.linalg.norm(gt_action, axis=-1).sum(axis=1)
    gt_path_len = np.clip(gt_path_len, 1e-6, None)

    diff = pred_action - gt_action
    return {
        "action_mse": np.mean(diff * diff, axis=(1, 2)),
        "action_mae": np.mean(np.abs(diff), axis=(1, 2)),
        "ade_m": dist.mean(axis=1),
        "fde_m": dist[:, -1],
        "path_error_pct": dist.sum(axis=1) / gt_path_len * 100.0,
        "fde_ratio_pct": dist[:, -1] / gt_path_len * 100.0,
    }


def summarize_metrics(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {key: float(np.mean(value)) for key, value in metrics.items()}


def neighbor_spread(neighbor_action: np.ndarray) -> dict[str, float]:
    neighbor_xyz = np.cumsum(neighbor_action, axis=2)
    mean_xyz = neighbor_xyz.mean(axis=1, keepdims=True)
    dist = np.linalg.norm(neighbor_xyz - mean_xyz, axis=-1)
    final_dist = np.linalg.norm(
        neighbor_xyz[:, :, -1, :] - mean_xyz[:, 0, -1, :][:, None, :],
        axis=-1,
    )
    return {
        "neighbor_ade_spread_m": float(np.mean(dist.mean(axis=2))),
        "neighbor_fde_spread_m": float(np.mean(final_dist)),
    }


def evaluate_neighbors(
    gt_action: np.ndarray,
    start_pos: np.ndarray,
    neighbor_action: np.ndarray,
) -> dict[str, dict[str, float]]:
    mean_pred = neighbor_action.mean(axis=1)
    mean_metrics = summarize_metrics(trajectory_metrics(mean_pred, gt_action, start_pos))

    nn1_pred = neighbor_action[:, 0]
    nn1_metrics = summarize_metrics(trajectory_metrics(nn1_pred, gt_action, start_pos))

    gt_path_len = np.linalg.norm(gt_action, axis=-1).sum(axis=1)
    gt_path_len = np.clip(gt_path_len, 1e-6, None)

    neighbor_xyz = np.cumsum(neighbor_action, axis=2)
    gt_xyz = np.cumsum(gt_action[:, None, :, :], axis=2)
    dist = np.linalg.norm(neighbor_xyz - gt_xyz, axis=-1)
    path_error_pct = dist.sum(axis=2) / gt_path_len[:, None] * 100.0
    best_idx = np.argmin(path_error_pct, axis=1)
    oracle_pred = neighbor_action[np.arange(len(neighbor_action)), best_idx]
    oracle_metrics = summarize_metrics(trajectory_metrics(oracle_pred, gt_action, start_pos))

    spread = neighbor_spread(neighbor_action)
    return {
        "nn1": nn1_metrics,
        "knn_mean": mean_metrics,
        "oracle_best_of_k": oracle_metrics,
        "neighbor_spread": spread,
    }


def maybe_subsample(
    indices: np.ndarray,
    max_windows: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_windows is None or len(indices) <= max_windows:
        return indices
    chosen = rng.choice(len(indices), size=int(max_windows), replace=False)
    return indices[np.sort(chosen)]


def print_metric_block(title: str, metrics: dict[str, float]) -> None:
    print(title)
    print(
        "  "
        f"path_err_pct={metrics['path_error_pct']:.3f}  "
        f"ade_m={metrics['ade_m']:.4f}  "
        f"fde_m={metrics['fde_m']:.4f}  "
        f"action_mae={metrics['action_mae']:.5f}  "
        f"action_mse={metrics['action_mse']:.6f}"
    )


def main() -> None:
    args = parse_args()
    spec = load_experiment_spec(Path(args.config))
    if args.zarr_path is not None:
        spec["zarr_path"] = args.zarr_path

    if spec["n_obs_steps"] + spec["n_action_steps"] > spec["horizon"]:
        raise ValueError("Expected n_obs_steps + n_action_steps <= horizon")

    print("Loading dataset...")
    print(json.dumps(spec, indent=2))

    replay_buffer = ReplayBuffer.copy_from_path(
        spec["zarr_path"],
        keys=[args.obs_key],
    )
    obs_raw = replay_buffer[args.obs_key][:]
    obs = obs_raw.reshape(obs_raw.shape[0], -1)
    obs = obs[:, spec["obs_slice"][0]:spec["obs_slice"][1]].astype(np.float32, copy=False)
    episode_ends = replay_buffer.episode_ends[:]

    val_mask = get_val_mask(
        n_episodes=len(episode_ends),
        val_ratio=spec["val_ratio"],
        seed=spec["split_seed"],
    )
    train_mask = ~val_mask

    train_indices = create_indices(
        episode_ends=episode_ends,
        sequence_length=spec["horizon"],
        pad_before=0,
        pad_after=0,
        episode_mask=train_mask,
        debug=False,
    )
    val_indices = create_indices(
        episode_ends=episode_ends,
        sequence_length=spec["horizon"],
        pad_before=0,
        pad_after=0,
        episode_mask=val_mask,
        debug=False,
    )

    rng = np.random.default_rng(args.seed)
    sampled_train_indices = maybe_subsample(train_indices, args.max_train_windows, rng)
    sampled_val_indices = maybe_subsample(val_indices, args.max_val_windows, rng)

    print(
        f"Window counts: train_total={len(train_indices)}  val_total={len(val_indices)}  "
        f"train_used={len(sampled_train_indices)}  val_used={len(sampled_val_indices)}"
    )

    train_data = sample_windows(
        obs=obs,
        indices=sampled_train_indices,
        n_obs_steps=spec["n_obs_steps"],
        n_action_steps=spec["n_action_steps"],
    )
    val_data = sample_windows(
        obs=obs,
        indices=sampled_val_indices,
        n_obs_steps=spec["n_obs_steps"],
        n_action_steps=spec["n_action_steps"],
    )

    max_k = max(args.k_values)
    summary: dict[str, Any] = {
        "spec": spec,
        "seed": args.seed,
        "feature_modes": args.feature_modes,
        "k_values": args.k_values,
        "max_train_windows": args.max_train_windows,
        "max_val_windows": args.max_val_windows,
        "counts": {
            "train_total": int(len(train_indices)),
            "val_total": int(len(val_indices)),
            "train_used": int(len(sampled_train_indices)),
            "val_used": int(len(sampled_val_indices)),
        },
        "results": {},
    }

    for mode in args.feature_modes:
        print(f"\n=== Feature Mode: {mode} ===")
        train_mode_obs, train_mode_future = prepare_mode_tensors(train_data, mode)
        val_mode_obs, val_mode_future = prepare_mode_tensors(val_data, mode)

        train_feat = build_features(train_mode_obs)
        val_feat = build_features(val_mode_obs)
        train_feat, val_feat = standardize_features(train_feat, val_feat)

        knn_idx, knn_dist = knn_search(
            train_feat=train_feat,
            query_feat=val_feat,
            k=max_k,
            batch_size=args.distance_batch_size,
        )

        mode_result: dict[str, Any] = {
            "nearest_feature_distance": {
                "mean": float(np.mean(knn_dist[:, 0])),
                "p50": float(np.percentile(knn_dist[:, 0], 50)),
                "p90": float(np.percentile(knn_dist[:, 0], 90)),
            },
            "k_results": {},
        }

        print(
            "nearest feature distance: "
            f"mean={mode_result['nearest_feature_distance']['mean']:.4f}  "
            f"p50={mode_result['nearest_feature_distance']['p50']:.4f}  "
            f"p90={mode_result['nearest_feature_distance']['p90']:.4f}"
        )

        for k in args.k_values:
            neighbor_action = train_mode_future[knn_idx[:, :k]]
            result = evaluate_neighbors(
                gt_action=val_mode_future,
                start_pos=np.zeros_like(val_data["start_pos"]),
                neighbor_action=neighbor_action,
            )
            mode_result["k_results"][str(k)] = result

            print(f"\n  k={k}")
            print_metric_block("  1-NN", result["nn1"])
            print_metric_block("  kNN-Mean", result["knn_mean"])
            print_metric_block("  Oracle-Best-of-k", result["oracle_best_of_k"])
            print(
                "  "
                f"neighbor_spread_ade={result['neighbor_spread']['neighbor_ade_spread_m']:.4f}  "
                f"neighbor_spread_fde={result['neighbor_spread']['neighbor_fde_spread_m']:.4f}"
            )
            print(
                "  "
                f"oracle_gap_vs_mean={result['knn_mean']['path_error_pct'] - result['oracle_best_of_k']['path_error_pct']:.3f} pct"
            )

        summary["results"][mode] = mode_result

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved summary: {output_path}")


if __name__ == "__main__":
    main()
