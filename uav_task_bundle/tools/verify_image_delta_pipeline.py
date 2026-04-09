#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_verify_image_delta")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache_verify_image_delta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify an image+state+delta_position offline trajectory pipeline. "
            "Defaults are set for the current UZH_FPV hybrid-image setup, but "
            "the dataset and runner classes are configurable."
        )
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default=str(PROJECT_ROOT / "data/uzh_fpv/uzh_fpv_image_96.zarr"),
        help="Image zarr path.",
    )
    parser.add_argument(
        "--dataset_target",
        type=str,
        default="diffusion_policy.dataset.uzh_fpv_image_dataset.UZHFPVImageDataset",
        help="Dataset class import path.",
    )
    parser.add_argument(
        "--runner_target",
        type=str,
        default="diffusion_policy.env_runner.uzh_fpv_image_offline_runner.UZHFPVImageOfflineRunner",
        help="Offline runner class import path.",
    )
    parser.add_argument("--horizon", type=int, default=75, help="Window length.")
    parser.add_argument("--n_obs_steps", type=int, default=25, help="Observed steps.")
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=50,
        help="Predicted future action steps used by the policy.",
    )
    parser.add_argument("--pad_before", type=int, default=0, help="Sampler left padding.")
    parser.add_argument("--pad_after", type=int, default=0, help="Sampler right padding.")
    parser.add_argument("--seed", type=int, default=42, help="Dataset split seed.")
    parser.add_argument("--val_ratio", type=float, default=0.17, help="Validation ratio.")
    parser.add_argument(
        "--max_train_episodes",
        type=int,
        default=None,
        help="Optional dataset training-episode cap.",
    )
    parser.add_argument(
        "--state_key",
        type=str,
        default="state",
        help="Observation state key inside obs dict.",
    )
    parser.add_argument(
        "--position_slice",
        type=str,
        default="0:3",
        help="Position slice in state, e.g. 0:3.",
    )
    parser.add_argument(
        "--num_alignment_samples",
        type=int,
        default=32,
        help="How many windows to sample for action/state alignment checks.",
    )
    parser.add_argument(
        "--roundtrip_samples",
        type=int,
        default=4096,
        help="How many raw frames to use for normalizer roundtrip checks.",
    )
    parser.add_argument(
        "--runner_batch_size",
        type=int,
        default=16,
        help="Batch size used only when instantiating the runner.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save the summary as JSON.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional trained checkpoint for image-ablation diagnostics.",
    )
    parser.add_argument(
        "--workspace_target",
        type=str,
        default="diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace.TrainDiffusionTransformerHybridWorkspace",
        help="Workspace class import path used when loading --checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device used only for checkpoint-based image ablation checks.",
    )
    parser.add_argument(
        "--ablation_batches",
        type=int,
        default=10,
        help="How many validation batches to use for image-ablation diagnostics.",
    )
    parser.add_argument(
        "--ablation_batch_size",
        type=int,
        default=16,
        help="Validation batch size for image-ablation diagnostics.",
    )
    parser.add_argument(
        "--use_model",
        action="store_true",
        help="Use the raw model instead of EMA when loading --checkpoint.",
    )
    parser.add_argument(
        "--ablation_plot_path",
        type=str,
        default=None,
        help="Optional path to save the image-ablation comparison plot.",
    )
    return parser.parse_args()


def resolve_class(target: str):
    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def parse_slice(spec: str) -> slice:
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid slice spec: {spec}")
    start = None if parts[0] == "" else int(parts[0])
    end = None if parts[1] == "" else int(parts[1])
    return slice(start, end)


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def move_to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    return x


def choose_evenly(indices: np.ndarray, n: int) -> np.ndarray:
    if len(indices) == 0:
        return indices
    if len(indices) <= n:
        return indices
    picked = np.linspace(0, len(indices) - 1, num=n, dtype=int)
    return indices[picked]


def compute_episode_lengths(episode_ends: np.ndarray) -> np.ndarray:
    starts = np.concatenate([[0], episode_ends[:-1]])
    return episode_ends - starts


def check_alignment(
    dataset,
    n_obs_steps: int,
    n_action_steps: int,
    state_key: str,
    position_slice: slice,
    num_samples: int,
) -> dict[str, Any]:
    if dataset.horizon < (n_obs_steps + n_action_steps):
        raise ValueError(
            "horizon is shorter than n_obs_steps + n_action_steps, "
            "cannot verify future action/state alignment."
        )

    sampler_indices = dataset.sampler.indices
    full_window_mask = (
        (sampler_indices[:, 2] == 0)
        & (sampler_indices[:, 3] == dataset.horizon)
    )
    candidate_indices = np.nonzero(full_window_mask)[0]
    selected = choose_evenly(candidate_indices, max(num_samples, 1))

    max_abs_err = 0.0
    max_l2_err = 0.0
    mean_abs_errs = []
    mean_l2_errs = []

    for dataset_idx in selected.tolist():
        sample = dataset[dataset_idx]
        state = to_numpy(sample["obs"][state_key]).astype(np.float64)
        action = to_numpy(sample["action"]).astype(np.float64)

        anchor = state[n_obs_steps - 1, position_slice]
        future_action = action[n_obs_steps : n_obs_steps + n_action_steps]
        future_state = state[n_obs_steps : n_obs_steps + n_action_steps, position_slice]
        recon_state = anchor[None, :] + np.cumsum(future_action, axis=0)

        diff = recon_state - future_state
        max_abs_err = max(max_abs_err, float(np.max(np.abs(diff))))
        l2_err = np.linalg.norm(diff, axis=-1)
        max_l2_err = max(max_l2_err, float(np.max(l2_err)))
        mean_abs_errs.append(float(np.mean(np.abs(diff))))
        mean_l2_errs.append(float(np.mean(l2_err)))

    return {
        "checked_windows": int(len(selected)),
        "candidate_full_windows": int(len(candidate_indices)),
        "alignment_max_abs_error": max_abs_err,
        "alignment_max_l2_error": max_l2_err,
        "alignment_mean_abs_error": float(np.mean(mean_abs_errs)) if mean_abs_errs else float("nan"),
        "alignment_mean_l2_error": float(np.mean(mean_l2_errs)) if mean_l2_errs else float("nan"),
        "alignment_pass": bool(max_abs_err < 1e-5),
    }


def check_normalizer_roundtrip(dataset, roundtrip_samples: int) -> dict[str, Any]:
    normalizer = dataset.get_normalizer()
    n = min(int(roundtrip_samples), int(dataset.replay_buffer["state"].shape[0]))

    state = dataset.replay_buffer["state"][:n].astype(np.float32)
    action = dataset.replay_buffer["action"][:n].astype(np.float32)
    image = dataset.replay_buffer["img"][: min(n, 32)].astype(np.float32) / 255.0

    state_t = torch.from_numpy(state)
    action_t = torch.from_numpy(action)
    image_t = torch.from_numpy(np.moveaxis(image, -1, 1))

    state_recon = normalizer["state"].unnormalize(normalizer["state"].normalize(state_t))
    action_recon = normalizer["action"].unnormalize(normalizer["action"].normalize(action_t))
    image_recon = normalizer["image"].unnormalize(normalizer["image"].normalize(image_t))

    state_err = float(torch.max(torch.abs(state_recon - state_t)).item())
    action_err = float(torch.max(torch.abs(action_recon - action_t)).item())
    image_err = float(torch.max(torch.abs(image_recon - image_t)).item())

    state_norm = normalizer["state"].normalize(state_t)
    action_norm = normalizer["action"].normalize(action_t)

    return {
        "roundtrip_frames": int(n),
        "state_roundtrip_max_abs_error": state_err,
        "action_roundtrip_max_abs_error": action_err,
        "image_roundtrip_max_abs_error": image_err,
        "state_norm_min": float(torch.min(state_norm).item()),
        "state_norm_max": float(torch.max(state_norm).item()),
        "action_norm_mean_abs": float(torch.mean(torch.abs(action_norm)).item()),
        "action_norm_std_mean": float(torch.mean(torch.std(action_norm, dim=0)).item()),
        "normalizer_roundtrip_pass": bool(
            state_err < 1e-5 and action_err < 1e-5 and image_err < 1e-5
        ),
    }


def check_runner_zero_error(
    runner,
    dataset,
    n_obs_steps: int,
    n_action_steps: int,
    state_key: str,
    position_slice: slice,
) -> dict[str, Any]:
    if not all(
        hasattr(runner, name)
        for name in ("_compute_path_stats", "_reduce_metrics")
    ):
        return {
            "runner_zero_error_supported": False,
            "runner_zero_error_pass": False,
            "runner_zero_error_reason": "runner missing _compute_path_stats/_reduce_metrics",
        }

    val_dataset = dataset.get_validation_dataset()
    if len(val_dataset) == 0:
        return {
            "runner_zero_error_supported": True,
            "runner_zero_error_pass": False,
            "runner_zero_error_reason": "validation dataset is empty",
        }

    count = min(len(val_dataset), 4)
    samples = [val_dataset[i] for i in range(count)]
    state = torch.stack([s["obs"][state_key] for s in samples], dim=0).float()
    action = torch.stack([s["action"] for s in samples], dim=0).float()

    future_action = action[:, n_obs_steps : n_obs_steps + n_action_steps]
    start_pos = state[:, n_obs_steps - 1, position_slice]

    scale = float(getattr(runner, "action_scale_to_meter", 1.0))
    future_action_m = future_action * scale
    start_pos_m = start_pos * scale

    dist, gt_path_len = runner._compute_path_stats(
        pred_action_m=future_action_m,
        gt_action_m=future_action_m,
        start_pos_m=start_pos_m,
    )
    metrics = runner._reduce_metrics(
        pred_action_m=future_action_m,
        gt_action_m=future_action_m,
        dist=dist,
        gt_path_len=gt_path_len,
    )
    max_metric = max(abs(float(v)) for v in metrics.values())
    result = {f"runner_zero_error_{k}": float(v) for k, v in metrics.items()}
    result.update(
        {
            "runner_zero_error_supported": True,
            "runner_zero_error_max_metric": max_metric,
            "runner_zero_error_pass": bool(max_metric < 1e-8),
        }
    )
    return result


def aggregate_metric_dicts(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if len(metrics_list) == 0:
        return {}
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


def add_metric_delta(base: dict[str, float], other: dict[str, float], prefix: str) -> dict[str, float]:
    result = {}
    for key, value in other.items():
        result[f"{prefix}_{key}"] = float(value)
        if key in base:
            result[f"{prefix}_{key}_delta_vs_base"] = float(value - base[key])
    return result


def check_image_ablation(
    args: argparse.Namespace,
    dataset,
    runner,
    state_key: str,
) -> dict[str, Any]:
    if args.checkpoint is None:
        return {
            "image_ablation_supported": False,
            "image_ablation_reason": "checkpoint not provided",
        }

    workspace_cls = resolve_class(args.workspace_target)
    workspace = workspace_cls.create_from_checkpoint(
        str(Path(args.checkpoint).expanduser().resolve())
    )
    cfg = workspace.cfg

    use_ema = bool(getattr(cfg.training, "use_ema", False)) and not args.use_model
    policy = workspace.ema_model if use_ema else workspace.model
    policy_class = f"{policy.__class__.__module__}.{policy.__class__.__name__}"
    if not hasattr(policy, "obs_encoder"):
        return {
            "image_ablation_supported": False,
            "image_ablation_reason": (
                "checkpoint policy is not a hybrid-image policy "
                f"({policy_class})"
            ),
            "image_ablation_policy_class": policy_class,
        }
    device = torch.device(args.device)
    policy.eval()
    policy.to(device)

    val_dataset = dataset.get_validation_dataset()
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.ablation_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    episode_ends = np.asarray(dataset.replay_buffer.episode_ends[:], dtype=np.int64)
    val_sampler_indices = val_dataset.sampler.indices
    val_sample_episode_ids = np.searchsorted(
        episode_ends,
        val_sampler_indices[:, 0],
        side="right",
    )
    donor_by_episode: dict[int, torch.Tensor] = {}
    for sample_idx, episode_id in enumerate(val_sample_episode_ids.tolist()):
        if episode_id not in donor_by_episode:
            donor_by_episode[episode_id] = val_dataset[sample_idx]["obs"]["image"].clone()
    donor_episode_ids = sorted(donor_by_episode.keys())

    start, end = policy.get_action_window_indices()
    anchor_idx = policy.get_action_anchor_obs_index()
    base_metrics_list = []
    zero_metrics_list = []
    batch_shuffled_metrics_list = []
    temporal_shuffled_metrics_list = []
    cross_episode_metrics_list = []
    processed_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= args.ablation_batches:
                break
            batch = move_to_device(batch, device)
            obs_dict = batch["obs"]
            if "image" not in obs_dict:
                return {
                    "image_ablation_supported": False,
                    "image_ablation_reason": "obs.image not found in dataset sample",
                }

            gt_action = batch["action"]
            if getattr(policy, "pred_action_steps_only", False):
                gt_action = gt_action[:, start:end]
            gt_action_m = gt_action * runner.action_scale_to_meter
            start_pos_m = (
                batch["obs"][state_key][:, anchor_idx, runner.position_slice]
                * runner.action_scale_to_meter
            )

            def eval_variant(obs_variant: dict[str, torch.Tensor], seed_offset: int) -> dict[str, float]:
                generator = torch.Generator(device=device)
                generator.manual_seed(int(args.seed + batch_idx * 1000 + seed_offset))
                try:
                    result = policy.predict_action(obs_variant, generator=generator)
                except (AssertionError, KeyError, RuntimeError) as exc:
                    raise RuntimeError(
                        f"incompatible checkpoint policy for image ablation: {policy_class}"
                    ) from exc
                pred_action = result["action"] if getattr(policy, "pred_action_steps_only", False) else result["action_pred"][:, start:end]
                pred_action_m = pred_action * runner.action_scale_to_meter
                dist, gt_path_len = runner._compute_path_stats(
                    pred_action_m=pred_action_m,
                    gt_action_m=gt_action_m,
                    start_pos_m=start_pos_m,
                )
                return runner._reduce_metrics(
                    pred_action_m=pred_action_m,
                    gt_action_m=gt_action_m,
                    dist=dist,
                    gt_path_len=gt_path_len,
                )

            try:
                base_metrics = eval_variant(obs_dict, seed_offset=0)
            except RuntimeError as exc:
                return {
                    "image_ablation_supported": False,
                    "image_ablation_reason": str(exc),
                    "image_ablation_policy_class": policy_class,
                }
            base_metrics_list.append(base_metrics)

            zero_obs = {
                k: (torch.zeros_like(v) if k == "image" else v)
                for k, v in obs_dict.items()
            }
            zero_metrics_list.append(eval_variant(zero_obs, seed_offset=0))

            if obs_dict["image"].shape[0] > 1:
                perm = torch.roll(
                    torch.arange(obs_dict["image"].shape[0], device=device),
                    shifts=1,
                )
                shuffled_image = obs_dict["image"][perm]
            else:
                shuffled_image = obs_dict["image"]
            shuffled_obs = {
                k: (shuffled_image if k == "image" else v)
                for k, v in obs_dict.items()
            }
            batch_shuffled_metrics_list.append(eval_variant(shuffled_obs, seed_offset=0))

            temporal_image = torch.empty_like(obs_dict["image"])
            for sample_i in range(obs_dict["image"].shape[0]):
                time_perm = torch.randperm(
                    obs_dict["image"].shape[1],
                    device=device,
                    generator=torch.Generator(device=device).manual_seed(
                        int(args.seed + batch_idx * 1000 + sample_i + 17)
                    ),
                )
                temporal_image[sample_i] = obs_dict["image"][sample_i, time_perm]
            temporal_obs = {
                k: (temporal_image if k == "image" else v)
                for k, v in obs_dict.items()
            }
            temporal_shuffled_metrics_list.append(eval_variant(temporal_obs, seed_offset=0))

            current_start = batch_idx * args.ablation_batch_size
            current_batch_size = obs_dict["image"].shape[0]
            sample_indices = np.arange(current_start, current_start + current_batch_size)
            sample_episode_ids = val_sample_episode_ids[sample_indices]
            cross_episode_available = len(donor_episode_ids) > 1
            if cross_episode_available:
                cross_episode_image = torch.empty_like(obs_dict["image"])
                for sample_i, episode_id in enumerate(sample_episode_ids.tolist()):
                    donor_episode_id = next(
                        ep for ep in donor_episode_ids if ep != episode_id
                    )
                    donor_image = donor_by_episode[donor_episode_id].to(device=device)
                    cross_episode_image[sample_i] = donor_image
                cross_episode_obs = {
                    k: (cross_episode_image if k == "image" else v)
                    for k, v in obs_dict.items()
                }
                cross_episode_metrics_list.append(eval_variant(cross_episode_obs, seed_offset=0))
            processed_batches += 1

    base_metrics = aggregate_metric_dicts(base_metrics_list)
    zero_metrics = aggregate_metric_dicts(zero_metrics_list)
    batch_shuffled_metrics = aggregate_metric_dicts(batch_shuffled_metrics_list)
    temporal_shuffled_metrics = aggregate_metric_dicts(temporal_shuffled_metrics_list)
    cross_episode_metrics = aggregate_metric_dicts(cross_episode_metrics_list)

    summary = {
        "image_ablation_supported": True,
        "image_ablation_use_ema": use_ema,
        "image_ablation_device": str(device),
        "image_ablation_batches": int(processed_batches),
        "image_ablation_policy_class": policy_class,
    }
    summary.update({f"base_{k}": float(v) for k, v in base_metrics.items()})
    summary.update(add_metric_delta(base_metrics, zero_metrics, "zero_image"))
    summary.update(add_metric_delta(base_metrics, batch_shuffled_metrics, "batch_shuffled_image"))
    summary.update(add_metric_delta(base_metrics, temporal_shuffled_metrics, "temporal_shuffled_image"))
    if len(cross_episode_metrics) > 0:
        summary.update(add_metric_delta(base_metrics, cross_episode_metrics, "cross_episode_image"))
    return summary


def save_image_ablation_plot(summary: dict[str, Any], plot_path: str | None) -> str | None:
    ablation = summary.get("image_ablation_check", {})
    if not ablation.get("image_ablation_supported", False):
        return None

    if plot_path is None:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("action_mae", "Action MAE"),
        ("traj_ade_m", "ADE (m)"),
        ("traj_fde_m", "FDE (m)"),
        ("traj_path_error_pct", "Path Error (%)"),
    ]
    labels = [
        "base",
        "zero_image",
        "batch_shuffled_image",
        "temporal_shuffled_image",
    ]
    if "cross_episode_image_action_mae" in ablation:
        labels.append("cross_episode_image")
    title_map = {
        "base": "Base",
        "zero_image": "Zeroed",
        "batch_shuffled_image": "Batch-shuffled",
        "temporal_shuffled_image": "Time-shuffled",
        "cross_episode_image": "Cross-episode",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for ax, (metric_key, title) in zip(axes, metrics):
        values = [float(ablation[f"{label}_{metric_key}"]) for label in labels]
        colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"][: len(labels)]
        bars = ax.bar(range(len(labels)), values, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([title_map[k] for k in labels], rotation=12)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)

        base_value = values[0]
        for idx, (bar, value) in enumerate(zip(bars, values)):
            delta = value - base_value
            text = f"{value:.3f}"
            if idx > 0:
                text += f"\nΔ{delta:+.3f}"
            ax.annotate(
                text,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("Image Ablation vs Trajectory Metrics", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = Path(plot_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def main() -> None:
    args = parse_args()
    dataset_cls = resolve_class(args.dataset_target)
    runner_cls = resolve_class(args.runner_target)
    position_slice = parse_slice(args.position_slice)

    dataset = dataset_cls(
        zarr_path=args.zarr_path,
        horizon=args.horizon,
        pad_before=args.pad_before,
        pad_after=args.pad_after,
        seed=args.seed,
        val_ratio=args.val_ratio,
        max_train_episodes=args.max_train_episodes,
    )
    val_dataset = dataset.get_validation_dataset()

    zroot = zarr.open(str(Path(args.zarr_path).expanduser().resolve()), mode="r")
    episode_ends = np.asarray(dataset.replay_buffer.episode_ends[:], dtype=np.int64)
    episode_lengths = compute_episode_lengths(episode_ends)
    train_episode_count = int(np.sum(dataset.train_mask))
    val_episode_count = int(np.sum(~dataset.train_mask))

    sample = dataset[0]
    runner = runner_cls(
        output_dir=str(PROJECT_ROOT / "outputs" / "_verify_image_delta_pipeline"),
        dataset=dataset,
        batch_size=args.runner_batch_size,
        num_workers=0,
        pin_memory=False,
        device="cpu",
        max_batches=1,
        multi_sample_eval_samples=1,
        multi_sample_eval_max_batches=1,
    )

    summary: dict[str, Any] = {
        "zarr_path": str(Path(args.zarr_path).expanduser().resolve()),
        "dataset_target": args.dataset_target,
        "runner_target": args.runner_target,
        "horizon": int(args.horizon),
        "n_obs_steps": int(args.n_obs_steps),
        "n_action_steps": int(args.n_action_steps),
        "pad_before": int(args.pad_before),
        "pad_after": int(args.pad_after),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "dataset_summary": {
            "total_episodes": int(dataset.replay_buffer.n_episodes),
            "train_episodes": train_episode_count,
            "val_episodes": val_episode_count,
            "total_frames": int(dataset.replay_buffer["state"].shape[0]),
            "train_windows": int(len(dataset)),
            "val_windows": int(len(val_dataset)),
            "episode_len_min": int(episode_lengths.min()) if len(episode_lengths) else 0,
            "episode_len_mean": float(episode_lengths.mean()) if len(episode_lengths) else 0.0,
            "episode_len_max": int(episode_lengths.max()) if len(episode_lengths) else 0,
            "sample_obs_image_shape": list(sample["obs"]["image"].shape),
            "sample_obs_state_shape": list(sample["obs"][args.state_key].shape),
            "sample_action_shape": list(sample["action"].shape),
            "meters_per_unit": float(getattr(dataset, "meters_per_unit", 1.0)),
            "delta_scale_abs": zroot.attrs.get("delta_scale_abs", None),
            "frame_stride": zroot.attrs.get("frame_stride", None),
        },
    }

    summary["alignment_check"] = check_alignment(
        dataset=dataset,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        state_key=args.state_key,
        position_slice=position_slice,
        num_samples=args.num_alignment_samples,
    )
    summary["normalizer_check"] = check_normalizer_roundtrip(
        dataset=dataset,
        roundtrip_samples=args.roundtrip_samples,
    )
    summary["runner_zero_error_check"] = check_runner_zero_error(
        runner=runner,
        dataset=dataset,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        state_key=args.state_key,
        position_slice=position_slice,
    )
    summary["image_ablation_check"] = check_image_ablation(
        args=args,
        dataset=dataset,
        runner=runner,
        state_key=args.state_key,
    )
    saved_plot = save_image_ablation_plot(summary, args.ablation_plot_path)
    if saved_plot is not None:
        summary["image_ablation_plot_path"] = saved_plot

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
