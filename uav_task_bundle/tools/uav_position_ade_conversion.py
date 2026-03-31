#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_policy.workspace.train_diffusion_transformer_lowdim_workspace import (
    TrainDiffusionTransformerLowdimWorkspace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute raw and normalized absolute-position ADE from a saved UAV "
            "diffusion checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the diffusion checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for model inference, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Validation batch size for inference.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Validation dataloader workers.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional cap on validation batches for a faster estimate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for the inference random seed.",
    )
    parser.add_argument(
        "--use_model",
        action="store_true",
        help="Use the main model instead of EMA even if EMA is enabled in the checkpoint config.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save the summary as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    workspace = TrainDiffusionTransformerLowdimWorkspace.create_from_checkpoint(
        str(ckpt_path)
    )
    cfg = workspace.cfg

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = dataset.get_validation_dataset()
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=val_dataset.get_collate_fn(),
    )

    use_ema = bool(getattr(cfg.training, "use_ema", False)) and not args.use_model
    policy = workspace.ema_model if use_ema else workspace.model
    device = torch.device(args.device)
    policy.eval()
    policy.to(device)

    seed = int(args.seed if args.seed is not None else cfg.training.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    start, end = policy.get_action_window_indices()
    anchor_idx = policy.get_action_anchor_obs_index()
    position_slice = getattr(dataset, "position_slice", slice(0, 3))

    obs_norm = policy.normalizer["obs"].params_dict
    pos_scale = obs_norm["scale"][position_slice].detach().cpu().to(torch.float64)
    pos_offset = obs_norm["offset"][position_slice].detach().cpu().to(torch.float64)

    raw_dist_sum = 0.0
    norm_dist_sum = 0.0
    count = 0
    processed_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            batch = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
            obs = batch["obs"]
            gt_action = batch["action"][:, start:end]

            result = policy.predict_action({"obs": obs})
            if policy.pred_action_steps_only:
                pred_action = result["action"]
            else:
                pred_action = result["action_pred"][:, start:end]

            start_pos = obs[:, anchor_idx, position_slice]
            pred_xyz = start_pos[:, None, :] + torch.cumsum(pred_action, dim=1)
            gt_xyz = start_pos[:, None, :] + torch.cumsum(gt_action, dim=1)
            raw_dist = torch.linalg.norm(pred_xyz - gt_xyz, dim=-1)

            pred_xyz_norm = (
                pred_xyz.detach().cpu().to(torch.float64) * pos_scale.view(1, 1, 3)
                + pos_offset.view(1, 1, 3)
            )
            gt_xyz_norm = (
                gt_xyz.detach().cpu().to(torch.float64) * pos_scale.view(1, 1, 3)
                + pos_offset.view(1, 1, 3)
            )
            norm_dist = torch.linalg.norm(pred_xyz_norm - gt_xyz_norm, dim=-1)

            raw_dist_sum += raw_dist.sum().item()
            norm_dist_sum += norm_dist.sum().item()
            count += raw_dist.numel()
            processed_batches += 1

    if count == 0:
        raise RuntimeError("No validation samples were processed.")

    raw_abs_position_ade = raw_dist_sum / count
    normalized_abs_position_ade = norm_dist_sum / count
    ratio_norm_over_raw = normalized_abs_position_ade / raw_abs_position_ade

    summary = {
        "checkpoint": str(ckpt_path),
        "used_ema": use_ema,
        "device": str(device),
        "seed": seed,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "max_batches": None if args.max_batches is None else int(args.max_batches),
        "processed_batches": processed_batches,
        "processed_points": count,
        "n_obs_steps": int(policy.n_obs_steps),
        "n_action_steps": int(policy.n_action_steps),
        "horizon": int(policy.horizon),
        "raw_abs_position_ade": raw_abs_position_ade,
        "normalized_abs_position_ade": normalized_abs_position_ade,
        "ratio_norm_over_raw": ratio_norm_over_raw,
        "position_scale_xyz": pos_scale.tolist(),
        "position_offset_xyz": pos_offset.tolist(),
    }

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
