import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import zarr


IMAGE_SIZE = 96
STATE_DIM = 7
ACTION_DIM = 3
PIL_RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert UZH_FPV image + groundtruth sequences into diffusion-policy "
            "image zarr format with obs.image RGB and obs.state=[x,y,z,vx,vy,vz,dt]."
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
        default="/home/dfc/Documents/zdy_documents/diffusion_policy/uav_task_bundle/data/uzh_fpv/uzh_fpv_image_96.zarr",
        help="Output zarr directory path.",
    )
    parser.add_argument(
        "--sequence_glob",
        type=str,
        default="*/*",
        help="Glob pattern relative to input_dir for locating sequence folders.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=IMAGE_SIZE,
        help="Square image size after resize.",
    )
    parser.add_argument(
        "--min_episode_len",
        type=int,
        default=100,
        help="Minimum number of aligned image frames required to keep one episode.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help=(
            "Keep one frame every N aligned image frames. "
            "For example, 5 means using frames [0, 5, 10, ...]."
        ),
    )
    parser.add_argument(
        "--delta_scale_quantile",
        type=float,
        default=0.999,
        help="Per-axis quantile of |delta_position| used for action normalization scale.",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Optional cap on how many sequences to convert.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output zarr if it already exists.",
    )
    return parser.parse_args()


def find_sequence_dirs(input_dir: Path, sequence_glob: str, max_sequences: int = None) -> List[Path]:
    sequence_dirs = sorted(
        p for p in input_dir.glob(sequence_glob)
        if p.is_dir() and (p / "images.txt").is_file() and (p / "groundtruth.txt").is_file()
    )
    if max_sequences is not None:
        sequence_dirs = sequence_dirs[: int(max_sequences)]
    if not sequence_dirs:
        raise FileNotFoundError(
            f"No sequence directories with images.txt and groundtruth.txt matched {sequence_glob} under {input_dir}"
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


def load_image_index(sequence_dir: Path) -> Tuple[np.ndarray, List[Path]]:
    image_index_path = sequence_dir / "images.txt"
    timestamps = []
    image_paths = []
    with image_index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            timestamp = float(parts[1])
            rel_path = parts[2]
            image_path = sequence_dir / rel_path
            if not image_path.is_file():
                continue
            timestamps.append(timestamp)
            image_paths.append(image_path)

    if len(timestamps) < 2:
        raise ValueError(f"{image_index_path} has fewer than 2 valid image entries.")

    timestamps = np.asarray(timestamps, dtype=np.float64)
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    image_paths = [image_paths[i] for i in order.tolist()]

    keep = np.ones(len(timestamps), dtype=bool)
    keep[1:] = np.diff(timestamps) > 0
    timestamps = timestamps[keep]
    image_paths = [path for path, is_keep in zip(image_paths, keep.tolist()) if is_keep]

    if len(timestamps) < 2:
        raise ValueError(f"{image_index_path} has fewer than 2 unique image timestamps.")
    return timestamps, image_paths


def align_images_to_groundtruth(
    image_timestamps: np.ndarray,
    image_paths: List[Path],
    gt_timestamps: np.ndarray,
    gt_positions: np.ndarray,
) -> Tuple[np.ndarray, List[Path], np.ndarray]:
    overlap_mask = (
        (image_timestamps >= float(gt_timestamps[0]))
        & (image_timestamps <= float(gt_timestamps[-1]))
    )
    image_timestamps = image_timestamps[overlap_mask]
    image_paths = [path for path, keep in zip(image_paths, overlap_mask.tolist()) if keep]
    if len(image_timestamps) < 2:
        raise ValueError("Fewer than 2 image frames remain after overlap filtering.")

    aligned_positions = np.stack(
        [np.interp(image_timestamps, gt_timestamps, gt_positions[:, axis]) for axis in range(3)],
        axis=-1,
    )
    return image_timestamps, image_paths, aligned_positions


def apply_frame_stride(
    image_timestamps: np.ndarray,
    image_paths: List[Path],
    aligned_positions: np.ndarray,
    frame_stride: int,
) -> Tuple[np.ndarray, List[Path], np.ndarray]:
    stride = max(int(frame_stride), 1)
    if stride == 1:
        return image_timestamps, image_paths, aligned_positions

    keep_idx = np.arange(0, len(image_timestamps), stride, dtype=np.int64)
    image_timestamps = image_timestamps[keep_idx]
    aligned_positions = aligned_positions[keep_idx]
    image_paths = [image_paths[i] for i in keep_idx.tolist()]
    if len(image_timestamps) < 2:
        raise ValueError(
            f"Fewer than 2 frames remain after frame_stride={stride} downsampling."
        )
    return image_timestamps, image_paths, aligned_positions


def load_and_resize_images(image_paths: List[Path], image_size: int) -> np.ndarray:
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            resized = img.convert("RGB").resize((image_size, image_size), PIL_RESAMPLE_BILINEAR)
            images.append(np.asarray(resized, dtype=np.uint8))
    return np.stack(images, axis=0)


def build_state_and_action(image_timestamps: np.ndarray, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    local_pos = positions - positions[0]
    edge_order = 2 if local_pos.shape[0] >= 3 else 1
    velocity = np.stack(
        [np.gradient(local_pos[:, axis], image_timestamps, edge_order=edge_order) for axis in range(3)],
        axis=-1,
    )
    dt = np.empty((len(image_timestamps),), dtype=np.float64)
    dt[1:] = np.diff(image_timestamps)
    if len(image_timestamps) > 1:
        dt[0] = dt[1]
    else:
        dt[0] = 0.0

    state = np.concatenate(
        [
            local_pos.astype(np.float32, copy=False),
            velocity.astype(np.float32, copy=False),
            dt[:, None].astype(np.float32, copy=False),
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    action = np.zeros_like(local_pos, dtype=np.float32)
    action[1:] = local_pos[1:] - local_pos[:-1]
    return state, action.astype(np.float32, copy=False)


def convert_uzh_fpv_image_to_zarr(
    input_dir: Path,
    output_path: Path,
    sequence_glob: str,
    image_size: int,
    min_episode_len: int,
    frame_stride: int,
    delta_scale_quantile: float,
    max_sequences: int,
    overwrite: bool,
):
    sequence_dirs = find_sequence_dirs(input_dir, sequence_glob, max_sequences=max_sequences)

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_img = []
    all_state = []
    all_action = []
    episode_ends = []
    episode_names = []
    skipped_sequences = 0
    total_raw_images = 0
    total_overlap_images = 0
    total_subsampled_images = 0
    current_idx = 0
    raw_position_max_abs = np.zeros((3,), dtype=np.float64)

    for seq_idx, seq_dir in enumerate(sequence_dirs, start=1):
        print(f"[{seq_idx}/{len(sequence_dirs)}] processing {seq_dir.relative_to(input_dir)}")
        try:
            gt_timestamps, gt_positions = load_groundtruth(seq_dir)
            image_timestamps, image_paths = load_image_index(seq_dir)
            total_raw_images += len(image_timestamps)
            image_timestamps, image_paths, aligned_positions = align_images_to_groundtruth(
                image_timestamps=image_timestamps,
                image_paths=image_paths,
                gt_timestamps=gt_timestamps,
                gt_positions=gt_positions,
            )
            total_overlap_images += len(image_timestamps)
            image_timestamps, image_paths, aligned_positions = apply_frame_stride(
                image_timestamps=image_timestamps,
                image_paths=image_paths,
                aligned_positions=aligned_positions,
                frame_stride=frame_stride,
            )
            total_subsampled_images += len(image_timestamps)
            state, action = build_state_and_action(image_timestamps, aligned_positions)
            if state.shape[0] < int(min_episode_len):
                raise ValueError(
                    f"aligned image count {state.shape[0]} < min_episode_len {min_episode_len}"
                )
            image_arr = load_and_resize_images(image_paths, image_size=image_size)
        except Exception as exc:
            skipped_sequences += 1
            print(f"  skipped: {exc}")
            continue

        all_img.append(image_arr)
        all_state.append(state)
        all_action.append(action)
        raw_position_max_abs = np.maximum(raw_position_max_abs, np.max(np.abs(gt_positions), axis=0))
        current_idx += state.shape[0]
        episode_ends.append(current_idx)
        episode_names.append(str(seq_dir.relative_to(input_dir)))

    if not all_state:
        raise RuntimeError("No valid UZH_FPV image episodes were kept after conversion.")

    full_img = np.concatenate(all_img, axis=0).astype(np.uint8, copy=False)
    full_state = np.concatenate(all_state, axis=0).astype(np.float32, copy=False)
    full_action = np.concatenate(all_action, axis=0).astype(np.float32, copy=False)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)

    delta_max_abs = np.max(np.abs(full_action), axis=0)
    raw_position_max_abs = np.maximum(raw_position_max_abs.astype(np.float32), 1e-6)
    scale_q = float(delta_scale_quantile)
    if scale_q >= 1.0:
        delta_scale_abs = delta_max_abs.copy()
    else:
        delta_scale_abs = np.quantile(np.abs(full_action), scale_q, axis=0).astype(np.float32)
    delta_scale_abs = np.maximum(delta_scale_abs, 1e-6).astype(np.float32)

    root = zarr.open(str(output_path), mode="w")
    root.attrs["source_dataset"] = "UZH_FPV"
    root.attrs["obs_type"] = "image_hybrid"
    root.attrs["image_size"] = int(image_size)
    root.attrs["state_dim"] = STATE_DIM
    root.attrs["action_dim"] = ACTION_DIM
    root.attrs["state_description"] = ["x", "y", "z", "vx", "vy", "vz", "dt"]
    root.attrs["action_description"] = ["dx", "dy", "dz"]
    root.attrs["coordinate_frame"] = "episode_local_xyz_m"
    root.attrs["meters_per_unit"] = 1.0
    root.attrs["time_axis"] = "images_txt_overlap_groundtruth"
    root.attrs["frame_stride"] = int(max(frame_stride, 1))
    root.attrs["raw_position_max_abs"] = raw_position_max_abs.astype(np.float32).tolist()
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
    data_group.create_dataset("img", data=full_img, chunks=(32, image_size, image_size, 3))
    data_group.create_dataset("state", data=full_state, chunks=(1024, STATE_DIM))
    data_group.create_dataset("action", data=full_action, chunks=(1024, ACTION_DIM))
    meta_group.create_dataset("episode_ends", data=episode_ends_arr)
    meta_group.create_dataset(
        "episode_names",
        data=np.asarray(episode_names, dtype=object),
        object_codec=zarr.codecs.VLenUTF8(),
    )

    print(f"Saved: {output_path}")
    print(f"Sequence dirs processed: {len(sequence_dirs)}")
    print(f"Episodes kept: {len(episode_ends)}")
    print(f"Total frames: {full_state.shape[0]}")
    print(f"Image shape: {full_img.shape}")
    print(f"State shape: {full_state.shape}")
    print(f"Action shape: {full_action.shape}")
    print(
        "Episode length min/mean/max: "
        f"{root.attrs['episode_len_min']}/{root.attrs['episode_len_mean']:.2f}/{root.attrs['episode_len_max']}"
    )
    print(f"Skipped sequences: {skipped_sequences}")
    print(f"Raw indexed images: {total_raw_images}")
    print(f"Overlap images kept: {total_overlap_images}")
    print(f"Subsampled images kept (frame_stride={max(frame_stride, 1)}): {total_subsampled_images}")
    print(f"Raw position max abs: {raw_position_max_abs.astype(np.float32).tolist()}")
    print(f"Raw delta max abs: {delta_max_abs.astype(np.float32).tolist()}")
    print(f"Action scale abs (q={scale_q}): {delta_scale_abs.astype(np.float32).tolist()}")


if __name__ == "__main__":
    args = parse_args()
    convert_uzh_fpv_image_to_zarr(
        input_dir=Path(os.path.expanduser(args.input_dir)),
        output_path=Path(os.path.expanduser(args.output_path)),
        sequence_glob=args.sequence_glob,
        image_size=int(args.image_size),
        min_episode_len=int(args.min_episode_len),
        frame_stride=int(args.frame_stride),
        delta_scale_quantile=float(args.delta_scale_quantile),
        max_sequences=args.max_sequences,
        overwrite=bool(args.overwrite),
    )
