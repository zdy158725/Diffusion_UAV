import argparse
import csv
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import zarr


OBS_DIM = 9
ACTION_DIM = 3
EARTH_RADIUS_M = 6_371_000.0
FEET_TO_METERS = 0.3048
KNOT_TO_MPS = 0.514444
METAR_WIND_RE = re.compile(r"\b(?P<dir>\d{3})(?P<speed>\d{2,3})(?:G\d{2,3})?KT\b")
METAR_VRB_WIND_RE = re.compile(r"\bVRB(?P<speed>\d{2,3})(?:G\d{2,3})?KT\b")


@dataclass
class ADSBRecord:
    timestamp_sec: float
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    age_sec: float
    track_id: str
    tail: str
    wind_speed: float
    wind_dir_rad: Optional[float]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw ADS-B CSV data into diffusion-policy zarr format "
            "with obs=[x,y,z,vx,vy,vz,wind_speed,wind_dir_sin,wind_dir_cos] "
            "and action target configurable as delta_position or absolute_position."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/dfc/Documents/zdy_documents/diffusion_policy/uav_task_bundle/path_trajectory_raw",
        help="Root directory containing raw ADS-B CSV files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/dfc/Documents/zdy_documents/diffusion_policy/uav_task_bundle/data/adsb/adsb_7days2_fixedref_wind9d.zarr",
        help="Output zarr directory path.",
    )
    parser.add_argument(
        "--glob_pattern",
        type=str,
        default="**/raw_data/**/*.csv",
        help="Glob pattern relative to input_dir for finding raw CSV files.",
    )
    parser.add_argument(
        "--resample_dt_sec",
        type=float,
        default=1.0,
        help="Uniform resampling interval in seconds.",
    )
    parser.add_argument(
        "--max_gap_sec",
        type=float,
        default=30.0,
        help="Split one aircraft track into a new episode if time gap exceeds this threshold.",
    )
    parser.add_argument(
        "--min_episode_len",
        type=int,
        default=30,
        help="Minimum number of resampled frames required to keep one episode.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional cap on how many CSV files to convert.",
    )
    parser.add_argument(
        "--altitude_unit",
        type=str,
        default="ft",
        choices=["ft", "m"],
        help="Unit used by the raw Altitude column.",
    )
    parser.add_argument(
        "--max_step_speed_mps",
        type=float,
        default=200.0,
        help=(
            "After resampling, split one episode whenever the implied single-step "
            "speed exceeds this threshold in m/s. Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--delta_scale_quantile",
        type=float,
        default=0.999,
        help=(
            "Per-axis quantile of |delta_position| used for action normalization "
            "scale. 1.0 falls back to raw max."
        ),
    )
    parser.add_argument(
        "--action_target_type",
        type=str,
        default="delta_position",
        choices=["delta_position", "absolute_position"],
        help=(
            "Target stored in uav_actions. delta_position keeps the current setup; "
            "absolute_position stores per-step absolute xyz in the shared coordinate frame."
        ),
    )
    parser.add_argument(
        "--origin_mode",
        type=str,
        default="fixed_reference",
        choices=["fixed_reference", "episode_start"],
        help=(
            "Coordinate origin mode. fixed_reference keeps one shared xy origin "
            "for the whole converted dataset; episode_start preserves the old "
            "behavior of resetting each episode to its own start point."
        ),
    )
    parser.add_argument(
        "--reference_lat_deg",
        type=float,
        default=None,
        help=(
            "Optional fixed reference latitude in degrees. If omitted and "
            "origin_mode=fixed_reference, the first valid ADS-B point in the "
            "selected input set is used."
        ),
    )
    parser.add_argument(
        "--reference_lon_deg",
        type=float,
        default=None,
        help=(
            "Optional fixed reference longitude in degrees. If omitted and "
            "origin_mode=fixed_reference, the first valid ADS-B point in the "
            "selected input set is used."
        ),
    )
    return parser.parse_args()


def parse_datetime_to_seconds(date_text: str, time_text: str) -> float:
    dt_text = f"{date_text.strip()} {time_text.strip()}"
    for fmt in ("%m/%d/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(dt_text, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"Unsupported ADS-B datetime format: {dt_text}")


def safe_float(text: str) -> Optional[float]:
    value = text.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def altitude_to_meters(altitude_value: float, altitude_unit: str) -> float:
    if altitude_unit == "ft":
        return altitude_value * FEET_TO_METERS
    if altitude_unit == "m":
        return altitude_value
    raise ValueError(f"Unsupported altitude_unit={altitude_unit}")


def parse_metar_wind(metar_text: str) -> Tuple[float, Optional[float], bool]:
    text = metar_text.strip().upper()
    if not text:
        return 0.0, None, False

    vrb_match = METAR_VRB_WIND_RE.search(text)
    if vrb_match is not None:
        return 0.0, None, True

    match = METAR_WIND_RE.search(text)
    if match is None:
        return 0.0, None, False

    wind_dir_deg = float(match.group("dir"))
    wind_speed = float(match.group("speed")) * KNOT_TO_MPS
    return wind_speed, math.radians(wind_dir_deg), False


def find_csv_files(input_dir: Path, glob_pattern: str, max_files: Optional[int]) -> List[Path]:
    csv_files = sorted(p for p in input_dir.glob(glob_pattern) if p.is_file())
    if max_files is not None:
        csv_files = csv_files[: int(max_files)]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched {glob_pattern} under {input_dir}")
    return csv_files


def load_records_grouped_by_track(
    csv_path: Path,
    altitude_unit: str,
) -> Tuple[Dict[str, List[ADSBRecord]], int, int, int, int]:
    groups: Dict[str, List[ADSBRecord]] = defaultdict(list)
    skipped_rows = 0
    valid_rows = 0
    metar_parse_success = 0
    metar_vrb_count = 0

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"ID", "Time", "Date", "Altitude", "Lat", "Lon"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{csv_path} missing required columns {required}")

        for row in reader:
            lat = safe_float(row.get("Lat", ""))
            lon = safe_float(row.get("Lon", ""))
            alt = safe_float(row.get("Altitude", ""))
            if lat is None or lon is None or alt is None:
                skipped_rows += 1
                continue

            try:
                timestamp_sec = parse_datetime_to_seconds(
                    row.get("Date", ""),
                    row.get("Time", ""),
                )
            except ValueError:
                skipped_rows += 1
                continue

            track_id = row.get("ID", "").strip()
            if not track_id:
                skipped_rows += 1
                continue

            tail = row.get("Tail", "").strip()
            age_sec = safe_float(row.get("Age", "")) or math.inf
            valid_rows += 1
            wind_speed, wind_dir_rad, is_vrb = parse_metar_wind(row.get("Metar", ""))
            if wind_dir_rad is not None:
                metar_parse_success += 1
            elif is_vrb:
                metar_vrb_count += 1

            groups[track_id].append(
                ADSBRecord(
                    timestamp_sec=timestamp_sec,
                    latitude_deg=float(lat),
                    longitude_deg=float(lon),
                    altitude_m=float(altitude_to_meters(float(alt), altitude_unit)),
                    age_sec=float(age_sec),
                    track_id=track_id,
                    tail=tail,
                    wind_speed=float(wind_speed),
                    wind_dir_rad=wind_dir_rad,
                )
            )

    return groups, skipped_rows, valid_rows, metar_parse_success, metar_vrb_count


def deduplicate_records(records: Sequence[ADSBRecord]) -> List[ADSBRecord]:
    if not records:
        return []

    records_sorted = sorted(records, key=lambda r: (r.timestamp_sec, r.age_sec))
    deduped: List[ADSBRecord] = [records_sorted[0]]
    for record in records_sorted[1:]:
        if abs(record.timestamp_sec - deduped[-1].timestamp_sec) < 1e-6:
            continue
        deduped.append(record)
    return deduped


def split_records_by_gap(
    records: Sequence[ADSBRecord],
    max_gap_sec: float,
) -> List[List[ADSBRecord]]:
    if not records:
        return []

    episodes: List[List[ADSBRecord]] = []
    current: List[ADSBRecord] = [records[0]]
    for record in records[1:]:
        dt = record.timestamp_sec - current[-1].timestamp_sec
        if dt > max_gap_sec:
            episodes.append(current)
            current = [record]
        else:
            current.append(record)
    episodes.append(current)
    return episodes


def latlon_to_local_xy_m(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    origin_lat_deg: float,
    origin_lon_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    lat_rad = np.deg2rad(lat_deg.astype(np.float64, copy=False))
    lon_rad = np.deg2rad(lon_deg.astype(np.float64, copy=False))
    origin_lat_rad = math.radians(origin_lat_deg)
    origin_lon_rad = math.radians(origin_lon_deg)
    x = (lon_rad - origin_lon_rad) * math.cos(origin_lat_rad) * EARTH_RADIUS_M
    y = (lat_rad - origin_lat_rad) * EARTH_RADIUS_M
    return x.astype(np.float32), y.astype(np.float32)


def resolve_reference_origin(
    csv_files: Sequence[Path],
    origin_mode: str,
    reference_lat_deg: Optional[float],
    reference_lon_deg: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    if origin_mode == "episode_start":
        return None, None

    if (reference_lat_deg is None) ^ (reference_lon_deg is None):
        raise ValueError(
            "reference_lat_deg and reference_lon_deg should either both be set or both be omitted."
        )

    if reference_lat_deg is not None and reference_lon_deg is not None:
        return float(reference_lat_deg), float(reference_lon_deg)

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            for row in reader:
                lat = safe_float(row.get("Lat", ""))
                lon = safe_float(row.get("Lon", ""))
                if lat is None or lon is None:
                    continue
                return float(lat), float(lon)

    raise RuntimeError("Failed to infer a fixed reference origin from the selected CSV files.")


def resample_episode(
    records: Sequence[ADSBRecord],
    resample_dt_sec: float,
    origin_lat_deg: Optional[float] = None,
    origin_lon_deg: Optional[float] = None,
) -> Optional[np.ndarray]:
    if len(records) < 2:
        return None

    timestamps = np.asarray([r.timestamp_sec for r in records], dtype=np.float64)
    times_rel = timestamps - timestamps[0]
    if np.any(np.diff(times_rel) <= 0):
        return None

    duration = float(times_rel[-1])
    if duration < resample_dt_sec:
        return None

    resampled_t = np.arange(0.0, duration + 1e-9, resample_dt_sec, dtype=np.float64)
    if resampled_t.shape[0] < 2:
        return None

    lat = np.asarray([r.latitude_deg for r in records], dtype=np.float64)
    lon = np.asarray([r.longitude_deg for r in records], dtype=np.float64)
    alt = np.asarray([r.altitude_m for r in records], dtype=np.float64)
    wind_speed = np.asarray([r.wind_speed for r in records], dtype=np.float32)
    wind_dir_rad = np.asarray(
        [0.0 if r.wind_dir_rad is None else r.wind_dir_rad for r in records],
        dtype=np.float32,
    )
    wind_dir_sin = np.sin(wind_dir_rad).astype(np.float32)
    wind_dir_cos = np.cos(wind_dir_rad).astype(np.float32)

    if origin_lat_deg is None:
        origin_lat_deg = float(lat[0])
    if origin_lon_deg is None:
        origin_lon_deg = float(lon[0])

    x, y = latlon_to_local_xy_m(
        lat_deg=lat,
        lon_deg=lon,
        origin_lat_deg=float(origin_lat_deg),
        origin_lon_deg=float(origin_lon_deg),
    )
    z = alt.astype(np.float32)

    x_i = np.interp(resampled_t, times_rel, x.astype(np.float64))
    y_i = np.interp(resampled_t, times_rel, y.astype(np.float64))
    z_i = np.interp(resampled_t, times_rel, z.astype(np.float64))
    wind_lookup_idx = np.searchsorted(times_rel, resampled_t, side="right") - 1
    wind_lookup_idx = np.clip(wind_lookup_idx, 0, len(times_rel) - 1)
    wind_speed_i = wind_speed[wind_lookup_idx]
    wind_dir_sin_i = wind_dir_sin[wind_lookup_idx]
    wind_dir_cos_i = wind_dir_cos[wind_lookup_idx]

    pos = np.stack([x_i, y_i, z_i], axis=-1).astype(np.float32)
    vel = np.gradient(pos, float(resample_dt_sec), axis=0).astype(np.float32)
    wind = np.stack([wind_speed_i, wind_dir_sin_i, wind_dir_cos_i], axis=-1).astype(
        np.float32
    )
    obs = np.concatenate([pos, vel, wind], axis=-1).astype(np.float32)
    return obs


def compute_action_from_obs(
    obs: np.ndarray,
    action_target_type: str = "delta_position",
) -> np.ndarray:
    pos = obs[:, 0:3]
    if action_target_type == "delta_position":
        action = np.zeros_like(pos, dtype=np.float32)
        if len(pos) > 1:
            action[1:] = pos[1:] - pos[:-1]
        return action
    if action_target_type == "absolute_position":
        return pos.astype(np.float32, copy=True)
    raise ValueError(f"Unsupported action_target_type={action_target_type}")


def split_obs_by_speed(
    obs: np.ndarray,
    resample_dt_sec: float,
    max_step_speed_mps: float,
) -> Tuple[List[np.ndarray], int]:
    if len(obs) < 2 or max_step_speed_mps <= 0:
        return [obs], 0

    pos = obs[:, 0:3].astype(np.float32, copy=False)
    step_delta = pos[1:] - pos[:-1]
    step_speed = np.linalg.norm(step_delta, axis=-1) / float(resample_dt_sec)
    bad_edges = step_speed > float(max_step_speed_mps)
    bad_count = int(np.sum(bad_edges))
    if bad_count == 0:
        return [obs], 0

    split_points = (np.nonzero(bad_edges)[0] + 1).tolist()
    segments: List[np.ndarray] = []
    start = 0
    for end in split_points:
        segments.append(obs[start:end])
        start = end
    segments.append(obs[start:])
    return segments, bad_count


def prepare_obs_segment(
    obs: np.ndarray,
    resample_dt_sec: float,
    rebase_position: bool,
) -> np.ndarray:
    if len(obs) == 0:
        return obs.astype(np.float32, copy=False)

    pos = obs[:, 0:3].astype(np.float32, copy=True)
    if rebase_position:
        pos -= pos[0:1]
    if len(pos) > 1:
        vel = np.gradient(pos, float(resample_dt_sec), axis=0).astype(np.float32)
    else:
        vel = np.zeros_like(pos, dtype=np.float32)
    wind = obs[:, 6:9].astype(np.float32, copy=False)
    return np.concatenate([pos, vel, wind], axis=-1).astype(np.float32, copy=False)


def compute_action_scale_abs(
    full_actions: np.ndarray,
    delta_scale_quantile: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < float(delta_scale_quantile) <= 1.0):
        raise ValueError(
            f"delta_scale_quantile should be in (0, 1], got {delta_scale_quantile}"
        )

    abs_actions = np.abs(full_actions.astype(np.float32, copy=False))
    raw_max_abs = np.max(abs_actions, axis=0).astype(np.float32)
    raw_max_abs = np.maximum(raw_max_abs, 1e-6)

    nonzero_mask = np.linalg.norm(abs_actions, axis=-1) > 1e-6
    quantile_source = abs_actions[nonzero_mask] if np.any(nonzero_mask) else abs_actions

    if float(delta_scale_quantile) >= 1.0:
        scale_abs = raw_max_abs.copy()
    else:
        scale_abs = np.quantile(
            quantile_source,
            float(delta_scale_quantile),
            axis=0,
        ).astype(np.float32)
        scale_abs = np.minimum(scale_abs, raw_max_abs)
        scale_abs = np.maximum(scale_abs, 1e-6)

    return raw_max_abs, scale_abs


def build_episode_metadata(
    rel_csv_path: str,
    track_id: str,
    tail: str,
    segment_idx,
) -> str:
    tail_text = tail if tail else "unknown_tail"
    return f"{rel_csv_path}::track={track_id}::tail={tail_text}::segment={segment_idx}"


def convert_adsb_csv_dir_to_zarr(
    input_dir: str,
    output_path: str,
    glob_pattern: str = "**/raw_data/**/*.csv",
    resample_dt_sec: float = 1.0,
    max_gap_sec: float = 30.0,
    min_episode_len: int = 30,
    max_files: Optional[int] = None,
    altitude_unit: str = "ft",
    max_step_speed_mps: float = 200.0,
    delta_scale_quantile: float = 0.999,
    action_target_type: str = "delta_position",
    origin_mode: str = "fixed_reference",
    reference_lat_deg: Optional[float] = None,
    reference_lon_deg: Optional[float] = None,
) -> None:
    input_root = Path(input_dir).expanduser().resolve()
    output_root = Path(output_path).expanduser().resolve()
    os.makedirs(output_root.parent, exist_ok=True)

    csv_files = find_csv_files(
        input_dir=input_root,
        glob_pattern=glob_pattern,
        max_files=max_files,
    )
    origin_lat_deg, origin_lon_deg = resolve_reference_origin(
        csv_files=csv_files,
        origin_mode=origin_mode,
        reference_lat_deg=reference_lat_deg,
        reference_lon_deg=reference_lon_deg,
    )
    if origin_mode == "fixed_reference":
        print(
            "Using fixed reference origin: "
            f"lat={origin_lat_deg:.6f}, lon={origin_lon_deg:.6f}"
        )

    all_obs: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    episode_ends: List[int] = []
    episode_names: List[str] = []
    lengths: List[int] = []

    raw_rows_skipped = 0
    raw_track_count = 0
    duplicate_points_removed = 0
    short_segments_skipped = 0
    speed_anomaly_edges = 0
    speed_split_segments_created = 0
    valid_rows = 0
    metar_parse_success = 0
    metar_vrb_count = 0
    current_idx = 0

    for file_idx, csv_path in enumerate(csv_files, start=1):
        rel_csv_path = str(csv_path.relative_to(input_root))
        print(f"[{file_idx}/{len(csv_files)}] processing {rel_csv_path}")
        grouped_records, skipped_rows, file_valid_rows, file_metar_success, file_metar_vrb = load_records_grouped_by_track(
            csv_path=csv_path,
            altitude_unit=altitude_unit,
        )
        raw_rows_skipped += skipped_rows
        raw_track_count += len(grouped_records)
        valid_rows += file_valid_rows
        metar_parse_success += file_metar_success
        metar_vrb_count += file_metar_vrb

        for track_id, records in grouped_records.items():
            tail = records[0].tail if records else ""
            deduped_records = deduplicate_records(records)
            duplicate_points_removed += len(records) - len(deduped_records)

            segments = split_records_by_gap(
                records=deduped_records,
                max_gap_sec=max_gap_sec,
            )
            for segment_idx, segment_records in enumerate(segments):
                obs = resample_episode(
                    records=segment_records,
                    resample_dt_sec=resample_dt_sec,
                    origin_lat_deg=origin_lat_deg,
                    origin_lon_deg=origin_lon_deg,
                )
                if obs is None:
                    short_segments_skipped += 1
                    continue

                speed_split_obs_list, bad_edge_count = split_obs_by_speed(
                    obs=obs,
                    resample_dt_sec=resample_dt_sec,
                    max_step_speed_mps=max_step_speed_mps,
                )
                speed_anomaly_edges += bad_edge_count
                speed_split_segments_created += max(0, len(speed_split_obs_list) - 1)

                for split_idx, split_obs in enumerate(speed_split_obs_list):
                    split_obs = prepare_obs_segment(
                        obs=split_obs,
                        resample_dt_sec=resample_dt_sec,
                        rebase_position=(origin_mode == "episode_start"),
                    )
                    if len(split_obs) < int(min_episode_len):
                        short_segments_skipped += 1
                        continue

                    action = compute_action_from_obs(
                        split_obs,
                        action_target_type=action_target_type,
                    )
                    all_obs.append(split_obs)
                    all_actions.append(action)
                    current_idx += len(split_obs)
                    episode_ends.append(current_idx)
                    lengths.append(len(split_obs))

                    segment_name = segment_idx
                    if len(speed_split_obs_list) > 1:
                        segment_name = f"{segment_idx}.{split_idx}"
                    episode_names.append(
                        build_episode_metadata(
                            rel_csv_path=rel_csv_path,
                            track_id=track_id,
                            tail=tail,
                            segment_idx=segment_name,
                        )
                    )

    if not all_obs:
        raise RuntimeError("No usable ADS-B trajectories found after filtering.")

    full_obs = np.concatenate(all_obs, axis=0).astype(np.float32)
    full_actions = np.concatenate(all_actions, axis=0).astype(np.float32)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)

    raw_max_abs_action = np.max(np.abs(full_actions.astype(np.float32, copy=False)), axis=0)
    raw_max_abs_action = np.maximum(raw_max_abs_action, 1e-6)
    scale_abs_action = None
    if action_target_type == "delta_position":
        _, scale_abs_action = compute_action_scale_abs(
            full_actions=full_actions,
            delta_scale_quantile=delta_scale_quantile,
        )

    store = zarr.DirectoryStore(str(output_root))
    root = zarr.group(store=store, overwrite=True)
    root.attrs["author"] = "OpenAI Codex"
    root.attrs["version"] = "1.2-adsb-raw-lowdim-wind9d"
    root.attrs["description"] = (
        "Single-aircraft ADS-B trajectories converted to diffusion-policy zarr "
        "with fixed/global or episode-local xy coordinates, altitude in meters, "
        "and wind features parsed from METAR."
    )
    root.attrs["source_dataset"] = "ADS-B raw CSV"
    root.attrs["obs_description"] = [
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "wind_speed",
        "wind_dir_sin",
        "wind_dir_cos",
    ]
    root.attrs["obs_dim"] = OBS_DIM
    root.attrs["action_target"] = action_target_type
    root.attrs["action_max_abs"] = raw_max_abs_action.tolist()
    if action_target_type == "delta_position" and scale_abs_action is not None:
        root.attrs["delta_max_abs"] = raw_max_abs_action.tolist()
        root.attrs["delta_scale_abs"] = scale_abs_action.tolist()
        root.attrs["delta_scale_quantile"] = float(delta_scale_quantile)
    root.attrs["meters_per_unit"] = 1.0
    root.attrs["coordinate_frame"] = (
        "fixed_reference_xy_m + altitude_m"
        if origin_mode == "fixed_reference"
        else "episode_local_xy_m + altitude_m"
    )
    root.attrs["origin_mode"] = origin_mode
    if origin_lat_deg is not None and origin_lon_deg is not None:
        root.attrs["reference_lat_deg"] = float(origin_lat_deg)
        root.attrs["reference_lon_deg"] = float(origin_lon_deg)
    root.attrs["input_glob"] = glob_pattern
    root.attrs["csv_file_count"] = len(csv_files)
    root.attrs["raw_track_count"] = int(raw_track_count)
    root.attrs["trajectory_count"] = len(episode_ends)
    root.attrs["raw_rows_skipped"] = int(raw_rows_skipped)
    root.attrs["duplicate_points_removed"] = int(duplicate_points_removed)
    root.attrs["short_segments_skipped"] = int(short_segments_skipped)
    root.attrs["speed_anomaly_edges"] = int(speed_anomaly_edges)
    root.attrs["speed_split_segments_created"] = int(speed_split_segments_created)
    root.attrs["metar_wind_valid_rows"] = int(valid_rows)
    root.attrs["metar_wind_parse_success"] = int(metar_parse_success)
    root.attrs["metar_wind_vrb_count"] = int(metar_vrb_count)
    root.attrs["metar_wind_parse_failed"] = int(
        max(0, valid_rows - metar_parse_success - metar_vrb_count)
    )
    root.attrs["wind_speed_unit"] = "m/s"
    root.attrs["max_step_speed_mps"] = float(max_step_speed_mps)
    root.attrs["resample_dt_sec"] = float(resample_dt_sec)
    root.attrs["max_gap_sec"] = float(max_gap_sec)
    root.attrs["episode_len_min"] = int(np.min(lengths))
    root.attrs["episode_len_max"] = int(np.max(lengths))
    root.attrs["episode_len_mean"] = float(np.mean(lengths))

    data_group = root.create_group("data")
    data_group.create_dataset("uav_observations", data=full_obs, chunks=(1024, OBS_DIM))
    data_group.create_dataset("uav_actions", data=full_actions, chunks=(1024, ACTION_DIM))

    meta_group = root.create_group("meta")
    meta_group.create_dataset("episode_ends", data=episode_ends_arr)
    meta_group.create_dataset(
        "episode_names",
        data=np.asarray(episode_names, dtype=object),
        object_codec=zarr.codecs.VLenUTF8(),
    )

    print(f"Saved: {output_root}")
    print(f"CSV files processed: {len(csv_files)}")
    print(f"Raw track groups: {raw_track_count}")
    print(f"Episodes kept: {len(episode_ends)}")
    print(f"Total samples: {full_obs.shape[0]}")
    print(f"Observation shape: {full_obs.shape}")
    print(f"Action shape: {full_actions.shape}")
    print(
        "Episode length min/mean/max: "
        f"{int(np.min(lengths))}/{float(np.mean(lengths)):.2f}/{int(np.max(lengths))}"
    )
    print(f"Rows skipped during parsing: {raw_rows_skipped}")
    print(f"Duplicate timestamps removed: {duplicate_points_removed}")
    print(f"Short/invalid segments skipped: {short_segments_skipped}")
    print(f"Speed anomaly edges: {speed_anomaly_edges}")
    print(f"Speed split segments created: {speed_split_segments_created}")
    print(f"METAR wind valid rows: {valid_rows}")
    print(f"METAR wind parsed successfully: {metar_parse_success}")
    print(f"METAR wind VRB/unsupported count: {metar_vrb_count}")
    print(f"Action target type: {action_target_type}")
    print(f"Origin mode: {origin_mode}")
    if origin_lat_deg is not None and origin_lon_deg is not None:
        print(f"Reference origin lat/lon: {origin_lat_deg:.6f}, {origin_lon_deg:.6f}")
    print(f"Raw action max abs: {raw_max_abs_action.tolist()}")
    if action_target_type == "delta_position" and scale_abs_action is not None:
        print(
            f"Action scale abs (q={delta_scale_quantile}): "
            f"{scale_abs_action.tolist()}"
        )


if __name__ == "__main__":
    args = parse_args()
    convert_adsb_csv_dir_to_zarr(
        input_dir=args.input_dir,
        output_path=args.output_path,
        glob_pattern=args.glob_pattern,
        resample_dt_sec=args.resample_dt_sec,
        max_gap_sec=args.max_gap_sec,
        min_episode_len=args.min_episode_len,
        max_files=args.max_files,
        altitude_unit=args.altitude_unit,
        max_step_speed_mps=args.max_step_speed_mps,
        delta_scale_quantile=args.delta_scale_quantile,
        action_target_type=args.action_target_type,
        origin_mode=args.origin_mode,
        reference_lat_deg=args.reference_lat_deg,
        reference_lon_deg=args.reference_lon_deg,
    )
