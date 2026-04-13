import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
CODE_TRAJ_DIR = ROOT_DIR / "code_traj"
if str(CODE_TRAJ_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_TRAJ_DIR))

from data import Environment, Node, Position, Scalar, Scene, Velocity  # noqa: E402


EXPECTED_COLUMNS = [
    "episode_id",
    "timestep",
    "drone_id",
    "team",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "hp",
]

NODE_TYPE_MAP = {
    "BLUE": "BLUE_UAV",
    "RED": "RED_UAV",
}

DEFAULT_STANDARDIZATION = {
    "position": {
        "x": {"mean": 0.0, "std": 100.0},
        "y": {"mean": 0.0, "std": 100.0},
        "z": {"mean": 0.0, "std": 50.0},
    },
    "velocity": {
        "x": {"mean": 0.0, "std": 20.0},
        "y": {"mean": 0.0, "std": 20.0},
        "z": {"mean": 0.0, "std": 10.0},
    },
    "hp": {
        "value": {"mean": 0.5, "std": 0.5},
    },
}


def build_standardization():
    return {
        "BLUE_UAV": json.loads(json.dumps(DEFAULT_STANDARDIZATION)),
        "RED_UAV": json.loads(json.dumps(DEFAULT_STANDARDIZATION)),
    }


def build_attention_radius(env, radius):
    attention_radius = {}
    for src in env.NodeType:
        for dst in env.NodeType:
            attention_radius[(src, dst)] = radius
    return attention_radius


def validate_dataframe(df):
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df["columns"]]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if not df["rows"]:
        raise ValueError("Input CSV is empty.")

    invalid_teams = sorted({row["team"].upper() for row in df["rows"]} - set(NODE_TYPE_MAP))
    if invalid_teams:
        raise ValueError(f"Unsupported team values: {invalid_teams}")


def build_scene(env, episode_id, episode_rows, dt, hp_scale, min_scene_length, stats):
    scene = Scene(timesteps=int(max(row["timestep"] for row in episode_rows)) + 1, dt=dt, name=str(episode_id))
    usable_nodes = 0

    rows_by_drone = defaultdict(list)
    for row in episode_rows:
        rows_by_drone[row["drone_id"]].append(row)

    for drone_id in sorted(rows_by_drone):
        node_rows = sorted(rows_by_drone[drone_id], key=lambda row: row["timestep"])

        if len(node_rows) < 2:
            stats["dropped_short_nodes"] += 1
            continue

        timesteps = np.array([row["timestep"] for row in node_rows], dtype=np.int64)
        diffs = np.diff(timesteps)
        if np.any(diffs <= 0):
            stats["dropped_non_monotonic_nodes"] += 1
            continue
        if np.any(diffs != 1):
            stats["dropped_internal_gap_nodes"] += 1
            continue

        team_name = str(node_rows[0]["team"]).upper()
        node = Node(type=getattr(env.NodeType, NODE_TYPE_MAP[team_name]))
        node.node_id = str(drone_id)
        node.first_timestep = int(timesteps[0])
        node.position = Position(
            np.array([row["x"] for row in node_rows], dtype=np.float32),
            np.array([row["y"] for row in node_rows], dtype=np.float32),
            np.array([row["z"] for row in node_rows], dtype=np.float32),
        )
        node.velocity = Velocity(
            np.array([row["vx"] for row in node_rows], dtype=np.float32),
            np.array([row["vy"] for row in node_rows], dtype=np.float32),
            np.array([row["vz"] for row in node_rows], dtype=np.float32),
        )
        node.hp = Scalar((np.array([row["hp"] for row in node_rows], dtype=np.float32) / hp_scale).astype(np.float32))
        node.description = team_name.lower()
        scene.nodes.append(node)
        usable_nodes += 1

    if usable_nodes == 0:
        stats["dropped_empty_scenes"] += 1
        return None

    if scene.timesteps < min_scene_length:
        stats["dropped_short_scenes"] += 1
        return None

    stats["kept_scenes"] += 1
    stats["kept_nodes"] += usable_nodes
    return scene


def build_environment(scenes, attention_radius):
    env = Environment(
        node_type_list=["BLUE_UAV", "RED_UAV"],
        standardization=build_standardization(),
        scenes=scenes,
        attention_radius=attention_radius,
    )
    return env


def split_episode_ids(episode_ids, train_ratio, seed):
    episode_ids = list(episode_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(episode_ids)

    if len(episode_ids) == 1:
        return episode_ids, []

    train_count = int(round(len(episode_ids) * train_ratio))
    train_count = max(1, min(len(episode_ids) - 1, train_count))
    return episode_ids[:train_count], episode_ids[train_count:]


def serialize_environment(output_path, env):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(env, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description="Convert 3v3 UAV CSV data into Trajectron++ pickles.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Input CSV path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "data" / "processed" / "uav_3v3",
        help="Directory where uav_train.pkl and uav_val.pkl will be written.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Episode-level train split ratio.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for episode split.")
    parser.add_argument("--dt", type=float, default=1.0, help="Scene timestep delta.")
    parser.add_argument("--hp-scale", type=float, default=1.0, help="HP normalization scale divisor.")
    parser.add_argument("--attention-radius", type=float, default=50.0, help="Uniform 2D attention radius.")
    parser.add_argument("--max-history", type=int, default=5, help="Expected maximum history length in training.")
    parser.add_argument("--prediction-horizon", type=int, default=6, help="Expected prediction horizon in training.")
    args = parser.parse_args()

    if args.hp_scale <= 0:
        raise ValueError("--hp-scale must be positive.")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be in (0, 1).")

    with args.input_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []

    df = {"columns": columns, "rows": rows}
    validate_dataframe(df)
    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                "episode_id": str(row["episode_id"]),
                "timestep": int(row["timestep"]),
                "drone_id": str(row["drone_id"]),
                "team": str(row["team"]).upper(),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "vx": float(row["vx"]),
                "vy": float(row["vy"]),
                "vz": float(row["vz"]),
                "hp": float(row["hp"]),
            }
        )

    min_scene_length = args.max_history + args.prediction_horizon + 1
    base_env = Environment(node_type_list=["BLUE_UAV", "RED_UAV"], standardization=build_standardization(), scenes=[], attention_radius={})
    attention_radius = build_attention_radius(base_env, args.attention_radius)

    all_stats = {
        "episodes_total": len({row["episode_id"] for row in normalized_rows}),
        "rows_total": len(normalized_rows),
        "dropped_short_nodes": 0,
        "dropped_non_monotonic_nodes": 0,
        "dropped_internal_gap_nodes": 0,
        "dropped_empty_scenes": 0,
        "dropped_short_scenes": 0,
        "kept_scenes": 0,
        "kept_nodes": 0,
    }

    episode_scenes = {}
    rows_by_episode = defaultdict(list)
    for row in normalized_rows:
        rows_by_episode[row["episode_id"]].append(row)

    for episode_id in sorted(rows_by_episode):
        scene = build_scene(
            base_env,
            episode_id,
            rows_by_episode[episode_id],
            args.dt,
            args.hp_scale,
            min_scene_length,
            all_stats,
        )
        if scene is not None:
            episode_scenes[episode_id] = scene

    if not episode_scenes:
        raise RuntimeError("No scenes survived preprocessing.")

    train_ids, val_ids = split_episode_ids(sorted(episode_scenes.keys()), args.train_ratio, args.seed)
    train_scenes = [episode_scenes[episode_id] for episode_id in train_ids]
    val_scenes = [episode_scenes[episode_id] for episode_id in val_ids]

    train_env = build_environment(train_scenes, attention_radius)
    val_env = build_environment(val_scenes, attention_radius)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    serialize_environment(args.output_dir / "uav_train.pkl", train_env)
    serialize_environment(args.output_dir / "uav_val.pkl", val_env)

    stats_path = args.output_dir / "uav_preprocess_stats.json"
    with stats_path.open("w") as f:
        json.dump(
            {
                **all_stats,
                "train_scene_count": len(train_scenes),
                "val_scene_count": len(val_scenes),
                "train_episode_ids": train_ids,
                "val_episode_ids": val_ids,
                "min_scene_length": min_scene_length,
                "dt": args.dt,
                "hp_scale": args.hp_scale,
                "attention_radius": args.attention_radius,
            },
            f,
            indent=2,
        )

    print(f"Wrote train pickle: {args.output_dir / 'uav_train.pkl'}")
    print(f"Wrote val pickle:   {args.output_dir / 'uav_val.pkl'}")
    print(f"Wrote stats json:   {stats_path}")
    print(f"Train scenes: {len(train_scenes)}, Val scenes: {len(val_scenes)}")


if __name__ == "__main__":
    main()
