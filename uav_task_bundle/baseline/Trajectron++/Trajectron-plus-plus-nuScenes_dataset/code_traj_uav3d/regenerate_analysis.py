import argparse
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch

import evaluation
from backfill_analysis_from_tensorboard import build_overview, write_analysis_outputs
from model.dyn_stg import SpatioTemporalGraphCVAEModel
from model.model_registrar import ModelRegistrar


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate analysis by re-evaluating saved UAV 3D checkpoints.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory with checkpoints/.")
    parser.add_argument("--device", type=str, default=None, help="Evaluation device override, e.g. cpu or cuda:0.")
    parser.add_argument("--limit-checkpoints", type=int, default=None, help="Only evaluate the first N checkpoints.")
    parser.add_argument("--checkpoint-iters", nargs="+", type=int, default=None, help="Specific checkpoint iteration ids to evaluate.")
    parser.add_argument("--snapshots-out", type=Path, default=None, help="Optional JSONL path to dump raw checkpoint analysis snapshots.")
    parser.add_argument("--snapshots-in", nargs="+", type=Path, default=None, help="Merge precomputed snapshot JSONL files instead of re-evaluating checkpoints.")
    parser.add_argument("--skip-analysis-write", action="store_true", help="Only write raw snapshots and skip the final analysis/ output.")
    parser.add_argument("--train-scene-limit", type=int, default=5, help="How many train scenes to use for train summaries.")
    parser.add_argument("--eval-scene-limit", type=int, default=None, help="Optional cap on evaluation scenes.")
    parser.add_argument("--eval-loss-scene-limit", type=int, default=25, help="How many eval scenes to use for eval_loss.")
    parser.add_argument("--enable-kde", action="store_true", help="Recompute KDE metrics. Disabled by default because it is very slow.")
    parser.add_argument("--enable-obs", action="store_true", help="Recompute obstacle violation metrics when map data is available.")
    return parser.parse_args()


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device):
    if requested_device is None:
        return torch.device("cpu")

    device_str = str(requested_device)
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"| requested {device_str} but CUDA is unavailable, falling back to cpu")
        return torch.device("cpu")
    return torch.device(device_str)


def resolve_path(code_dir, path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (code_dir / path).resolve()


def override_attention_radius(env, radius):
    if radius is None:
        return

    for edge_type in env.get_edge_types():
        env.attention_radius[edge_type] = radius


def configure_scene_graphs(scenes, graph_mode, graph_top_k):
    normalized_top_k = None if graph_top_k is None or graph_top_k <= 0 else graph_top_k
    for scene in scenes:
        scene.graph_mode = graph_mode
        scene.graph_top_k = normalized_top_k


def maybe_precompute_scene_graphs(env, hyperparams, run_config):
    override_attention_radius(env, run_config.get("graph_radius"))
    configure_scene_graphs(env.scenes, run_config.get("graph_mode", "radius"), run_config.get("graph_top_k", 0))

    if run_config.get("offline_scene_graph") != "yes":
        return

    for scene in env.scenes:
        scene.calculate_scene_graph(
            env.attention_radius,
            hyperparams["state"],
            hyperparams["edge_addition_filter"],
            hyperparams["edge_removal_filter"],
        )


def advance_annealers(stg, curr_iter):
    stg.set_curr_iter(curr_iter)
    stg.set_annealing_params()
    for _ in range(curr_iter + 1):
        stg.step_annealers()


def load_environment(data_path):
    with data_path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def checkpoint_iters(checkpoint_dir):
    iters = sorted(
        int(path.stem.split("-")[-1])
        for path in checkpoint_dir.glob("model_registrar-*.pt")
    )
    return iters


def write_snapshots_jsonl(output_path, snapshots):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for snapshot in snapshots:
            f.write(json.dumps(snapshot, sort_keys=True))
            f.write("\n")


def load_snapshots_jsonl(input_paths):
    snapshots = []
    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                snapshots.append(json.loads(line))

    snapshots.sort(key=lambda snapshot: (int(snapshot.get("source_step", snapshot["iteration"] - 1)), int(snapshot["iteration"])))
    return snapshots


def build_models(model_registrar, hyperparams, train_env, eval_env, device, curr_iter):
    train_stg = SpatioTemporalGraphCVAEModel(model_registrar, hyperparams, None, device)
    train_stg.set_scene_graph(train_env)
    advance_annealers(train_stg, curr_iter)

    eval_stg = SpatioTemporalGraphCVAEModel(model_registrar, hyperparams, None, device)
    eval_stg.set_scene_graph(eval_env)
    advance_annealers(eval_stg, curr_iter)

    return train_stg, eval_stg


def evaluate_scene_predictions(stg, env, scene, ph, max_hl, eval_batch_size, enable_kde=False, enable_obs=False):
    timesteps = scene.sample_timesteps(eval_batch_size)
    predictions = stg.predict(
        scene,
        timesteps,
        ph,
        num_samples_z=100,
        min_future_timesteps=ph,
        max_nodes=4 * eval_batch_size,
    )
    return evaluation.compute_batch_statistics(
        predictions,
        scene.dt,
        max_hl=max_hl,
        ph=ph,
        node_type_enum=env.NodeType,
        kde=enable_kde,
        obs=enable_obs,
        map=scene.map,
    )


def evaluate_scene_loss(eval_stg, scene, eval_batch_size):
    timesteps = scene.sample_timesteps(eval_batch_size)
    return eval_stg.eval_loss(scene, timesteps)


def evaluate_checkpoint(
    run_dir,
    curr_iter,
    hyperparams,
    train_env,
    eval_env,
    device,
    train_scenes,
    eval_scenes,
    eval_loss_scenes,
    seed,
    enable_kde,
    enable_obs,
):
    set_global_seed(seed)

    model_registrar = ModelRegistrar(str(run_dir), device)
    model_registrar.load_models(curr_iter)
    model_registrar.to(device)

    train_stg, eval_stg = build_models(model_registrar, hyperparams, train_env, eval_env, device, curr_iter)

    max_hl = hyperparams["maximum_history_length"]
    ph = hyperparams["prediction_horizon"]
    eval_batch_size = hyperparams["run"]["eval_batch_size"]

    with torch.no_grad():
        train_batch_errors = [
            evaluate_scene_predictions(
                train_stg,
                train_env,
                scene,
                ph,
                max_hl,
                eval_batch_size,
                enable_kde=enable_kde,
                enable_obs=enable_obs,
            )
            for scene in train_scenes
        ]
        eval_batch_errors = [
            evaluate_scene_predictions(
                eval_stg,
                eval_env,
                scene,
                ph,
                max_hl,
                eval_batch_size,
                enable_kde=enable_kde,
                enable_obs=enable_obs,
            )
            for scene in eval_scenes
        ]
        eval_loss = [
            evaluate_scene_loss(eval_stg, scene, eval_batch_size)
            for scene in eval_loss_scenes
        ]

    return {
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S", __import__("time").localtime()),
        "iteration": curr_iter + 1,
        "source_step": curr_iter,
        "source": "checkpoint_reevaluation",
        "summaries": {
            "train": evaluation.summarize_batch_errors(train_batch_errors),
            "eval": evaluation.summarize_batch_errors(eval_batch_errors),
            "eval_loss": evaluation.summarize_batch_errors(eval_loss),
        },
    }


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()

    if args.snapshots_in:
        snapshots = load_snapshots_jsonl([path.resolve() for path in args.snapshots_in])
        if not snapshots:
            raise SystemExit("no snapshots were loaded from --snapshots-in")
        overview = build_overview(snapshots, {})
        write_analysis_outputs(run_dir, snapshots, overview)
        print(f"| merged {len(snapshots)} snapshots into {run_dir / 'analysis'}")
        return

    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"missing config.json under {run_dir}")

    hyperparams = json.loads(config_path.read_text())
    run_config = hyperparams["run"]

    code_dir = Path(__file__).resolve().parent
    data_dir = resolve_path(code_dir, run_config["data_dir"])
    train_data_path = data_dir / run_config["train_data_dict"]
    eval_data_path = data_dir / run_config["eval_data_dict"]

    train_env = load_environment(train_data_path)
    eval_env = load_environment(eval_data_path)
    maybe_precompute_scene_graphs(train_env, hyperparams, run_config)
    maybe_precompute_scene_graphs(eval_env, hyperparams, run_config)

    train_scenes = train_env.scenes[: min(len(train_env.scenes), args.train_scene_limit)]
    eval_scenes = eval_env.scenes if args.eval_scene_limit is None else eval_env.scenes[: min(len(eval_env.scenes), args.eval_scene_limit)]
    eval_loss_scenes = eval_env.scenes[: min(len(eval_env.scenes), args.eval_loss_scene_limit)]

    ckpt_iters = checkpoint_iters(run_dir / "checkpoints")
    if args.checkpoint_iters is not None:
        requested = set(args.checkpoint_iters)
        ckpt_iters = [iter_num for iter_num in ckpt_iters if iter_num in requested]
    if args.limit_checkpoints is not None:
        ckpt_iters = ckpt_iters[: args.limit_checkpoints]
    if not ckpt_iters:
        raise SystemExit("no checkpoints selected for evaluation")

    device = resolve_device(args.device or run_config.get("eval_device"))
    print(f"| device: {device}")
    print(f"| checkpoints: {len(ckpt_iters)}")
    print(f"| train scenes: {len(train_scenes)}")
    print(f"| eval scenes: {len(eval_scenes)}")
    print(f"| eval loss scenes: {len(eval_loss_scenes)}")

    snapshots = []
    seed = int(run_config.get("seed", 123))
    for index, curr_iter in enumerate(ckpt_iters, start=1):
        print(f"| [{index}/{len(ckpt_iters)}] evaluating checkpoint iter={curr_iter} step={curr_iter + 1}")
        snapshot = evaluate_checkpoint(
            run_dir,
            curr_iter,
            hyperparams,
            train_env,
            eval_env,
            device,
            train_scenes,
            eval_scenes,
            eval_loss_scenes,
            seed,
            args.enable_kde,
            args.enable_obs,
        )
        snapshots.append(snapshot)

    if args.snapshots_out is not None:
        snapshots_out = args.snapshots_out.resolve()
        write_snapshots_jsonl(snapshots_out, snapshots)
        print(f"| wrote {len(snapshots)} raw snapshots to {snapshots_out}")

    if args.skip_analysis_write:
        return

    overview = build_overview(snapshots, {})
    write_analysis_outputs(run_dir, snapshots, overview)
    print(f"| wrote analysis to {run_dir / 'analysis'}")


if __name__ == "__main__":
    main()
