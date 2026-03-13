import argparse
import json
import os

import numpy as np
import zarr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def episode_bounds(episode_ends: np.ndarray, episode_idx: int):
    if episode_idx < 0 or episode_idx >= len(episode_ends):
        raise IndexError(f"episode_idx={episode_idx} out of range [0, {len(episode_ends)-1}]")
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end = int(episode_ends[episode_idx])
    return start, end


def hp_indices():
    own_hp = [6 + 7 * k for k in range(3)]
    enemy_hp = [27 + 7 * k for k in range(3)]
    return own_hp, enemy_hp


def enemy_pos_slice(enemy_idx: int):
    base = 21 + 7 * enemy_idx
    return slice(base, base + 3)


def own_pos_slice(own_idx: int):
    base = 7 * own_idx
    return slice(base, base + 3)


def save_position_delta_plot(ep_obs_m, ep_action_m, out_path):
    t = np.arange(ep_obs_m.shape[0])
    enemy0_pos = ep_obs_m[:, enemy_pos_slice(0)]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    # Absolute position of enemy0
    for d, name in enumerate(["x", "y", "z"]):
        axes[0].plot(t, enemy0_pos[:, d], label=name)
    axes[0].set_title("Enemy0 absolute position")
    axes[0].set_ylabel("m")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Saved action (delta)
    for d, name in enumerate(["dx", "dy", "dz"]):
        axes[1].plot(t, ep_action_m[:, d], label=name)
    axes[1].set_title("Saved action (enemy0 delta position)")
    axes[1].set_ylabel("m/step")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Compare saved delta vs recomputed delta from obs
    if len(t) > 1:
        recomputed = enemy0_pos[1:] - enemy0_pos[:-1]
        t1 = np.arange(1, len(t))
        for d, name in enumerate(["dx", "dy", "dz"]):
            axes[2].plot(t1, recomputed[:, d], linestyle="--", label=f"recomputed_{name}")
            axes[2].plot(t1, ep_action_m[1:, d], alpha=0.7, label=f"saved_{name}")
    axes[2].set_title("Delta consistency: recomputed vs saved")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("m/step")
    axes[2].legend(ncol=2, fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_hp_plot(ep_obs, out_path):
    t = np.arange(ep_obs.shape[0])
    own_hp_idx, enemy_hp_idx = hp_indices()

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for k in range(3):
        own_hp = ep_obs[:, own_hp_idx[k]]
        enemy_hp = ep_obs[:, enemy_hp_idx[k]]
        axes[0].plot(t, own_hp, label=f"own{k}_hp")
        axes[0].plot(t, enemy_hp, linestyle="--", label=f"enemy{k}_hp")
    axes[0].set_title("Stored HP curves")
    axes[0].set_ylabel("[0, 1]")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Recompute HP from XY distance and compare with stored values.
    for k in range(3):
        own_xy = ep_obs[:, own_pos_slice(k).start:own_pos_slice(k).start + 2]
        enemy_xy = ep_obs[:, enemy_pos_slice(k).start:enemy_pos_slice(k).start + 2]
        dist = np.linalg.norm(enemy_xy - own_xy, axis=1)
        hp_formula = np.clip(1.0 - dist / 1.5, 0.0, 1.0)
        own_err = np.abs(ep_obs[:, own_hp_idx[k]] - hp_formula)
        enemy_err = np.abs(ep_obs[:, enemy_hp_idx[k]] - hp_formula)
        axes[1].plot(t, own_err, label=f"own{k}_hp_err")
        axes[1].plot(t, enemy_err, linestyle="--", label=f"enemy{k}_hp_err")
    axes[1].set_title("HP consistency error |stored - formula|")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("abs error")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_trajectory3d_plot(ep_obs_m, out_path):
    enemy0_pos = ep_obs_m[:, enemy_pos_slice(0)]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(enemy0_pos[:, 0], enemy0_pos[:, 1], enemy0_pos[:, 2], label="enemy0")
    ax.scatter(enemy0_pos[0, 0], enemy0_pos[0, 1], enemy0_pos[0, 2], c="k", s=24, label="start")
    ax.scatter(enemy0_pos[-1, 0], enemy0_pos[-1, 1], enemy0_pos[-1, 2], c="r", s=24, label="end")
    ax.set_title("Enemy0 3D trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_summary(root, obs, action, episode_ends, episode_idx):
    start, end = episode_bounds(episode_ends, episode_idx)
    ep_obs = obs[start:end]
    ep_action = action[start:end]

    meters_per_unit = float(root.attrs.get("meters_per_unit", 1.0))
    ep_obs_m = ep_obs.copy()
    ep_action_m = ep_action * meters_per_unit
    for k in range(3):
        ep_obs_m[:, own_pos_slice(k)] *= meters_per_unit
        ep_obs_m[:, enemy_pos_slice(k)] *= meters_per_unit

    # Delta consistency on episode
    enemy0_pos_m = ep_obs_m[:, enemy_pos_slice(0)]
    if len(ep_obs_m) > 1:
        recomputed = enemy0_pos_m[1:] - enemy0_pos_m[:-1]
        saved = ep_action_m[1:]
        delta_err = saved - recomputed
        delta_err_abs_max = float(np.max(np.abs(delta_err)))
        delta_err_l2_mean = float(np.mean(np.linalg.norm(delta_err, axis=1)))
    else:
        delta_err_abs_max = 0.0
        delta_err_l2_mean = 0.0

    # HP consistency across all own/enemy pairs
    own_hp_idx, enemy_hp_idx = hp_indices()
    hp_err_max = 0.0
    hp_err_mean_all = []
    for k in range(3):
        own_xy = ep_obs[:, own_pos_slice(k).start:own_pos_slice(k).start + 2]
        enemy_xy = ep_obs[:, enemy_pos_slice(k).start:enemy_pos_slice(k).start + 2]
        dist = np.linalg.norm(enemy_xy - own_xy, axis=1)
        hp_formula = np.clip(1.0 - dist / 1.5, 0.0, 1.0)
        own_err = np.abs(ep_obs[:, own_hp_idx[k]] - hp_formula)
        enemy_err = np.abs(ep_obs[:, enemy_hp_idx[k]] - hp_formula)
        hp_err_max = max(hp_err_max, float(np.max(own_err)), float(np.max(enemy_err)))
        hp_err_mean_all.extend([float(np.mean(own_err)), float(np.mean(enemy_err))])

    # Attribute consistency on whole dataset
    delta_max_abs_attr = root.attrs.get("delta_max_abs", None)
    action_max_abs = np.max(np.abs(action), axis=0)
    if delta_max_abs_attr is not None:
        delta_max_abs_attr = np.asarray(delta_max_abs_attr, dtype=np.float32)
        delta_max_abs_err = float(np.max(np.abs(delta_max_abs_attr - action_max_abs)))
    else:
        delta_max_abs_err = None

    summary = {
        "episode_idx": int(episode_idx),
        "episode_start": int(start),
        "episode_end": int(end),
        "episode_len": int(end - start),
        "obs_shape_full": list(obs.shape),
        "action_shape_full": list(action.shape),
        "obs_dim_expected": 42,
        "action_dim_expected": 3,
        "meters_per_unit": meters_per_unit,
        "delta_err_abs_max_m": delta_err_abs_max,
        "delta_err_l2_mean_m": delta_err_l2_mean,
        "hp_err_abs_max": hp_err_max,
        "hp_err_abs_mean": float(np.mean(hp_err_mean_all)),
        "delta_max_abs_attr_vs_data_err": delta_max_abs_err,
    }
    return summary, ep_obs, ep_action, ep_obs_m, ep_action_m


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize and validate UAV zarr dataset.")
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="uav_task_bundle/data/uav_combat/test_data_mode_switch.zarr",
        help="Path to zarr dataset directory.",
    )
    parser.add_argument("--episode_idx", type=int, default=0, help="Episode index to visualize.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="uav_task_bundle/data_checks",
        help="Directory to save plots and summary.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    root = zarr.open(args.zarr_path, mode="r")
    obs = root["data"]["uav_observations"][:]
    action = root["data"]["uav_actions"][:]
    episode_ends = root["meta"]["episode_ends"][:]

    summary, ep_obs, ep_action, ep_obs_m, ep_action_m = compute_summary(
        root=root,
        obs=obs,
        action=action,
        episode_ends=episode_ends,
        episode_idx=args.episode_idx,
    )

    stem = f"episode{args.episode_idx:03d}"
    pos_plot = os.path.join(args.out_dir, f"{stem}_position_delta.png")
    hp_plot = os.path.join(args.out_dir, f"{stem}_hp.png")
    traj3d_plot = os.path.join(args.out_dir, f"{stem}_traj3d.png")
    summary_path = os.path.join(args.out_dir, f"{stem}_summary.json")

    save_position_delta_plot(ep_obs_m=ep_obs_m, ep_action_m=ep_action_m, out_path=pos_plot)
    save_hp_plot(ep_obs=ep_obs, out_path=hp_plot)
    save_trajectory3d_plot(ep_obs_m=ep_obs_m, out_path=traj3d_plot)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] zarr: {os.path.abspath(args.zarr_path)}")
    print(f"[OK] output dir: {os.path.abspath(args.out_dir)}")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] plot: {pos_plot}")
    print(f"[OK] plot: {hp_plot}")
    print(f"[OK] plot: {traj3d_plot}")
    print("[Summary]")
    for k, v in summary.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
