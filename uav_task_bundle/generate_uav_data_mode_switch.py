import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import zarr


# Mode ids
CRUISE = 0
ORBIT_LEFT = 1
ORBIT_RIGHT = 2
EVADE = 3
CLIMB = 4
DIVE = 5
RETURN_CENTER = 6
ZIGZAG = 7

MODE_NAMES = {
    CRUISE: "cruise",
    ORBIT_LEFT: "orbit_left",
    ORBIT_RIGHT: "orbit_right",
    EVADE: "evade",
    CLIMB: "climb",
    DIVE: "dive",
    RETURN_CENTER: "return_center",
    ZIGZAG: "zigzag",
}


@dataclass
class SimConfig:
    dt: float = 0.05
    v_max_xy: float = 0.24
    v_max_z: float = 0.10
    own_ctrl_kp: float = 2.0
    own_noise_std: float = 0.02
    enemy_acc_noise_std: float = 0.03
    z_acc_noise_std: float = 0.02
    min_mode_dwell: int = 24
    meters_per_unit: float = 1000.0


def _safe_norm(vec: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.linalg.norm(vec) + eps)


def _sample_mode(rng: np.random.Generator, weights: Dict[int, float]) -> int:
    modes = np.array(list(weights.keys()), dtype=np.int32)
    probs = np.array(list(weights.values()), dtype=np.float32)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(rng.choice(modes, p=probs))


def _select_mode(
    rng: np.random.Generator,
    cur_mode: int,
    hold_steps: int,
    enemy_pos: np.ndarray,
    own_pos: np.ndarray,
    cfg: SimConfig,
) -> int:
    if hold_steps < cfg.min_mode_dwell:
        return cur_mode

    rel = own_pos[:, :2] - enemy_pos[:2][None, :]
    dists = np.linalg.norm(rel, axis=1)
    nearest_dist = float(np.min(dists))
    boundary_margin = float(np.min(np.concatenate([enemy_pos, 1.0 - enemy_pos])))
    z = float(enemy_pos[2])

    # Hard safety rule: near map boundary, return to center.
    if boundary_margin < 0.05:
        return RETURN_CENTER

    # Threat-dependent switching probability (structured randomness).
    p_switch = 0.03
    if nearest_dist < 0.18:
        p_switch = 0.45
    elif nearest_dist < 0.30:
        p_switch = 0.20
    elif boundary_margin < 0.10:
        p_switch = 0.22

    if rng.uniform() > p_switch:
        return cur_mode

    if nearest_dist < 0.18:
        weights = {
            EVADE: 0.55,
            ORBIT_LEFT: 0.12,
            ORBIT_RIGHT: 0.12,
            ZIGZAG: 0.16,
            CRUISE: 0.05,
        }
    elif z < 0.25:
        weights = {CLIMB: 0.58, CRUISE: 0.22, ORBIT_LEFT: 0.10, ORBIT_RIGHT: 0.10}
    elif z > 0.85:
        weights = {DIVE: 0.58, CRUISE: 0.22, ORBIT_LEFT: 0.10, ORBIT_RIGHT: 0.10}
    elif boundary_margin < 0.12:
        weights = {RETURN_CENTER: 0.62, CRUISE: 0.18, ORBIT_LEFT: 0.10, ORBIT_RIGHT: 0.10}
    else:
        weights = {
            CRUISE: 0.36,
            ORBIT_LEFT: 0.14,
            ORBIT_RIGHT: 0.14,
            ZIGZAG: 0.18,
            CLIMB: 0.09,
            DIVE: 0.09,
        }
    return _sample_mode(rng, weights)


def _enemy_mode_accel(
    mode: int,
    t: int,
    enemy_pos: np.ndarray,
    enemy_vel: np.ndarray,
    own_pos: np.ndarray,
    center: np.ndarray,
    zigzag_phase: float,
) -> np.ndarray:
    nearest_idx = int(np.argmin(np.linalg.norm(own_pos[:, :2] - enemy_pos[:2][None, :], axis=1)))
    nearest_own = own_pos[nearest_idx]
    rel_xy = enemy_pos[:2] - nearest_own[:2]
    rel_xy_norm = _safe_norm(rel_xy)
    away_xy = rel_xy / rel_xy_norm
    tang_left = np.array([-away_xy[1], away_xy[0]], dtype=np.float32)
    tang_right = -tang_left

    to_center = center - enemy_pos
    to_center_norm = _safe_norm(to_center)
    to_center_dir = to_center / to_center_norm

    # Base damping
    acc = np.array([-0.25 * enemy_vel[0], -0.25 * enemy_vel[1], -0.18 * enemy_vel[2]], dtype=np.float32)

    if mode == CRUISE:
        osc = 0.10 * np.array(
            [np.sin(0.05 * t + zigzag_phase), np.cos(0.04 * t + 0.3 * zigzag_phase), 0.0],
            dtype=np.float32,
        )
        acc += osc
    elif mode == ORBIT_LEFT:
        acc[:2] += 0.78 * tang_left
    elif mode == ORBIT_RIGHT:
        acc[:2] += 0.78 * tang_right
    elif mode == EVADE:
        acc[:2] += 1.15 * away_xy
        acc[2] += 0.10 * np.sign(enemy_pos[2] - nearest_own[2])
    elif mode == CLIMB:
        acc[:2] += 0.25 * away_xy
        acc[2] += 0.52
    elif mode == DIVE:
        acc[:2] += 0.25 * away_xy
        acc[2] -= 0.52
    elif mode == RETURN_CENTER:
        acc += 0.95 * to_center_dir
    elif mode == ZIGZAG:
        lat = np.sin(0.23 * t + zigzag_phase)
        acc[:2] += 0.85 * lat * tang_left
        acc[2] += 0.20 * np.sin(0.10 * t + 0.7 * zigzag_phase)
    return acc


def generate_uav_dataset_mode_switch(
    save_path: str,
    num_episodes: int,
    episode_len: int,
    seed: int = 42,
    pattern: str = "structured",
    config: SimConfig = SimConfig(),
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    rng = np.random.default_rng(seed)

    obs_dim = 42
    action_dim = 3  # enemy0 delta position
    n_own = 3
    n_enemy = 3

    all_obs: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    episode_ends: List[int] = []
    current_idx = 0

    print(f"Generating mode-switch UAV data: episodes={num_episodes}, ep_len={episode_len}")

    for _ in range(num_episodes):
        obs = np.zeros((episode_len, obs_dim), dtype=np.float32)
        action = np.zeros((episode_len, action_dim), dtype=np.float32)

        own_pos = rng.uniform(0.2, 0.8, (n_own, 3)).astype(np.float32)
        enemy_pos = rng.uniform(0.2, 0.8, (n_enemy, 3)).astype(np.float32)
        own_vel = rng.uniform(-0.05, 0.05, (n_own, 3)).astype(np.float32)
        enemy_vel = rng.uniform(-0.05, 0.05, (n_enemy, 3)).astype(np.float32)

        enemy_mode = rng.integers(low=0, high=3, size=(n_enemy,), endpoint=False).astype(np.int32)
        enemy_mode_hold = np.zeros((n_enemy,), dtype=np.int32)
        enemy_zigzag_phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_enemy,)).astype(np.float32)

        prev_enemy_pos = enemy_pos.copy()
        center = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        for t in range(episode_len):
            if pattern == "random":
                enemy_vel += rng.normal(0.0, 0.02, size=enemy_vel.shape).astype(np.float32)
            else:
                # Mode switching by state of both sides.
                for k in range(n_enemy):
                    new_mode = _select_mode(
                        rng=rng,
                        cur_mode=int(enemy_mode[k]),
                        hold_steps=int(enemy_mode_hold[k]),
                        enemy_pos=enemy_pos[k],
                        own_pos=own_pos,
                        cfg=config,
                    )
                    if new_mode != enemy_mode[k]:
                        enemy_mode[k] = new_mode
                        enemy_mode_hold[k] = 0
                        enemy_zigzag_phase[k] = float(rng.uniform(0.0, 2.0 * np.pi))
                    else:
                        enemy_mode_hold[k] += 1

                enemy_acc = np.zeros_like(enemy_vel)
                for k in range(n_enemy):
                    enemy_acc[k] = _enemy_mode_accel(
                        mode=int(enemy_mode[k]),
                        t=t,
                        enemy_pos=enemy_pos[k],
                        enemy_vel=enemy_vel[k],
                        own_pos=own_pos,
                        center=center,
                        zigzag_phase=float(enemy_zigzag_phase[k]),
                    )

                enemy_acc += rng.normal(0.0, config.enemy_acc_noise_std, size=enemy_acc.shape).astype(np.float32)
                enemy_acc[:, 2] += rng.normal(0.0, config.z_acc_noise_std, size=(n_enemy,)).astype(np.float32)

                enemy_vel[:, 0] = np.clip(
                    enemy_vel[:, 0] + enemy_acc[:, 0] * config.dt, -config.v_max_xy, config.v_max_xy
                )
                enemy_vel[:, 1] = np.clip(
                    enemy_vel[:, 1] + enemy_acc[:, 1] * config.dt, -config.v_max_xy, config.v_max_xy
                )
                enemy_vel[:, 2] = np.clip(
                    enemy_vel[:, 2] + enemy_acc[:, 2] * config.dt, -config.v_max_z, config.v_max_z
                )

            enemy_pos = np.clip(enemy_pos + enemy_vel * config.dt, 0.0, 1.0)

            # Own side simple pursuit controller.
            rel = enemy_pos[:, :2] - own_pos[:, :2]
            own_acc_xy = np.clip(config.own_ctrl_kp * rel, -1.0, 1.0)
            own_roll = np.arctan2(rel[:, 1], rel[:, 0]) / np.pi
            own_act = np.stack([own_acc_xy[:, 0], own_acc_xy[:, 1], own_roll], axis=1).astype(np.float32)
            own_act += rng.normal(0.0, config.own_noise_std, size=own_act.shape).astype(np.float32)
            own_act = np.clip(own_act, -1.0, 1.0)

            own_vel[:, 0] = np.clip(
                own_vel[:, 0] + own_act[:, 0] * config.dt, -config.v_max_xy, config.v_max_xy
            )
            own_vel[:, 1] = np.clip(
                own_vel[:, 1] + own_act[:, 1] * config.dt, -config.v_max_xy, config.v_max_xy
            )
            own_vel[:, 2] = 0.90 * own_vel[:, 2] + rng.normal(0.0, 0.005, size=(n_own,)).astype(np.float32)
            own_vel[:, 2] = np.clip(own_vel[:, 2], -0.04, 0.04)
            own_pos = np.clip(own_pos + own_vel * config.dt, 0.0, 1.0)

            dist = np.linalg.norm(rel, axis=1)
            own_hp = np.clip(1.0 - dist / 1.5, 0.0, 1.0)
            enemy_hp = np.clip(1.0 - dist / 1.5, 0.0, 1.0)

            obs_t = []
            for k in range(n_own):
                obs_t.append(
                    [
                        own_pos[k, 0],
                        own_pos[k, 1],
                        own_pos[k, 2],
                        own_vel[k, 0],
                        own_vel[k, 1],
                        own_vel[k, 2],
                        own_hp[k],
                    ]
                )
            for k in range(n_enemy):
                obs_t.append(
                    [
                        enemy_pos[k, 0],
                        enemy_pos[k, 1],
                        enemy_pos[k, 2],
                        enemy_vel[k, 0],
                        enemy_vel[k, 1],
                        enemy_vel[k, 2],
                        enemy_hp[k],
                    ]
                )

            obs[t] = np.asarray(obs_t, dtype=np.float32).reshape(-1)

            delta_enemy0 = enemy_pos[0] - prev_enemy_pos[0]
            action[t] = delta_enemy0.astype(np.float32)
            prev_enemy_pos = enemy_pos.copy()

        all_obs.append(obs)
        all_actions.append(action)
        current_idx += episode_len
        episode_ends.append(current_idx)

    full_obs = np.concatenate(all_obs, axis=0)
    full_actions = np.concatenate(all_actions, axis=0)
    episode_ends = np.asarray(episode_ends, dtype=np.int64)

    max_abs_delta = np.max(np.abs(full_actions), axis=0).astype(np.float32)
    max_abs_delta = np.maximum(max_abs_delta, 1e-6)

    store = zarr.DirectoryStore(save_path)
    root = zarr.group(store=store, overwrite=True)
    root.attrs["author"] = "UAV_Project"
    root.attrs["version"] = "2.0-mode-switch"
    root.attrs["description"] = "UAV enemy0 delta with state-conditioned mode library"
    root.attrs["action_target"] = "enemy0_delta_position"
    root.attrs["delta_max_abs"] = max_abs_delta.tolist()
    root.attrs["meters_per_unit"] = float(config.meters_per_unit)
    root.attrs["mode_library"] = [MODE_NAMES[i] for i in sorted(MODE_NAMES.keys())]

    data_group = root.create_group("data")
    data_group.create_dataset("uav_observations", data=full_obs, chunks=(1000, obs_dim))
    data_group.create_dataset("uav_actions", data=full_actions, chunks=(1000, action_dim))

    meta_group = root.create_group("meta")
    meta_group.create_dataset("episode_ends", data=episode_ends)

    print(f"Saved: {os.path.abspath(save_path)}")
    print(f"Samples: {full_obs.shape[0]}")
    print(f"delta_max_abs: {max_abs_delta}")
    print(f"mode_library: {[MODE_NAMES[i] for i in sorted(MODE_NAMES.keys())]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate UAV data with mode library + state-based switching")
    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/dfc/Documents/zdy_documents/diffusion_policy/uav_task_bundle/data/uav_combat/test_data_mode_switch.zarr",
    )
    parser.add_argument("--num_episodes", type=int, default=32)
    parser.add_argument("--episode_len", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pattern", type=str, default="structured", choices=["structured", "random"])
    parser.add_argument("--meters_per_unit", type=float, default=1000.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = SimConfig(meters_per_unit=args.meters_per_unit)
    generate_uav_dataset_mode_switch(
        save_path=args.save_path,
        num_episodes=args.num_episodes,
        episode_len=args.episode_len,
        seed=args.seed,
        pattern=args.pattern,
        config=cfg,
    )
