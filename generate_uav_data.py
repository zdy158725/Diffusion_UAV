import math
import os

import numpy as np
import zarr


def generate_fake_uav_dataset(
    save_path,
    num_episodes,
    episode_len,
    pattern="structured",
    seed=42,
    noise_std=0.02,
    z_acc_noise_std=0.03,
    meters_per_unit=1000.0,
):
    """
    生成 3v3 无人机空战伪数据集
    观测: 6 架机 * 7 维 = 42 维 (pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, hp)
    目标: 3 架敌机 * 3 维 = 9 维, 每步相对位移 delta (dx, dy, dz)
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    obs_dim = 42
    action_dim = 9

    all_obs = []
    all_actions = []
    episode_ends = []
    current_idx = 0

    print(f"正在生成数据: {num_episodes} 条轨迹, 每条长度 {episode_len}...")
    rng = np.random.default_rng(seed)

    for _ in range(num_episodes):
        if pattern == "random":
            obs = rng.uniform(0, 1, (episode_len, obs_dim)).astype(np.float32)
            # random 模式下也使用较小的 delta 位移，避免量级和 structured 差异过大
            action = rng.uniform(-0.01, 0.01, (episode_len, action_dim)).astype(np.float32)
        else:
            obs = np.zeros((episode_len, obs_dim), dtype=np.float32)
            action = np.zeros((episode_len, action_dim), dtype=np.float32)

            n_own = 3
            n_enemy = 3
            dt = 0.05
            v_max_xy = 0.2
            v_max_z = 0.08
            kp = 2.0
            roll_scale = 1.0

            own_pos = rng.uniform(0.2, 0.8, (n_own, 3))
            enemy_pos = rng.uniform(0.2, 0.8, (n_enemy, 3))
            own_vel = rng.uniform(-0.05, 0.05, (n_own, 3))
            enemy_vel = rng.uniform(-0.05, 0.05, (n_enemy, 3))

            phase = rng.uniform(0.0, 2 * math.pi, n_enemy)
            omega = 0.08 + rng.uniform(0.0, 0.04)
            prev_enemy_pos = enemy_pos.copy()

            for t in range(episode_len):
                # 敌机机动: xy 为周期机动, z 引入噪声扰动避免 dz 恒为 0
                enemy_acc_t = 0.6 * np.sin(omega * t + phase)
                enemy_acc_n = 0.6 * np.cos(omega * t + phase)
                enemy_acc_z = 0.25 * np.sin(omega * t + phase + 0.7)
                enemy_acc_z += rng.normal(0.0, z_acc_noise_std, size=n_enemy)

                enemy_vel[:, 0] = np.clip(enemy_vel[:, 0] + enemy_acc_t * dt, -v_max_xy, v_max_xy)
                enemy_vel[:, 1] = np.clip(enemy_vel[:, 1] + enemy_acc_n * dt, -v_max_xy, v_max_xy)
                enemy_vel[:, 2] = np.clip(enemy_vel[:, 2] + enemy_acc_z * dt, -v_max_z, v_max_z)

                enemy_pos[:, 0:2] = np.clip(enemy_pos[:, 0:2] + enemy_vel[:, 0:2] * dt, 0.0, 1.0)
                enemy_pos[:, 2] = np.clip(enemy_pos[:, 2] + enemy_vel[:, 2] * dt, 0.0, 1.0)

                rel = enemy_pos[:, 0:2] - own_pos[:, 0:2]
                acc = np.clip(kp * rel, -1.0, 1.0)
                roll = np.arctan2(rel[:, 1], rel[:, 0]) / math.pi
                own_act = np.stack([acc[:, 0], acc[:, 1], roll * roll_scale], axis=1)
                if noise_std > 0:
                    own_act += rng.normal(0.0, noise_std, own_act.shape)
                own_act = np.clip(own_act, -1.0, 1.0)

                own_vel[:, 0] = np.clip(own_vel[:, 0] + own_act[:, 0] * dt, -v_max_xy, v_max_xy)
                own_vel[:, 1] = np.clip(own_vel[:, 1] + own_act[:, 1] * dt, -v_max_xy, v_max_xy)
                own_pos[:, 0:2] = np.clip(own_pos[:, 0:2] + own_vel[:, 0:2] * dt, 0.0, 1.0)

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

                obs[t] = np.array(obs_t, dtype=np.float32).reshape(-1)

                # 目标: 三架敌机每步相对位移 delta
                delta_enemy = enemy_pos - prev_enemy_pos
                action[t] = delta_enemy.reshape(-1).astype(np.float32)
                prev_enemy_pos = enemy_pos.copy()

        all_obs.append(obs)
        all_actions.append(action)
        current_idx += episode_len
        episode_ends.append(current_idx)

    full_obs = np.concatenate(all_obs, axis=0)
    full_actions = np.concatenate(all_actions, axis=0)
    episode_ends = np.array(episode_ends, dtype=np.int64)

    max_abs_delta = np.max(np.abs(full_actions), axis=0).astype(np.float32)
    max_abs_delta = np.maximum(max_abs_delta, 1e-6)

    store = zarr.DirectoryStore(save_path)
    root = zarr.group(store=store, overwrite=True)
    root.attrs["author"] = "UAV_Project"
    root.attrs["version"] = "1.0"
    root.attrs["description"] = "3v3 UAV Combat Delta-Position Data"
    root.attrs["action_target"] = "enemy_delta_position"
    root.attrs["delta_max_abs"] = max_abs_delta.tolist()
    root.attrs["meters_per_unit"] = float(meters_per_unit)

    data_group = root.create_group("data")
    data_group.create_dataset("uav_observations", data=full_obs, chunks=(1000, obs_dim))
    data_group.create_dataset("uav_actions", data=full_actions, chunks=(1000, action_dim))

    meta_group = root.create_group("meta")
    meta_group.create_dataset("episode_ends", data=episode_ends)

    print(f"✅ 数据集已成功保存至: {os.path.abspath(save_path)}")
    print(f"总计样本数: {full_obs.shape[0]} 行")
    print(f"delta 最大绝对值(前3维): {max_abs_delta[:3]}")
    print(f"绘图换算比例: 1.0 单位 = {meters_per_unit:.1f} 米")


if __name__ == "__main__":
    MY_SAVE_PATH = "/home/dfc/Documents/zdy_documents/data/uav_combat/test_data.zarr"
    MY_NUM_EPISODES = 20
    MY_EPISODE_LEN = 600
    MY_PATTERN = "structured"

    generate_fake_uav_dataset(
        save_path=MY_SAVE_PATH,
        num_episodes=MY_NUM_EPISODES,
        episode_len=MY_EPISODE_LEN,
        pattern=MY_PATTERN,
    )
