import numpy as np
import zarr
import os
import math

def generate_fake_uav_dataset(
    save_path,
    num_episodes,
    episode_len,
    pattern="structured",
    seed=42,
    noise_std=0.02,
):
    """
    生成3v3无人机空战伪数据集
    观测: 6架机 * 7维 = 42维 (pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, hp)
    目标: 敌方3架机 * 3维 = 9维 (pos_x, pos_y, pos_z)
    """
    # 确保父目录存在
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    obs_dim = 42    # 6架无人机 * 7维属性
    action_dim = 9  # 3架敌方无人机 * 3维位置

    all_obs = []
    all_actions = []
    episode_ends = []
    current_idx = 0

    print(f"正在生成数据: {num_episodes} 条轨迹, 每条长度 {episode_len}...")

    rng = np.random.default_rng(seed)

    for i in range(num_episodes):
        if pattern == "random":
            # 纯随机
            obs = rng.uniform(0, 1, (episode_len, obs_dim)).astype(np.float32)
            action = rng.uniform(-1, 1, (episode_len, action_dim)).astype(np.float32)
        else:
            # 有规律：我方追踪敌方的简单策略 + 平滑动力学
            obs = np.zeros((episode_len, obs_dim), dtype=np.float32)
            action = np.zeros((episode_len, action_dim), dtype=np.float32)

            n_own = 3
            n_enemy = 3
            dt = 0.05
            v_max = 0.2
            kp = 2.0
            roll_scale = 1.0

            # 初始化位置/速度
            own_pos = rng.uniform(0.2, 0.8, (n_own, 3))
            enemy_pos = rng.uniform(0.2, 0.8, (n_enemy, 3))
            own_vel = rng.uniform(-0.05, 0.05, (n_own, 3))
            enemy_vel = rng.uniform(-0.05, 0.05, (n_enemy, 3))

            # 敌方轨迹相位
            phase = rng.uniform(0.0, 2 * math.pi, n_enemy)
            omega = 0.08 + rng.uniform(0.0, 0.04)

            for t in range(episode_len):
            # 敌方机动（用于生成速度变化）
                enemy_acc_t = 0.6 * np.sin(omega * t + phase)
                enemy_acc_n = 0.6 * np.cos(omega * t + phase)

            # 更新敌方状态
                enemy_vel[:, 0] = np.clip(enemy_vel[:, 0] + enemy_acc_t * dt, -v_max, v_max)
                enemy_vel[:, 1] = np.clip(enemy_vel[:, 1] + enemy_acc_n * dt, -v_max, v_max)
                enemy_pos[:, 0:2] = np.clip(enemy_pos[:, 0:2] + enemy_vel[:, 0:2] * dt, 0.0, 1.0)

                # 我方动作：追踪对应敌机
                rel = enemy_pos[:, 0:2] - own_pos[:, 0:2]
                acc = np.clip(kp * rel, -1.0, 1.0)
                roll = np.arctan2(rel[:, 1], rel[:, 0]) / math.pi
                own_act = np.stack([acc[:, 0], acc[:, 1], roll * roll_scale], axis=1)
                if noise_std > 0:
                    own_act += rng.normal(0.0, noise_std, own_act.shape)
                own_act = np.clip(own_act, -1.0, 1.0)

                # 更新我方状态
                own_vel[:, 0] = np.clip(own_vel[:, 0] + own_act[:, 0] * dt, -v_max, v_max)
                own_vel[:, 1] = np.clip(own_vel[:, 1] + own_act[:, 1] * dt, -v_max, v_max)
                own_pos[:, 0:2] = np.clip(own_pos[:, 0:2] + own_vel[:, 0:2] * dt, 0.0, 1.0)

                # 血量：与目标距离越近越低（示意）
                dist = np.linalg.norm(rel, axis=1)
                own_hp = np.clip(1.0 - dist / 1.5, 0.0, 1.0)
                enemy_hp = np.clip(1.0 - dist / 1.5, 0.0, 1.0)

                # 组织 obs: 先我方后敌方
                obs_t = []
                for k in range(n_own):
                    obs_t.append([
                        own_pos[k, 0], own_pos[k, 1], own_pos[k, 2],
                        own_vel[k, 0], own_vel[k, 1], own_vel[k, 2],
                        own_hp[k]
                    ])
                for k in range(n_enemy):
                    obs_t.append([
                        enemy_pos[k, 0], enemy_pos[k, 1], enemy_pos[k, 2],
                        enemy_vel[k, 0], enemy_vel[k, 1], enemy_vel[k, 2],
                        enemy_hp[k]
                    ])

                obs[t] = np.array(obs_t, dtype=np.float32).reshape(-1)
                # 目标: 敌方位置轨迹
                action[t] = enemy_pos.reshape(-1).astype(np.float32)

        all_obs.append(obs)
        all_actions.append(action)
        
        current_idx += episode_len
        episode_ends.append(current_idx)

    # 合并所有轨迹
    full_obs = np.concatenate(all_obs, axis=0)
    full_actions = np.concatenate(all_actions, axis=0)
    episode_ends = np.array(episode_ends, dtype=np.int64)

    # 写入 Zarr 文件
    # 模式 'w' 会覆盖已存在的同名文件夹
    store = zarr.DirectoryStore(save_path)
    root = zarr.group(store=store, overwrite=True)
    root.attrs['author'] = 'UAV_Project'
    root.attrs['version'] = '1.0'
    # 官方通常会在这里存一些关于 step 的元数据
    root.attrs['description'] = '3v3 UAV Combat Trajectory Data'
    
    data_group = root.create_group('data')
    # 使用 chunks 优化读取速度
    data_group.create_dataset('uav_observations', data=full_obs, chunks=(1000, obs_dim))
    data_group.create_dataset('uav_actions', data=full_actions, chunks=(1000, action_dim))
    
    meta_group = root.create_group('meta')
    meta_group.create_dataset('episode_ends', data=episode_ends)

    print(f"✅ 数据集已成功保存至: {os.path.abspath(save_path)}")
    print(f"总计样本数: {full_obs.shape[0]} 行")

if __name__ == "__main__":
    # --- 在这里自定义参数 ---
    MY_SAVE_PATH = "/home/dfc/Documents/zdy_documents/diffusion_policy/uav_task_bundle/data/uav_combat/test_data.zarr" # 存储路径
    MY_NUM_EPISODES = 20                              # 轨迹数量
    MY_EPISODE_LEN = 600                              # 轨迹步长（如60秒数据，10Hz采样）
    MY_PATTERN = "structured"                         # "structured" or "random"
    # -----------------------

    generate_fake_uav_dataset(
        save_path=MY_SAVE_PATH, 
        num_episodes=MY_NUM_EPISODES, 
        episode_len=MY_EPISODE_LEN,
        pattern=MY_PATTERN
    )
