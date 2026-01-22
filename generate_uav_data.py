import numpy as np
import zarr
import os

def generate_fake_uav_dataset(save_path, num_episodes, episode_len):
    """
    生成3v3无人机空战伪数据集
    观测: 6架机 * 7维 = 42维 (pos_x, pos_y, pos_z, acc_t, acc_n, roll, hp)
    动作: 我方3架机 * 3维 = 9维 (acc_t, acc_n, roll)
    """
    # 确保父目录存在
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    obs_dim = 42    # 6架无人机 * 7维属性
    action_dim = 9  # 3架我方无人机 * 3维动作指令

    all_obs = []
    all_actions = []
    episode_ends = []
    current_idx = 0

    print(f"正在生成数据: {num_episodes} 条轨迹, 每条长度 {episode_len}...")

    for i in range(num_episodes):
        # 模拟生成观测数据 (T, 42)
        # 包含位置(0-1000m), 动作(物理值), 血量(0-1)
        obs = np.random.uniform(0, 1, (episode_len, obs_dim)).astype(np.float32)
        
        # 模拟生成动作数据 (T, 9)
        # 对应你说的: 切向过载、法向过载、滚转角
        action = np.random.uniform(-1, 1, (episode_len, action_dim)).astype(np.float32)

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
    MY_SAVE_PATH = "/home/dfc/Documents/zdy_documents/data/uav_combat/test_data.zarr" # 存储路径
    MY_NUM_EPISODES = 20                              # 轨迹数量
    MY_EPISODE_LEN = 600                              # 轨迹步长（如60秒数据，10Hz采样）
    # -----------------------

    generate_fake_uav_dataset(
        save_path=MY_SAVE_PATH, 
        num_episodes=MY_NUM_EPISODES, 
        episode_len=MY_EPISODE_LEN
    )