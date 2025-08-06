from habitat_baselines.common.tensor_dict import TensorDict
import torch
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.multi_agent.pop_play_wrappers import MultiStorage
import imageio
import os.path as osp
import numpy as np

def inspect_rollouts(file_path):
    """
    加载并检查保存的 RolloutStorage 对象。

    Args:
        file_path (str): 保存的 .pt 文件的路径。
    """
    print(f"正在加载文件: {file_path} ...")
    
    
    # 加载完整的 RolloutStorage 对象
    # torch.load 会使用 __setstate__ 方法来重建对象
    rollouts = torch.load(file_path, weights_only=False, map_location='cpu')
    print("RolloutStorage 对象加载成功！")
    


    # --- 数据检查 ---
    
    # RolloutStorage 的核心数据都存储在 self.buffers 这个 TensorDict 中
    # 我们直接访问这个属性
        
    buffers = rollouts._active_storages[0].buffers if isinstance(rollouts, MultiStorage) else rollouts.buffers
    
    print("\n--- 缓冲区结构 ---")
    print("缓冲区中包含的键:", list(buffers.keys()))
    if "observations" in buffers:
        print("观测数据中包含的传感器:", list(buffers["observations"].keys()))
        
    print("\n--- 数据详情 ---")

    # 1. 检查奖励 (rewards)
    if "rewards" in buffers and isinstance(buffers["rewards"], torch.Tensor):
        rewards = buffers["rewards"]
        print(f"奖励 (rewards) 的形状: {rewards.shape}")
        # 打印出前5个环境，前10步的奖励
        print("奖励数据 (前5个环境，前10步):")
        print(rewards[:10, :5, 0])
    else:
        print("未找到 'rewards' 数据。")
        
    # 2. 检查掩码 (masks) - 这个至关重要
    if "masks" in buffers and isinstance(buffers["masks"], torch.Tensor):
        masks = buffers["masks"]
        print(f"\n掩码 (masks) 的形状: {masks.shape}")
        print("掩码代表 'not done'。True (1) 表示未结束，False (0) 表示 episode 结束。")
        # 打印出前5个环境，前10步的掩码
        print("掩码数据 (前5个环境，前10步):")
        # 将布尔值转为整数方便查看
        print(masks[:10, :5, 0].int()) 
        
        # 统计有多少个 'done' 信号
        total_steps = masks.numel()
        done_count = total_steps - masks.sum().item()
        print(f"在整个缓冲区中，共发生了 {int(done_count)} 次 episode 结束事件 (done=True)。")

    else:
        print("未找到 'masks' 数据。")

    # 3. 检查回报 (returns) - 如果已经计算过的话
    if "returns" in buffers and isinstance(buffers["returns"], torch.Tensor):
        returns = buffers["returns"]
        print(f"\n回报 (returns) 的形状: {returns.shape}")
        # 回报通常在 rollout 结束后才计算，所以我们看前几步可能还是0
        print("回报数据 (前5个环境，前10步):")
        print(returns[:10, :5, 0])


def create_rollout_video(rollout_file_path, output_video_path, env_index=0):
    """
    从保存的 rollout 文件中创建一个视频。

    Args:
        rollout_file_path (str): .pt 文件的路径。
        output_video_path (str): 保存视频的路径，如 'rollout_video.mp4'。
        env_index (int): 你想可视化的环境索引（从 0 开始）。
    """
    print(f"正在加载 {rollout_file_path}...")
    try:
        # 加载 RolloutStorage 对象
        rollouts = torch.load(rollout_file_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"加载失败: {e}")
        return
    
    rollouts = rollouts._active_storages[0]

    # !!! 关键: 找到你的 RGB 传感器键名 !!!
    # 你需要根据你的缓冲区结构来确定正确的键名。
    # 对于多智能体，它可能是 'agent_0_rgb_sensor' 或类似的名字。
    # 这里我们先猜测一个，你需要根据你的实际情况修改。
    
    # 尝试在 observations 字典中寻找包含 'rgb' 的键
    obs_keys = rollouts.buffers['observations'].keys()
    rgb_sensor_key = None
    for key in obs_keys:
        # 这是一个启发式查找，找到第一个包含'rgb'的key
        if 'depth' in key.lower():
            rgb_sensor_key = key
            break
            
    if rgb_sensor_key is None:
        print("错误: 在观测数据中找不到 RGB 传感器。请检查你的传感器键名。")
        print("可用的观测键:", list(obs_keys))
        return

    print(f"使用传感器 '{rgb_sensor_key}' 创建视频...")

    depth_observations = rollouts.buffers['observations'][rgb_sensor_key][:, env_index].cpu().numpy()


# --- 核心步骤：归一化深度图以便可视化 ---
    frames = []
    # 找到所有帧中的最大深度值用于归一化（忽略无穷大的值）
    max_depth = np.max(depth_observations[np.isfinite(depth_observations)])
    if max_depth == 0:
        print("警告: 缓冲区中所有深度值均为 0。视频将是全黑的。")
        max_depth = 1.0 # 防止除以零

    for depth_frame in depth_observations:
        # 将深度值归一化到 0-1 的范围
        normalized_frame = depth_frame / max_depth
        # 翻转颜色，让近处更亮，远处更暗
        normalized_frame = 1.0 - normalized_frame
        # 转换到 0-255 的灰度范围
        frame_uint8 = (normalized_frame * 255).astype(np.uint8)
        frames.append(frame_uint8)

    print(f"共找到 {len(frames)} 帧图像，正在写入视频...")
    
    try:
        # 使用 imageio 将帧序列写入视频文件
        imageio.mimsave(output_video_path, frames, fps=10)
        print(f"深度视频已成功保存到: {output_video_path}")
    except Exception as e:
        print(f"保存视频时出错: {e}")


if __name__ == '__main__':
    # --- 请修改以下参数 ---
    ROLLOUT_FILE = 'rollouts/rollouts_data_step_0.pt' # 你的 .pt 文件
    OUTPUT_VIDEO = 'my_rollout_video.mp4'      # 输出视频的文件名
    ENVIRONMENT_TO_VIEW = 0                    # 你想看的环境编号 (0 到 7)
    # ---------------------
    
    create_rollout_video(ROLLOUT_FILE, OUTPUT_VIDEO, ENVIRONMENT_TO_VIEW)


# if __name__ == '__main__':
#     # 替换成你自己的文件路径
#     # 例如: "rollouts_data_step_128.pt"
#     file_to_inspect = "rollouts/rollouts_data_step_0.pt" 
#     inspect_rollouts(file_to_inspect)