#!/usr/bin/env python3
"""
从保存的rollout数据文件生成视频的脚本

使用方法:
python scripts/create_rollout_video.py --data_file /path/to/rollout_data.pt --output_dir /path/to/output
"""

import argparse
import os
import torch
import numpy as np
import imageio
from pathlib import Path

def print_performance_stats(data):
    """
    打印rollout数据的性能统计信息
    
    Args:
        data: 加载的rollout数据字典
    """
    print("\n" + "="*50)
    print("性能统计信息")
    print("="*50)
    
    if 'running_episode_stats' not in data:
        print("警告: 数据中未找到running_episode_stats")
        return
    
    stats = data['running_episode_stats']
    
    # 基本信息
    num_envs = data.get('num_envs', 'Unknown')
    num_steps = data.get('num_steps', 'Unknown')
    update_count = data.get('update_count', 'Unknown')
    
    print(f"更新轮次: {update_count}")
    print(f"环境数量: {num_envs}")
    print(f"步数: {num_steps}")
    
    # 奖励统计
    if 'rewards' in data:
        rewards = data['rewards'].cpu().numpy()
        total_reward = float(rewards.sum())
        mean_reward = float(rewards.mean())
        print(f"\n奖励统计:")
        print(f"  总奖励: {total_reward:.4f}")
        print(f"  平均奖励: {mean_reward:.4f}")
    
    # 关键性能指标
    key_metrics = ['success', 'spl', 'psc', 'stl', 'distance_to_goal', 
                   'human_collision', 'did_multi_agents_collide']
    
    print(f"\n各环境性能指标:")
    print(f"{'环境':<4} {'成功率':<8} {'SPL':<8} {'PSC':<8} {'STL':<8} {'目标距离':<10} {'人体碰撞':<8} {'智能体碰撞':<10}")
    print("-" * 70)
    
    for env_idx in range(num_envs):
        env_stats = {}
        for metric in key_metrics:
            if metric in stats:
                value = stats[metric][env_idx].item() if hasattr(stats[metric][env_idx], 'item') else stats[metric][env_idx]
                env_stats[metric] = value
            else:
                env_stats[metric] = 'N/A'
        
        print(f"{env_idx:<4} {env_stats.get('success', 'N/A'):<8.3f} {env_stats.get('spl', 'N/A'):<8.3f} "
              f"{env_stats.get('psc', 'N/A'):<8.3f} {env_stats.get('stl', 'N/A'):<8.3f} "
              f"{env_stats.get('distance_to_goal', 'N/A'):<10.3f} {env_stats.get('human_collision', 'N/A'):<8.3f} "
              f"{env_stats.get('did_multi_agents_collide', 'N/A'):<10.3f}")
    
    # 总体统计
    print(f"\n总体统计:")
    overall_stats = {}
    for metric in key_metrics:
        if metric in stats:
            values = [stats[metric][i].item() if hasattr(stats[metric][i], 'item') else stats[metric][i] 
                     for i in range(num_envs)]
            overall_stats[metric] = {
                'mean': sum(values) / len(values),
                'max': max(values),
                'min': min(values)
            }
    
    if 'success' in overall_stats:
        print(f"  平均成功率: {overall_stats['success']['mean']:.3f} (最高: {overall_stats['success']['max']:.3f}, 最低: {overall_stats['success']['min']:.3f})")
    
    if 'spl' in overall_stats:
        print(f"  平均SPL: {overall_stats['spl']['mean']:.3f} (最高: {overall_stats['spl']['max']:.3f}, 最低: {overall_stats['spl']['min']:.3f})")
    
    if 'psc' in overall_stats:
        print(f"  平均PSC: {overall_stats['psc']['mean']:.3f} (最高: {overall_stats['psc']['max']:.3f}, 最低: {overall_stats['psc']['min']:.3f})")
    
    if 'stl' in overall_stats:
        print(f"  平均STL: {overall_stats['stl']['mean']:.3f} (最高: {overall_stats['stl']['max']:.3f}, 最低: {overall_stats['stl']['min']:.3f})")
    
    if 'distance_to_goal' in overall_stats:
        print(f"  平均目标距离: {overall_stats['distance_to_goal']['mean']:.3f}")
    
    if 'human_collision' in overall_stats:
        collision_rate = overall_stats['human_collision']['mean']
        print(f"  人体碰撞率: {collision_rate:.3f}")
    
    if 'did_multi_agents_collide' in overall_stats:
        agent_collision_rate = overall_stats['did_multi_agents_collide']['mean']
        print(f"  智能体碰撞率: {agent_collision_rate:.3f}")
    
    print("="*50)

def create_rollout_video(data_file_path, output_dir=None, env_index=0, fps=10):
    """
    从保存的rollout数据文件创建视频
    
    Args:
        data_file_path (str): rollout数据文件路径 (.pt文件)
        output_dir (str): 输出目录，如果为None则保存到数据文件同目录
        env_index (int): 要可视化的环境索引
        fps (int): 视频帧率
    """
    return create_single_env_video(data_file_path, output_dir, env_index, fps)

def create_all_env_videos(data_file_path, output_dir=None, fps=10):
    """
    从保存的rollout数据文件为所有环境创建视频
    
    Args:
        data_file_path (str): rollout数据文件路径 (.pt文件)
        output_dir (str): 输出目录，如果为None则保存到数据文件同目录
        fps (int): 视频帧率
    """
    print(f"正在为所有环境生成视频: {data_file_path}")
    
    try:
        # 加载rollout数据
        data = torch.load(data_file_path, map_location='cpu', weights_only=False)
        print(f"数据加载成功，数据类型: {type(data)}")
        
        if isinstance(data, dict) and 'observations' in data:
            observations = data['observations']
            print(f"观测数据包含传感器: {list(observations.keys())}")
        else:
            print(f"错误: 数据格式不正确，期望包含'observations'键的字典")
            return
            
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 智能检测传感器
    obs_keys = observations.keys()
    
    # 优先使用RGB传感器
    rgb_sensor_key = None
    depth_sensor_key = None
    
    for key in obs_keys:
        if 'rgb' in key.lower():
            rgb_sensor_key = key
            break
    
    for key in obs_keys:
        if 'depth' in key.lower():
            depth_sensor_key = key
            break
    
    sensor_to_use = None
    is_depth = False
    
    if rgb_sensor_key is not None:
        sensor_to_use = rgb_sensor_key
        print(f"找到RGB传感器 '{sensor_to_use}' 用于生成视频")
    elif depth_sensor_key is not None:
        sensor_to_use = depth_sensor_key
        is_depth = True
        print(f"未找到RGB传感器，将使用深度传感器 '{sensor_to_use}'")
    else:
        print(f"错误: 未找到RGB或深度传感器。可用传感器: {list(obs_keys)}")
        return
    
    # 提取传感器数据
    sensor_data = observations[sensor_to_use]
    print(f"传感器数据形状: {sensor_data.shape}")
    
    num_envs = sensor_data.shape[1]
    print(f"检测到 {num_envs} 个环境")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(data_file_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个环境生成视频
    for env_index in range(num_envs):
        print(f"\n正在处理环境 {env_index}/{num_envs-1}...")
        create_single_env_video(data_file_path, output_dir, env_index, fps, 
                               sensor_data=sensor_data, sensor_to_use=sensor_to_use, is_depth=is_depth)
    
    print(f"\n所有视频生成完成！共生成 {num_envs} 个视频文件。")
    
    # 打印性能统计信息
    print_performance_stats(data)

def create_single_env_video(data_file_path, output_dir=None, env_index=0, fps=10, 
                           sensor_data=None, sensor_to_use=None, is_depth=None):
    """
    为单个环境创建视频
    
    Args:
        data_file_path (str): rollout数据文件路径 (.pt文件)
        output_dir (str): 输出目录，如果为None则保存到数据文件同目录
        env_index (int): 要可视化的环境索引
        fps (int): 视频帧率
        sensor_data: 预加载的传感器数据（可选，用于批量处理）
        sensor_to_use: 使用的传感器名称（可选）
        is_depth: 是否为深度传感器（可选）
    """
    # 如果没有预加载数据，则加载数据文件
    if sensor_data is None:
        print(f"正在加载rollout数据: {data_file_path}")
        
        try:
            # 加载rollout数据
            data = torch.load(data_file_path, map_location='cpu', weights_only=False)
            print(f"数据加载成功，数据类型: {type(data)}")
            
            if isinstance(data, dict) and 'observations' in data:
                observations = data['observations']
                print(f"观测数据包含传感器: {list(observations.keys())}")
            else:
                print(f"错误: 数据格式不正确，期望包含'observations'键的字典")
                return
                
        except Exception as e:
            print(f"加载数据失败: {e}")
            return
        
        # 智能检测传感器
        obs_keys = observations.keys()
        
        # 优先使用RGB传感器
        rgb_sensor_key = None
        depth_sensor_key = None
        
        for key in obs_keys:
            if 'rgb' in key.lower():
                rgb_sensor_key = key
                break
        
        for key in obs_keys:
            if 'depth' in key.lower():
                depth_sensor_key = key
                break
        
        if rgb_sensor_key is not None:
            sensor_to_use = rgb_sensor_key
            is_depth = False
            print(f"找到RGB传感器 '{sensor_to_use}' 用于生成视频")
        elif depth_sensor_key is not None:
            sensor_to_use = depth_sensor_key
            is_depth = True
            print(f"未找到RGB传感器，将使用深度传感器 '{sensor_to_use}'")
        else:
            print(f"错误: 未找到RGB或深度传感器。可用传感器: {list(obs_keys)}")
            return
        
        # 提取传感器数据
        sensor_data = observations[sensor_to_use]
    print(f"传感器数据形状: {sensor_data.shape}")
    
    # 检查环境索引是否有效
    if env_index >= sensor_data.shape[1]:
        print(f"错误: 环境索引 {env_index} 超出范围，最大索引为 {sensor_data.shape[1]-1}")
        return
    
    # 提取指定环境的观测数据
    env_observations = sensor_data[:, env_index].cpu().numpy()
    print(f"环境 {env_index} 的观测数据形状: {env_observations.shape}")
    
    # 处理帧数据
    frames = []
    
    if is_depth:
        # 深度图处理
        print("处理深度图数据...")
        max_depth = np.max(env_observations[np.isfinite(env_observations)])
        if max_depth == 0:
            print("警告: 所有深度值为0，视频将是全黑的")
            max_depth = 1.0
        
        for depth_frame in env_observations:
            # 归一化深度值到0-1范围
            normalized_frame = np.clip(depth_frame / max_depth, 0, 1)
            # 翻转颜色，让近处更亮
            normalized_frame = 1.0 - normalized_frame
            # 转换到0-255灰度范围
            frame_uint8 = (normalized_frame * 255).astype(np.uint8)
            frames.append(frame_uint8)
    else:
        # RGB图处理
        print("处理RGB图数据...")
        for rgb_frame in env_observations:
            # 确保数据在0-255范围内
            if rgb_frame.max() <= 1.0:
                # 如果数据在0-1范围，转换到0-255
                frame_uint8 = (rgb_frame * 255).astype(np.uint8)
            else:
                # 如果已经在0-255范围，直接转换类型
                frame_uint8 = rgb_frame.astype(np.uint8)
            frames.append(frame_uint8)
    
    print(f"共处理 {len(frames)} 帧图像")
    
    # 确定输出目录和文件名
    if output_dir is None:
        output_dir = os.path.dirname(data_file_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成视频文件名
    data_filename = os.path.basename(data_file_path)
    video_filename = data_filename.replace('.pt', f'_env{env_index}_video.mp4')
    output_path = os.path.join(output_dir, video_filename)
    
    try:
        # 保存视频
        print(f"正在保存视频到: {output_path}")
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"视频保存成功: {output_path}")
        
        # 打印视频信息
        print(f"视频信息:")
        print(f"  - 帧数: {len(frames)}")
        print(f"  - 帧率: {fps} FPS")
        print(f"  - 时长: {len(frames)/fps:.2f} 秒")
        print(f"  - 传感器: {sensor_to_use}")
        print(f"  - 环境索引: {env_index}")
        
    except Exception as e:
        print(f"保存视频时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='从rollout数据生成视频')
    parser.add_argument('--data_file', type=str, required=True,
                       help='rollout数据文件路径 (.pt文件)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录 (默认为数据文件同目录)')
    parser.add_argument('--env_index', type=int, default=0,
                       help='要可视化的环境索引 (默认: 0)')
    parser.add_argument('--fps', type=int, default=10,
                       help='视频帧率 (默认: 10)')
    parser.add_argument('--all_envs', action='store_true',
                       help='为所有环境生成视频 (忽略env_index参数)')
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_file):
        print(f"错误: 数据文件不存在: {args.data_file}")
        return
    
    if args.all_envs:
        # 为所有环境生成视频
        create_all_env_videos(
            data_file_path=args.data_file,
            output_dir=args.output_dir,
            fps=args.fps
        )
    else:
        # 为单个环境生成视频
        create_rollout_video(
            data_file_path=args.data_file,
            output_dir=args.output_dir,
            env_index=args.env_index,
            fps=args.fps
        )

if __name__ == '__main__':
    main()