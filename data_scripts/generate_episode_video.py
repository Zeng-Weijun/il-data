#!/usr/bin/env python3
"""
Falcon Episode视频生成脚本
用于生成包含RGB图像、深度图和Topdown Map的同步视频，并实时显示指标
"""

import os
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from typing import Dict, Any, Optional, Tuple, List
import cv2
from datetime import datetime

def load_episode_data(base_dir: str, episode_filename: str) -> Dict[str, Any]:
    """
    加载一个episode的所有数据文件
    
    Args:
        base_dir: 基础数据目录路径
        episode_filename: episode文件名（不含扩展名），如 "2azQ1b91cZZ_ep000001"
    
    Returns:
        包含所有数据的字典
    """
    base_path = Path(base_dir)
    data_folders = ['jaw_rgb_data', 'jaw_depth_data', 'topdown_map', 'other_data']
    
    episode_data = {}
    
    for folder in data_folders:
        file_path = base_path / folder / f"{episode_filename}.pkl"
        
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                episode_data[folder] = data
                print(f"✅ 成功加载: {folder}/{episode_filename}.pkl")
            except Exception as e:
                print(f"❌ 加载失败: {folder}/{episode_filename}.pkl - {e}")
                episode_data[folder] = None
        else:
            print(f"❌ 文件不存在: {file_path}")
            episode_data[folder] = None
    
    return episode_data

def find_latest_data_directory() -> str:
    """
    自动查找最新的模仿学习数据目录
    
    Returns:
        最新数据目录的完整路径
    """
    falcon_data_root = "/root/zwj/Falcon/falcon_imitation_data"
    
    if not os.path.exists(falcon_data_root):
        raise FileNotFoundError(f"数据根目录不存在: {falcon_data_root}")
    
    # 查找所有时间戳目录（格式：YYYYMMDD_HHMMSS）
    timestamp_dirs = []
    for item in os.listdir(falcon_data_root):
        item_path = os.path.join(falcon_data_root, item)
        if os.path.isdir(item_path) and len(item) == 15 and '_' in item:
            try:
                # 验证是否为有效的时间戳格式
                date_part, time_part = item.split('_')
                if len(date_part) == 8 and len(time_part) == 6:
                    timestamp_dirs.append(item)
            except ValueError:
                continue
    
    if not timestamp_dirs:
        raise FileNotFoundError(f"在 {falcon_data_root} 中未找到任何时间戳目录")
    
    # 按时间戳排序，获取最新的
    latest_timestamp = sorted(timestamp_dirs)[-1]
    latest_dir = os.path.join(falcon_data_root, latest_timestamp)
    
    print(f"🔍 自动检测到最新数据目录: {latest_dir}")
    return latest_dir

def find_episodes(base_dir: str) -> list:
    """
    查找目录中所有可用的episode
    
    Args:
        base_dir: 基础数据目录
    
    Returns:
        episode文件名列表（不含扩展名）
    """
    base_path = Path(base_dir)
    episodes = set()
    
    # 从other_data文件夹中查找所有episode
    other_data_dir = base_path / 'other_data'
    if other_data_dir.exists():
        for pkl_file in other_data_dir.glob('*.pkl'):
            episode_name = pkl_file.stem  # 去掉.pkl扩展名
            episodes.add(episode_name)
    
    return sorted(list(episodes))

def extract_frame_data(episode_data: Dict[str, Any], frame_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    提取指定帧的RGB、深度和topdown数据以及指标信息
    
    Args:
        episode_data: episode数据
        frame_idx: 帧索引
    
    Returns:
        (rgb_frame, depth_frame, topdown_frame, metrics)
    """
    rgb_frame = None
    depth_frame = None
    topdown_frame = None
    metrics = {}
    
    # 提取RGB数据
    if 'jaw_rgb_data' in episode_data and episode_data['jaw_rgb_data'] is not None:
        rgb_data = episode_data['jaw_rgb_data']
        if isinstance(rgb_data, dict) and len(rgb_data) > 0:
            first_key = list(rgb_data.keys())[0]
            rgb_array = rgb_data[first_key]
            if isinstance(rgb_array, np.ndarray) and frame_idx < len(rgb_array):
                rgb_frame = rgb_array[frame_idx]
    
    # 提取深度数据
    if 'jaw_depth_data' in episode_data and episode_data['jaw_depth_data'] is not None:
        depth_data = episode_data['jaw_depth_data']
        if isinstance(depth_data, dict) and len(depth_data) > 0:
            first_key = list(depth_data.keys())[0]
            depth_array = depth_data[first_key]
            if isinstance(depth_array, np.ndarray) and frame_idx < len(depth_array):
                depth_frame = depth_array[frame_idx]
                if len(depth_frame.shape) == 3 and depth_frame.shape[2] == 1:
                    depth_frame = depth_frame.squeeze(2)
    
    # 提取topdown数据
    if 'topdown_map' in episode_data and episode_data['topdown_map'] is not None:
        topdown_data = episode_data['topdown_map']
        if isinstance(topdown_data, dict) and len(topdown_data) > 0:
            first_key = list(topdown_data.keys())[0]
            topdown_array = topdown_data[first_key]
            if isinstance(topdown_array, np.ndarray) and frame_idx < len(topdown_array):
                topdown_frame = topdown_array[frame_idx]
    
    # 提取指标数据
    if 'other_data' in episode_data and episode_data['other_data'] is not None:
        other_data = episode_data['other_data']
        
        # 提取奖励
        if 'rewards' in other_data and isinstance(other_data['rewards'], np.ndarray):
            if frame_idx < len(other_data['rewards']):
                metrics['reward'] = other_data['rewards'][frame_idx]
                metrics['cumulative_reward'] = np.sum(other_data['rewards'][:frame_idx+1])
        
        # 提取info数据中的指标
        if 'info_data' in other_data and other_data['info_data'] is not None:
            info_data = other_data['info_data']
            
            if isinstance(info_data, dict):
                # 新格式：info_data是字典，每个键对应一个numpy数组
                for key, value in info_data.items():
                    if isinstance(value, np.ndarray) and frame_idx < len(value):
                        metrics[key] = value[frame_idx]
            elif isinstance(info_data, list) and frame_idx < len(info_data):
                # 旧格式：info_data是字典列表
                frame_info = info_data[frame_idx]
                if isinstance(frame_info, dict):
                    metrics.update(frame_info)
    
    return rgb_frame, depth_frame, topdown_frame, metrics

def enhance_topdown_with_agents(topdown_frame: np.ndarray, metrics: Dict[str, Any]) -> np.ndarray:
    """
    在topdown图上标记ego和agent的位置（如果有位置信息）
    
    Args:
        topdown_frame: 原始topdown图像
        metrics: 包含位置信息的指标字典
    
    Returns:
        增强后的topdown图像
    """
    if topdown_frame is None:
        return None
    
    # 复制图像以避免修改原始数据
    enhanced_frame = topdown_frame.copy()
    
    # 如果是灰度图，转换为RGB
    if len(enhanced_frame.shape) == 2:
        enhanced_frame = np.stack([enhanced_frame] * 3, axis=-1)
    
    # 获取图像尺寸
    height, width = enhanced_frame.shape[:2]
    
    # 在图像中心标记ego位置（红色圆圈）
    ego_x, ego_y = width // 2, height // 2
    cv2.circle(enhanced_frame, (ego_x, ego_y), 8, (255, 0, 0), 2)  # 红色圆圈
    cv2.circle(enhanced_frame, (ego_x, ego_y), 3, (255, 0, 0), -1)  # 红色实心圆
    
    # 如果有GPS或位置信息，可以标记目标位置（绿色）
    if 'distance_to_goal' in metrics:
        # 在右上角标记目标方向（示例）
        goal_x, goal_y = width - 30, 30
        cv2.circle(enhanced_frame, (goal_x, goal_y), 6, (0, 255, 0), 2)  # 绿色圆圈
        cv2.circle(enhanced_frame, (goal_x, goal_y), 2, (0, 255, 0), -1)  # 绿色实心圆
    
    # 添加路径轨迹（如果有历史位置信息）
    # 这里可以根据实际数据格式添加轨迹绘制逻辑
    
    return enhanced_frame

def create_episode_video(episode_data: Dict[str, Any], episode_name: str, output_path: str, fps: int = 10) -> None:
    """
    创建episode视频
    
    Args:
        episode_data: episode数据
        episode_name: episode名称
        output_path: 输出视频路径
        fps: 视频帧率
    """
    print(f"\n🎬 开始生成视频: {episode_name}")
    print(f"📁 输出路径: {output_path}")
    
    # 确定总帧数
    total_frames = 0
    for data_type in ['jaw_rgb_data', 'jaw_depth_data', 'topdown_map']:
        if data_type in episode_data and episode_data[data_type] is not None:
            data = episode_data[data_type]
            if isinstance(data, dict) and len(data) > 0:
                first_key = list(data.keys())[0]
                array_data = data[first_key]
                if isinstance(array_data, np.ndarray):
                    total_frames = max(total_frames, len(array_data))
    
    if total_frames == 0:
        print("❌ 未找到有效的图像数据")
        return
    
    print(f"📊 总帧数: {total_frames}")
    
    # 设置图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Episode: {episode_name}', fontsize=16, fontweight='bold')
    
    # 调整子图布局
    axes[0, 0].set_title('RGB Camera')
    axes[0, 1].set_title('Depth Camera')
    axes[1, 0].set_title('Topdown Map')
    axes[1, 1].set_title('Metrics')
    
    # 隐藏坐标轴
    for i in range(2):
        for j in range(2):
            if i != 1 or j != 1:  # 保留metrics子图的坐标轴
                axes[i, j].axis('off')
    
    # 初始化图像显示
    rgb_im = axes[0, 0].imshow(np.zeros((224, 224, 3), dtype=np.uint8))
    depth_im = axes[0, 1].imshow(np.zeros((224, 224)), cmap='viridis')
    topdown_im = axes[1, 0].imshow(np.zeros((224, 224, 3), dtype=np.uint8))
    
    # 设置metrics子图
    axes[1, 1].clear()
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    metrics_text = axes[1, 1].text(0.05, 0.95, '', transform=axes[1, 1].transAxes, 
                                  fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].axis('off')
    
    def animate(frame_idx):
        """动画更新函数"""
        # 提取当前帧数据
        rgb_frame, depth_frame, topdown_frame, metrics = extract_frame_data(episode_data, frame_idx)
        
        # 更新RGB图像
        if rgb_frame is not None and len(rgb_frame.shape) == 3:
            rgb_im.set_array(rgb_frame)
        else:
            # 显示占位符
            placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
            placeholder[:, :, 0] = 50  # 深红色背景表示无数据
            rgb_im.set_array(placeholder)
        
        # 更新深度图
        if depth_frame is not None and len(depth_frame.shape) == 2:
            depth_im.set_array(depth_frame)
            depth_im.set_clim(vmin=np.min(depth_frame), vmax=np.max(depth_frame))
        else:
            # 显示占位符
            placeholder = np.zeros((224, 224))
            depth_im.set_array(placeholder)
        
        # 更新topdown图
        if topdown_frame is not None:
            # 归一化topdown图像到0-255范围以提高可视性
            if len(topdown_frame.shape) == 2:
                # 灰度图，归一化到0-255
                normalized_topdown = ((topdown_frame - np.min(topdown_frame)) / 
                                    (np.max(topdown_frame) - np.min(topdown_frame) + 1e-8) * 255).astype(np.uint8)
                # 转换为RGB格式
                normalized_topdown = np.stack([normalized_topdown] * 3, axis=-1)
            else:
                # 已经是RGB格式，直接归一化
                normalized_topdown = ((topdown_frame - np.min(topdown_frame)) / 
                                    (np.max(topdown_frame) - np.min(topdown_frame) + 1e-8) * 255).astype(np.uint8)
            
            # 增强topdown图（标记智能体位置）
            enhanced_topdown = enhance_topdown_with_agents(normalized_topdown, metrics)
            if enhanced_topdown is not None:
                topdown_im.set_array(enhanced_topdown)
            else:
                topdown_im.set_array(normalized_topdown)
        else:
            # 显示占位符
            placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
            placeholder[:, :, 1] = 50  # 深绿色背景表示无数据
            topdown_im.set_array(placeholder)
        
        # 更新指标文本
        metrics_str = f"Frame: {frame_idx + 1}/{total_frames}\n"
        metrics_str += f"Progress: {(frame_idx + 1) / total_frames * 100:.1f}%\n\n"
        
        if metrics:
            # 显示主要指标
            key_metrics = ['reward', 'cumulative_reward', 'distance_to_goal', 'success', 
                          'spl', 'num_steps', 'did_multi_agents_collide']
            
            for key in key_metrics:
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            metrics_str += f"{key}: {value:.3f}\n"
                        else:
                            metrics_str += f"{key}: {value}\n"
                    else:
                        metrics_str += f"{key}: {value}\n"
            
            # 显示其他指标
            other_metrics = {k: v for k, v in metrics.items() if k not in key_metrics}
            if other_metrics:
                metrics_str += "\nOther metrics:\n"
                for key, value in list(other_metrics.items())[:5]:  # 只显示前5个其他指标
                    if isinstance(value, (int, float, bool)):
                        if isinstance(value, float):
                            metrics_str += f"{key}: {value:.3f}\n"
                        else:
                            metrics_str += f"{key}: {value}\n"
        
        metrics_text.set_text(metrics_str)
        
        return [rgb_im, depth_im, topdown_im, metrics_text]
    
    # 创建动画
    print(f"🎥 开始渲染视频...")
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=1000//fps, blit=False)
    
    # 保存视频
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存为MP4
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Falcon'), bitrate=1800)
        anim.save(output_path, writer=writer)
        
        print(f"✅ 视频生成成功: {output_path}")
        print(f"📊 视频信息: {total_frames}帧, {fps}fps, 时长{total_frames/fps:.1f}秒")
        
    except Exception as e:
        print(f"❌ 视频保存失败: {e}")
        print("💡 请确保已安装ffmpeg: sudo apt-get install ffmpeg")
    
    plt.close(fig)

def generate_video_for_episode(base_dir: str, episode_name: str, output_dir: str = None, fps: int = 10) -> None:
    """
    为指定episode生成视频
    
    Args:
        base_dir: 数据目录
        episode_name: episode名称
        output_dir: 输出目录（可选）
        fps: 视频帧率
    """
    print("="*80)
    print(f"🎬 Falcon Episode视频生成器")
    print(f"📂 数据目录: {base_dir}")
    print(f"🎯 Episode: {episode_name}")
    print("="*80)
    
    # 加载数据
    episode_data = load_episode_data(base_dir, episode_name)
    
    # 检查数据完整性
    has_data = False
    for data_type in ['jaw_rgb_data', 'jaw_depth_data', 'topdown_map']:
        if episode_data.get(data_type) is not None:
            has_data = True
            break
    
    if not has_data:
        print("❌ 未找到有效的图像数据，无法生成视频")
        return
    
    # 设置输出路径
    if output_dir is None:
        output_dir = os.path.join(base_dir, 'videos')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{episode_name}_{timestamp}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    # 生成视频
    create_episode_video(episode_data, episode_name, output_path, fps)
    
    print("\n" + "="*80)
    print("✅ 视频生成完成")
    print("="*80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 如果提供了episode名称
        episode_name = sys.argv[1]
        if len(sys.argv) > 2:
            # 如果提供了自定义目录
            base_dir = sys.argv[2]
        else:
            # 自动检测最新数据目录
            try:
                base_dir = find_latest_data_directory()
            except FileNotFoundError as e:
                print(f"❌ {e}")
                print("请手动指定数据目录路径")
                sys.exit(1)
        
        # 设置输出目录和帧率
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        fps = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        
        generate_video_for_episode(base_dir, episode_name, output_dir, fps)
    else:
        # 列出所有可用的episode并生成第一个的视频
        try:
            base_dir = find_latest_data_directory()
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("请手动指定数据目录路径")
            sys.exit(1)
            
        episodes = find_episodes(base_dir)
        
        if episodes:
            print(f"📋 找到 {len(episodes)} 个episode:")
            for i, episode in enumerate(episodes[:10]):  # 只显示前10个
                print(f"   {i+1}. {episode}")
            
            if len(episodes) > 10:
                print(f"   ... 还有 {len(episodes) - 10} 个episode")
            
            # 生成第一个episode的视频
            print(f"\n🎯 自动生成第一个episode的视频: {episodes[0]}")
            generate_video_for_episode(base_dir, episodes[0])
        else:
            print(f"❌ 在 {base_dir} 中未找到任何episode数据")
            print("\n使用方法:")
            print(f"  python {sys.argv[0]} <episode_name> [base_dir] [output_dir] [fps]")
            print(f"  例如: python {sys.argv[0]} 2azQ1b91cZZ_ep000001")
            print(f"  例如: python {sys.argv[0]} 2azQ1b91cZZ_ep000001 /path/to/data /path/to/output 15")