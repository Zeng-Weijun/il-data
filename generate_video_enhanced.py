#!/usr/bin/env python3
"""
增强版视频生成脚本：将falcon_evaluator保存的数据转换为MP4视频
同步播放RGB图像、深度图像和俯视图，并显示实时指标
支持不同人和狗的颜色标识
"""

import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob
from tqdm import tqdm
import argparse

# 定义不同智能体的颜色
AGENT_COLORS = {
    'human_0': (255, 0, 0),    # 红色
    'human_1': (0, 255, 0),    # 绿色
    'human_2': (0, 0, 255),    # 蓝色
    'human_3': (255, 255, 0),  # 黄色
    'dog_0': (255, 0, 255),    # 紫色
    'dog_1': (0, 255, 255),    # 青色
    'dog_2': (255, 128, 0),    # 橙色
    'dog_3': (128, 0, 255),    # 紫罗兰色
    'default': (255, 255, 255) # 白色（默认）
}

def load_episode_data(data_dir, scene_episode):
    """
    加载单个episode的所有数据
    """
    rgb_file = os.path.join(data_dir, 'jaw_rgb_data', f'{scene_episode}.pkl')
    depth_file = os.path.join(data_dir, 'jaw_depth_data', f'{scene_episode}.pkl')
    topdown_file = os.path.join(data_dir, 'topdown_map', f'{scene_episode}.pkl')
    other_file = os.path.join(data_dir, 'other_data', f'{scene_episode}.pkl')
    
    # 加载RGB数据
    with open(rgb_file, 'rb') as f:
        rgb_data = pickle.load(f)['agent_0_articulated_agent_jaw_rgb']
    
    # 加载深度数据
    with open(depth_file, 'rb') as f:
        depth_data = pickle.load(f)['agent_0_articulated_agent_jaw_depth']
    
    # 加载topdown数据
    with open(topdown_file, 'rb') as f:
        topdown_data = pickle.load(f)['top_down_map']
    
    # 加载其他数据（指标信息）
    with open(other_file, 'rb') as f:
        other_data = pickle.load(f)
    
    return rgb_data, depth_data, topdown_data, other_data

def normalize_depth(depth_image):
    """
    将深度图像归一化到0-255范围并转换为彩色图像
    """
    # 移除最后一个维度如果存在
    if len(depth_image.shape) == 3 and depth_image.shape[2] == 1:
        depth_image = depth_image.squeeze(2)
    
    # 归一化到0-1范围
    depth_norm = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min() + 1e-8)
    
    # 转换为0-255范围
    depth_norm = (depth_norm * 255).astype(np.uint8)
    
    # 应用颜色映射
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    
    return depth_colored

def process_topdown_map(topdown_data, frame_idx, info_data=None):
    """
    处理topdown地图数据，支持多智能体颜色标识
    """
    # 新格式：topdown_data是一个(T, H, W)的数组序列
    if isinstance(topdown_data, np.ndarray) and len(topdown_data.shape) == 3:
        # 确保frame_idx不超出范围
        frame_idx = min(frame_idx, topdown_data.shape[0] - 1)
        current_map = topdown_data[frame_idx]
        
        # 从info_data中获取智能体信息
        agent_coords = []
        agent_angles = []
        agent_types = []
        
        if info_data and frame_idx < len(info_data):
            current_info = info_data[frame_idx]
            # 尝试从info_data中获取智能体信息
            if 'top_down_map' in current_info:
                topdown_info = current_info['top_down_map']
                agent_coords = topdown_info.get('agent_map_coord', [])
                agent_angles = topdown_info.get('agent_angle', [])
                
                # 尝试获取智能体类型信息
                if 'agent_types' in topdown_info:
                    agent_types = topdown_info['agent_types']
                else:
                    # 如果没有类型信息，根据智能体数量推断
                    num_agents = len(agent_coords) if isinstance(agent_coords, list) else 1
                    agent_types = [f'agent_{i}' for i in range(num_agents)]
    else:
        # 旧格式：字典格式
        map_data = topdown_data['map']
        agent_coords = topdown_data.get('agent_map_coord', [])
        agent_angles = topdown_data.get('agent_angle', [])
        agent_types = topdown_data.get('agent_types', [])
        
        # 确保frame_idx不超出范围
        frame_idx = min(frame_idx, len(agent_coords) - 1 if agent_coords else 0)
        
        # 获取当前帧的地图
        if len(map_data.shape) == 2:
            # 如果地图是2D的，复制到所有帧
            current_map = map_data
        else:
            # 如果地图是3D的，取当前帧
            current_map = map_data[frame_idx]
    
    # 创建彩色地图
    # 0: 未探索区域 (黑色)
    # 1: 自由空间 (白色)
    # 2: 障碍物 (灰色)
    color_map = np.zeros((current_map.shape[0], current_map.shape[1], 3), dtype=np.uint8)
    color_map[current_map == 0] = [0, 0, 0]      # 黑色
    color_map[current_map == 1] = [255, 255, 255]  # 白色
    color_map[current_map == 2] = [128, 128, 128]  # 灰色
    
    # 绘制智能体位置
    if len(agent_coords) > 0:
        # 处理单个智能体的情况
        if not isinstance(agent_coords[0], (list, tuple, np.ndarray)):
            agent_coords = [agent_coords]
            agent_angles = [agent_angles] if agent_angles else [0]
            agent_types = [agent_types[0]] if agent_types else ['agent_0']
        
        # 绘制每个智能体
        for i, coords in enumerate(agent_coords):
            if isinstance(coords, (list, tuple, np.ndarray)) and len(coords) >= 2:
                agent_x, agent_y = int(coords[0]), int(coords[1])
                
                # 确保坐标在地图范围内
                if 0 <= agent_x < current_map.shape[1] and 0 <= agent_y < current_map.shape[0]:
                    # 确定智能体类型和颜色
                    agent_type = agent_types[i] if i < len(agent_types) else f'agent_{i}'
                    
                    # 根据智能体类型选择颜色
                    if 'human' in agent_type.lower():
                        color_key = f'human_{i}'
                    elif 'dog' in agent_type.lower():
                        color_key = f'dog_{i}'
                    else:
                        # 尝试从agent_type中提取索引
                        try:
                            agent_idx = int(agent_type.split('_')[-1])
                            color_key = f'human_{agent_idx}'  # 默认为human
                        except:
                            color_key = 'default'
                    
                    agent_color = AGENT_COLORS.get(color_key, AGENT_COLORS['default'])
                    
                    # 绘制智能体位置（彩色圆点）
                    cv2.circle(color_map, (agent_x, agent_y), 5, agent_color, -1)
                    cv2.circle(color_map, (agent_x, agent_y), 6, (0, 0, 0), 1)  # 黑色边框
                    
                    # 绘制智能体朝向（彩色箭头）
                    if i < len(agent_angles) and agent_angles[i] is not None:
                        angle = agent_angles[i]
                        arrow_length = 12
                        end_x = int(agent_x + arrow_length * np.cos(angle))
                        end_y = int(agent_y + arrow_length * np.sin(angle))
                        cv2.arrowedLine(color_map, (agent_x, agent_y), (end_x, end_y), agent_color, 2)
    
    return color_map

def create_metrics_panel(info_data, frame_idx, width=300, height=240):
    """
    创建指标显示面板
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel.fill(50)  # 深灰色背景
    
    if frame_idx < len(info_data):
        current_info = info_data[frame_idx]
        
        # 定义要显示的指标
        metrics = [
            ('Frame', frame_idx + 1),
            ('Distance to Goal', f"{current_info.get('distance_to_goal', 0):.2f}m"),
            ('Reward', f"{current_info.get('distance_to_goal_reward', 0):.3f}"),
            ('Success', 'Yes' if current_info.get('success', False) else 'No'),
            ('Steps', current_info.get('num_steps', 0)),
            ('SPL', f"{current_info.get('spl', 0):.3f}"),
            ('PSC', f"{current_info.get('psc', 0):.3f}"),
            ('Human Collision', 'Yes' if current_info.get('human_collision', False) else 'No'),
            ('Agent Collision', 'Yes' if current_info.get('did_multi_agents_collide', False) else 'No')
        ]
        
        # 绘制指标文本
        y_offset = 25
        for i, (label, value) in enumerate(metrics):
            text = f"{label}: {value}"
            
            # 根据指标类型选择颜色
            if 'Success' in label and 'Yes' in str(value):
                color = (0, 255, 0)  # 绿色表示成功
            elif 'Collision' in label and 'Yes' in str(value):
                color = (0, 0, 255)  # 红色表示碰撞
            else:
                color = (255, 255, 255)  # 白色默认
            
            cv2.putText(panel, text, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    return panel

def create_legend_panel(width=300, height=240):
    """
    创建颜色图例面板
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel.fill(30)  # 深灰色背景
    
    # 标题
    cv2.putText(panel, "Agent Colors:", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 绘制颜色图例
    y_offset = 50
    for i, (agent_type, color) in enumerate(AGENT_COLORS.items()):
        if agent_type == 'default':
            continue
            
        # 绘制颜色圆点
        cv2.circle(panel, (20, y_offset + i * 25), 8, color, -1)
        
        # 绘制标签
        label = agent_type.replace('_', ' ').title()
        cv2.putText(panel, label, (40, y_offset + i * 25 + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return panel

def resize_image(image, target_height):
    """
    调整图像大小，保持宽高比
    """
    h, w = image.shape[:2]
    aspect_ratio = w / h
    target_width = int(target_height * aspect_ratio)
    return cv2.resize(image, (target_width, target_height))

def create_combined_frame(rgb_frame, depth_frame, topdown_frame, metrics_panel, legend_panel, target_height=240):
    """
    将所有图像合并为一个帧
    """
    # 调整主要图像到相同高度
    rgb_resized = resize_image(rgb_frame, target_height)
    depth_resized = resize_image(depth_frame, target_height)
    topdown_resized = resize_image(topdown_frame, target_height)
    
    # 水平拼接主要图像
    main_images = np.hstack([rgb_resized, depth_resized, topdown_resized])
    
    # 垂直拼接指标和图例面板
    side_panel = np.vstack([metrics_panel, legend_panel])
    
    # 调整侧边面板高度以匹配主图像
    if side_panel.shape[0] != main_images.shape[0]:
        side_panel = cv2.resize(side_panel, (side_panel.shape[1], main_images.shape[0]))
    
    # 水平拼接所有内容
    combined = np.hstack([main_images, side_panel])
    
    return combined

def generate_video(data_dir, scene_episode, output_path, fps=10):
    """
    生成增强版视频
    """
    print(f"正在处理 {scene_episode}...")
    
    # 加载数据
    rgb_data, depth_data, topdown_data, other_data = load_episode_data(data_dir, scene_episode)
    
    # 获取帧数（取最小值确保同步）
    num_frames = min(len(rgb_data), len(depth_data), len(other_data['info_data']))
    print(f"总帧数: {num_frames}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 先创建一个示例帧来确定视频尺寸
    sample_rgb = rgb_data[0]
    sample_depth = normalize_depth(depth_data[0])
    sample_topdown = process_topdown_map(topdown_data, 0, other_data['info_data'])
    sample_metrics = create_metrics_panel(other_data['info_data'], 0)
    sample_legend = create_legend_panel()
    sample_combined = create_combined_frame(sample_rgb, sample_depth, sample_topdown, 
                                          sample_metrics, sample_legend)
    
    height, width = sample_combined.shape[:2]
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 生成每一帧
    for i in tqdm(range(num_frames), desc="生成视频帧"):
        # 获取当前帧数据
        rgb_frame = rgb_data[i]
        depth_frame = normalize_depth(depth_data[i])
        topdown_frame = process_topdown_map(topdown_data, i, other_data['info_data'])
        metrics_panel = create_metrics_panel(other_data['info_data'], i)
        legend_panel = create_legend_panel()
        
        # 合并帧
        combined_frame = create_combined_frame(rgb_frame, depth_frame, topdown_frame, 
                                             metrics_panel, legend_panel)
        
        # 转换颜色格式 (RGB -> BGR for OpenCV)
        combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
        
        # 写入视频
        video_writer.write(combined_frame_bgr)
    
    # 释放资源
    video_writer.release()
    print(f"增强版视频已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='生成falcon数据的增强版MP4视频')
    parser.add_argument('--data_dir', type=str, default='/home/ubuntu/Falcon/falcon_imitation_data/20250805_183840',
                       help='数据目录路径')
    parser.add_argument('--scene_episode', type=str, default='33ypawbKCQf.basis_ep000000',
                       help='场景和episode名称')
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频路径')
    parser.add_argument('--fps', type=int, default=10,
                       help='视频帧率')
    parser.add_argument('--all', action='store_true',
                       help='为所有episode生成视频')
    
    args = parser.parse_args()
    
    if args.all:
        # 获取所有episode
        rgb_files = glob.glob(os.path.join(args.data_dir, 'jaw_rgb_data', '*.pkl'))
        episodes = [os.path.basename(f).replace('.pkl', '') for f in rgb_files]
        
        for episode in episodes:
            output_path = args.output or f'{episode}_enhanced_video.mp4'
            try:
                generate_video(args.data_dir, episode, output_path, args.fps)
            except Exception as e:
                print(f"处理 {episode} 时出错: {e}")
    else:
        output_path = args.output or f'{args.scene_episode}_enhanced_video.mp4'
        generate_video(args.data_dir, args.scene_episode, output_path, args.fps)

if __name__ == '__main__':
    main()