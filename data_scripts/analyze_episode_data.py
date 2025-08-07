#!/usr/bin/env python3
"""
模仿学习数据分析脚本
用于分析一个episode的4个.pkl文件：jaw_rgb_data, jaw_depth_data, topdown_map, other_data
"""

import os
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

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

def get_data_explanation(name: str, data_type: str) -> str:
    """
    根据falcon_evaluator.py中的实际数据保存方式提供准确的解释说明
    
    Args:
        name: 数据字段名称
        data_type: 数据类型
    
    Returns:
        数据解释字符串
    """
    explanations = {
        # === 观测数据 ===
        'jaw_rgb_data': '🤖 机械臂末端摄像头RGB图像序列 - 从batch["agent_0_articulated_agent_jaw_rgb"]收集，转换为uint8格式，形状(T,H,W,C)',
        'jaw_depth_data': '📏 机械臂末端摄像头深度图像序列 - 从batch["agent_0_articulated_agent_jaw_depth"]收集，float32格式，形状(T,H,W,1)',
        'topdown_map': '🗺️ 俯视图地图数据 - 字典列表，每个字典包含map、fog_of_war_mask、agent_map_coord、agent_angle等字段',
        'rgb': '🎥 RGB图像序列，每帧包含彩色视觉信息',
        'depth': '📏 深度图像序列，每个像素表示到物体的距离',
        'map': '🗺️ 环境的俯视图表示 - uint8数组，形状(H,W)，数值范围[0-44]，表示不同地图元素',
        'fog_of_war_mask': '🌫️ 战争迷雾遮罩 - bool数组，标记agent已探索和未探索的区域',
        'agent_map_coord': '📍 Agent地图坐标 - float32数组，形状(2,)，表示agent在topdown地图中的[x,y]位置',
        'agent_angle': '🧭 Agent朝向角度 - float32值，单位弧度，表示agent在地图中的朝向',
        
        # === 动作和控制数据 ===
        'other_data': '📊 包含动作、奖励、状态等训练相关的核心数据',
        'actions': '🎮 智能体动作序列 - 从action_data.env_actions收集，每步的离散或连续动作，转换为int64格式',
        'global_actions': '🌐 跨环境动作矩阵 - 所有并行环境的动作序列，形状(T,envs,action_dim)，包含当前episode数据',
        
        # === 奖励数据 ===
        'rewards': '🏆 逐步奖励序列 - 从envs.step()返回的rewards_l收集，每个时间步的即时奖励',
        'global_rewards': '🏆 Episode奖励汇总 - 所有完成episode的总奖励列表，用于跨episode性能对比',
        
        # === 掩码数据 ===
        'masks': '🎭 Episode继续掩码 - 基于dones计算，True=episode继续，False=episode结束',
        'global_masks': '🎭 跨环境掩码数据 - 所有并行环境的episode结束标志，用于训练时序列处理',
        
        # === Info数据字段（从infos[i]收集） ===
        'info_data': '📊 Episode执行信息字典 - 从envs.step()返回的infos收集，包含任务相关的详细信息',
        'distance_to_goal': '📍 到目标距离 - 智能体当前位置到目标的欧几里得距离（米）',
        'distance_to_goal_reward': '📏 距离改善奖励 - 距离变化量(米)，正值=靠近目标，负值=远离目标，公式:-(当前距离-上步距离)',
        'success': '✅ 任务成功标志 - Boolean值，True表示成功到达目标或完成任务',
        'spl': '📈 SPL指标 - Success weighted by Path Length，成功率×最短路径长度/实际路径长度',
        'softspl': '📊 软SPL指标 - 考虑部分成功的SPL变体',
        'num_steps': '👣 Episode步数 - 当前episode已执行的动作步数',
        'collisions': '💥 碰撞统计 - 记录智能体与环境的碰撞次数和类型',
        'did_multi_agents_collide': '💥 多智能体碰撞检测 - 检测两个智能体是否碰撞，0.0=无碰撞，1.0=发生碰撞',
        'composite_reward': '🎁 综合奖励值，结合多个奖励组件',
        'force_terminate': '🛑 是否强制终止episode的标志',
        
        # === Episode和场景信息 ===
        'episode_stats': '📈 Episode统计摘要 - 包含reward和extract_scalars_from_info()提取的所有标量指标',
        'running_episode_stats': '📊 运行时统计信息 - 从eval_data_collection收集，包含轨迹数据和训练过程监控信息，可选保存',
        'scene_id': '🏠 3D场景标识 - 指定episode使用的Habitat场景文件路径',
        'episode_id': '🎬 Episode唯一ID - 在数据集中唯一标识一个episode实例',
        
        # === GPS和导航数据 ===
        'agent_0_pointgoal_with_gps_compass': '🧭 智能体GPS导航 - 从batch收集，包含相对目标的GPS坐标(x,y)，float32格式',
        'pointgoal_with_gps_compass': '🧭 目标点GPS信息 - 智能体相对于目标点的GPS坐标和方向信息',
        
        # === 检查点和训练相关 ===
        'checkpoint_index': '🔢 模型检查点索引 - 对应训练的update次数，用于标识模型版本和数据版本',
        'update_count': '🔄 更新计数 - 等同于checkpoint_index，表示训练更新次数',
        'num_envs': '🌍 环境数量 - 并行运行的环境实例数量',
        'total_steps': '📏 总步数 - 所有环境累计执行的步数',
        'trajectory': '🛤️ 轨迹数据 - 包含每步的action、position、heading等信息，用于路径分析'
    }
    
    # 尝试精确匹配
    if name in explanations:
        return explanations[name]
    
    # 尝试部分匹配
    for key, explanation in explanations.items():
        if key in name.lower() or name.lower() in key:
            return explanation
    
    # 根据数据类型提供通用解释
    if data_type == 'numpy.ndarray':
        return '🔢 数值数组数据'
    elif data_type == 'dict':
        return '📁 字典结构数据'
    elif data_type in ['list', 'tuple']:
        return '📋 序列数据'
    else:
        return '📄 其他类型数据'

def analyze_data_structure(data: Any, name: str, indent: int = 0) -> None:
    """
    分析数据结构并提供解释说明
    
    Args:
        data: 要分析的数据
        name: 数据名称
        indent: 缩进级别
    """
    prefix = "  " * indent
    
    if isinstance(data, dict):
        explanation = get_data_explanation(name, 'dict')
        print(f"{prefix}📁 {name}: dict (包含 {len(data)} 个键)")
        print(f"{prefix}   💡 说明: {explanation}")
        for key, value in data.items():
            analyze_data_structure(value, key, indent + 1)
    
    elif isinstance(data, (list, tuple)):
        explanation = get_data_explanation(name, type(data).__name__)
        print(f"{prefix}📋 {name}: {type(data).__name__} (长度: {len(data)})")
        print(f"{prefix}   💡 说明: {explanation}")
        if len(data) > 0:
            print(f"{prefix}   第一个元素类型: {type(data[0])}")
            if hasattr(data[0], 'shape'):
                print(f"{prefix}   第一个元素形状: {data[0].shape}")
    
    elif isinstance(data, np.ndarray):
        explanation = get_data_explanation(name, 'numpy.ndarray')
        print(f"{prefix}🔢 {name}: numpy.ndarray")
        print(f"{prefix}   💡 说明: {explanation}")
        print(f"{prefix}   形状: {data.shape}")
        print(f"{prefix}   数据类型: {data.dtype}")
        if data.size > 0:
            print(f"{prefix}   数值范围: [{np.min(data):.3f}, {np.max(data):.3f}]")
            if len(data.shape) > 0 and data.shape[0] > 0:
                print(f"{prefix}   第一个元素: {data[0] if data.ndim == 1 else 'shape=' + str(data[0].shape)}")
    
    else:
        explanation = get_data_explanation(name, type(data).__name__)
        print(f"{prefix}📄 {name}: {type(data).__name__} = {data}")
        print(f"{prefix}   💡 说明: {explanation}")

def analyze_episode_summary(other_data: Dict[str, Any]) -> None:
    """
    分析episode的总结信息
    
    Args:
        other_data: other_data.pkl中的数据
    """
    print("\n" + "="*60)
    print("📊 EPISODE 总结信息")
    print("="*60)
    
    if 'episode_stats' in other_data:
        stats = other_data['episode_stats']
        print("🏆 Episode统计:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    if 'info_data' in other_data and other_data['info_data'] is not None:
        info_data = other_data['info_data']
        
        if isinstance(info_data, dict):
            # 新格式：info_data是字典，每个键对应一个numpy数组
            print("\n📋 info_data结构 (numpy数组格式):")
            for key, value in info_data.items():
                if isinstance(value, np.ndarray):
                    print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                    if len(value) > 0:
                        print(f"      最后一帧值: {value[-1]}")
                        print(f"      数值范围: [{np.min(value):.3f}, {np.max(value):.3f}]")
                else:
                    print(f"   {key}: {type(value)} = {value}")
        elif isinstance(info_data, list) and len(info_data) > 0:
            # 旧格式：info_data是字典列表
            final_info = info_data[-1]  # 最后一帧的info
            print("\n🎯 最后一帧信息:")
            if isinstance(final_info, dict):
                analyze_data_structure(final_info, "最后一帧info", 1)
            else:
                print(f"   类型: {type(final_info)}, 值: {final_info}")
            
            # 显示info_data的整体结构
            print("\n📋 info_data整体结构:")
            print(f"   总帧数: {len(info_data)}")
            if len(info_data) > 0:
                print("   第一帧info结构:")
                analyze_data_structure(info_data[0], "第一帧info", 1)
        else:
            print("\n📋 info_data为空或格式未知")
            print(f"   类型: {type(info_data)}")
            if hasattr(info_data, '__len__'):
                print(f"   长度: {len(info_data)}")
    
    # 分析奖励序列
    if 'rewards' in other_data:
        rewards = other_data['rewards']
        if isinstance(rewards, np.ndarray) and len(rewards) > 0:
            print(f"\n💰 奖励分析:")
            print(f"   总奖励: {np.sum(rewards):.3f}")
            print(f"   平均奖励: {np.mean(rewards):.3f}")
            print(f"   最终奖励: {rewards[-1]:.3f}")
            print(f"   奖励范围: [{np.min(rewards):.3f}, {np.max(rewards):.3f}]")

# 可视化函数已移除，请使用 generate_episode_video.py 生成视频

def analyze_episode(base_dir: str, episode_filename: str, visualize: bool = True) -> None:
    """
    分析一个episode的完整数据
    
    Args:
        base_dir: 基础数据目录
        episode_filename: episode文件名
        visualize: 是否显示可视化
    """
    print("="*80)
    print(f"🔍 分析Episode: {episode_filename}")
    print(f"📁 数据目录: {base_dir}")
    print("="*80)
    
    # 加载数据
    episode_data = load_episode_data(base_dir, episode_filename)
    
    # 分析每个文件的数据结构
    for folder_name, data in episode_data.items():
        print(f"\n{'='*60}")
        print(f"📂 {folder_name.upper()} 数据结构分析")
        print(f"{'='*60}")
        
        if data is not None:
            analyze_data_structure(data, folder_name)
        else:
            print("❌ 数据为空或加载失败")
    
    # 分析episode总结
    if episode_data.get('other_data') is not None:
        analyze_episode_summary(episode_data['other_data'])
    
    # 可视化已移除，如需生成视频请使用 generate_episode_video.py
    
    print("\n" + "="*80)
    print("✅ 分析完成")
    print("="*80)

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
        analyze_episode(base_dir, episode_name)
    else:
        # 列出所有可用的episode
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
            
            # 分析第一个episode
            print(f"\n🎯 自动分析第一个episode: {episodes[0]}")
            analyze_episode(base_dir, episodes[0])
        else:
            print(f"❌ 在 {base_dir} 中未找到任何episode数据")
            print("\n使用方法:")
            print(f"  python {sys.argv[0]} <episode_name> [base_dir]")
            print(f"  例如: python {sys.argv[0]} 2azQ1b91cZZ_ep000001")