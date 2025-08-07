# Falcon 模仿学习数据格式说明

## 概述

本文档详细说明了 Falcon 项目中模仿学习数据的格式、结构和分析脚本的使用方法。

## 数据结构概览

Falcon模仿学习数据采用分层存储结构，由`FALCONEvaluator`类在评估过程中自动生成。数据收集流程如下：

1. **数据收集阶段**：在`evaluate_agent()`函数中，通过`envs.step()`循环收集每个时间步的数据
2. **数据处理阶段**：在episode结束时，调用`_save_imitation_learning_data()`进行数据转换和保存
3. **存储结构**：按时间戳创建主目录，数据分类存储在四个子目录中

```
falcon_imitation_data/
├── 20250807_164151/          # 时间戳目录（格式：YYYYMMDD_HHMMSS）
│   ├── jaw_rgb_data/         # 机械臂末端RGB图像序列
│   ├── jaw_depth_data/       # 机械臂末端深度图像序列
│   ├── topdown_map/          # 环境俯视图地图数据
│   └── other_data/           # 动作、奖励、状态等核心训练数据
```

## 详细数据格式说明

### 1. jaw_rgb_data/ - 机械臂末端RGB图像序列

**数据来源**: `batch["agent_0_articulated_agent_jaw_rgb"][i]` (falcon_evaluator.py:285)
**收集方式**: 每个时间步从观测batch中提取，按环境索引分别收集
**用途**: 🎥 机械臂末端视觉感知，物体识别，精细操作引导

**格式**: 
- 文件类型: `.pkl` (pickle)
- 数据结构: `dict`
- 内容: 包含传感器名称为键的字典，值为 numpy 数组
- 数组形状: `(时间步数, 高度, 宽度, 3)` 
- 数据类型: `uint8` (0-255)
- 颜色通道: RGB 格式

**数据处理**:
```python
rgb_data = batch['agent_0_articulated_agent_jaw_rgb'][i].cpu().numpy()
if rgb_data.dtype != np.uint8:
    rgb_data = (rgb_data * 255).astype(np.uint8)
```

### 2. jaw_depth_data/ - 机械臂末端深度图像序列

**数据来源**: `batch["agent_0_articulated_agent_jaw_depth"][i]` (falcon_evaluator.py:293)
**收集方式**: 每个时间步从观测batch中提取，自动添加深度维度
**用途**: 📏 距离感知，3D空间理解，精确抓取定位

**格式**:
- 文件类型: `.pkl` (pickle)
- 数据结构: `dict`
- 内容: 包含传感器名称为键的字典，值为 numpy 数组
- 数组形状: `(时间步数, 高度, 宽度, 1)` 或 `(时间步数, 高度, 宽度)`
- 数据类型: `float32`
- 数值含义: 每个像素表示到物体的距离（米）

**数据处理**:
```python
depth_data = batch['agent_0_articulated_agent_jaw_depth'][i].cpu().numpy()
if depth_data.ndim == 2:
    depth_data = np.expand_dims(depth_data, axis=-1)
depth_data = depth_data.astype(np.float32)
```

### 3. topdown_map/ - 环境俯视图地图数据

**数据来源**: 环境渲染的鸟瞰视图
**收集方式**: 通过环境的topdown传感器获取，每个时间步收集一次
**用途**: 🗺️ 全局导航，路径规划，环境布局理解，agent位置追踪

**文件类型**: `.pkl` (pickle)
**数据结构**: `List[Dict]` - 字典列表，每个字典包含一个时间步的完整topdown信息

**字典内部结构**:
- `map`: `numpy.ndarray` - 环境地图的俯视图
  - 形状: `(高度, 宽度)` 通常为 `(256, 303)` 或类似尺寸
  - 数据类型: `uint8`
  - 数值范围: `[0-44]` 表示不同的地图元素（墙壁、地面、物体等）
- `fog_of_war_mask`: `numpy.ndarray` - 战争迷雾遮罩
  - 形状: 与map相同
  - 数据类型: `bool`
  - 用途: 标记agent已探索和未探索的区域
- `agent_map_coord`: `numpy.ndarray` - agent在地图中的坐标
  - 形状: `(2,)` 表示 `[x, y]` 坐标
  - 数据类型: `float32`
  - 用途: agent在topdown地图中的精确位置
- `agent_angle`: `float` - agent的朝向角度
  - 数据类型: `float32`
  - 单位: 弧度
  - 用途: agent在地图中的朝向信息

**数据处理示例**:
```python
# 加载topdown数据
with open('topdown_map.pkl', 'rb') as f:
    topdown_data = pickle.load(f)

# 访问第一帧的数据
first_frame = topdown_data[0]
map_data = first_frame['map']  # 地图数据
fog_mask = first_frame['fog_of_war_mask']  # 迷雾遮罩
agent_pos = first_frame['agent_map_coord']  # agent位置
agent_angle = first_frame['agent_angle']  # agent角度
```

### 4. other_data/ - 核心训练数据

**用途**: 📊 包含动作、奖励、状态等训练相关的核心数据
**文件格式**: pickle (.pkl)
**数据类型**: 字典 (dict)
**生成位置**: `_save_imitation_learning_data()` 函数 (falcon_evaluator.py:696-921)

**主要字段**:

#### 4.1 actions - 智能体动作序列
**数据来源**: `eval_data_collection['actions'][-1][i]` (falcon_evaluator.py:318)
**收集方式**: 从`action_data.env_actions`中按环境索引提取
- **用途**: 🎮 机器人在每个时间步执行的动作序列，用于模仿学习训练
- **格式**: `numpy.ndarray`
- **形状**: `(时间步数,)` 或 `(时间步数, 动作维度)`
- **数据类型**: `int64`
- **数据处理**:
```python
action_data = eval_data_collection['actions'][-1][i].cpu().numpy()
if action_data.ndim == 0:
    action_data = np.array([action_data], dtype=np.int64)
```

#### 4.2 rewards - 即时奖励序列
**数据来源**: `rewards_l[i]` from `envs.step()` (falcon_evaluator.py:323)
**收集方式**: 每个时间步从环境返回的奖励中提取
- **用途**: 💰 每个时间步获得的奖励值，用于强化学习训练
- **格式**: `numpy.ndarray`
- **形状**: `(时间步数,)`
- **数据类型**: `float64`
- **数据处理**: `np.float64(reward_value)`
- **含义**: 正值表示好的行为，负值表示不良行为

#### 4.3 masks - Episode继续掩码
**数据来源**: `dones[i]` from `envs.step()` (falcon_evaluator.py:326)
**收集方式**: 基于episode结束标志计算
- **用途**: 🎭 episode结束标志，1表示继续，0表示结束
- **格式**: `numpy.ndarray`
- **形状**: `(时间步数,)`
- **数据类型**: `float64`
- **数据处理**: `np.float64(not dones[i])`
- **含义**: 1.0 = 继续，0.0 = episode 结束

#### 4.4 info_data - Episode执行信息
**数据来源**: `infos[i]` from `envs.step()` (falcon_evaluator.py:329)
**收集方式**: 每个时间步复制完整的info字典
- **用途**: 📋 每个时间步的详细状态信息和任务相关指标
- **原始格式**: 字典列表 `[{}, {}, ...]`
- **转换格式**: numpy数组字典 `{key: np.array([...])}`

**转换逻辑** (falcon_evaluator.py:780-820):
```python
# 提取所有唯一的键
all_keys = set()
for info in info_data_list:
    if isinstance(info, dict):
        all_keys.update(info.keys())

# 按键转换为numpy数组
for key in all_keys:
    values = [info.get(key) if isinstance(info, dict) else None 
              for info in info_data_list]
    if all(isinstance(v, (int, float, bool)) for v in values if v is not None):
        info_arrays[key] = np.array(values, dtype=np.float64)
    elif all(isinstance(v, np.ndarray) for v in values if v is not None):
        info_arrays[key] = np.stack(values, axis=0)
```

**info_data 包含的指标**:

| 字段名 | 用途 | 数据类型 | 说明 |
|--------|------|----------|------|
| `distance_to_goal` | 🎯 到目标距离 | float64 | 机器人到目标位置的欧几里得距离 |
| `success` | ✅ 成功标志 | float64 | 任务是否成功完成 (0.0/1.0) |
| `spl` | 📊 SPL 指标 | float64 | Success weighted by Path Length |
| `softspl` | 📊 软 SPL | float64 | 软化版本的 SPL 指标 |
| `num_steps` | 👣 步数统计 | float64 | 当前 episode 已执行的步数 |
| `collisions` | 💥 碰撞信息 | dict | 碰撞相关的统计信息 |
| `composite_reward` | 🎁 综合奖励 | float64 | 结合多个奖励组件的综合值 |
| `force_terminate` | 🛑 强制终止 | float64 | 是否强制终止 episode |

#### 4.5 agent_0_pointgoal_with_gps_compass - GPS导航数据
**数据来源**: `batch['agent_0_pointgoal_with_gps_compass'][i]` (falcon_evaluator.py:303)
**收集方式**: 每个时间步从观测batch中提取GPS坐标
- **用途**: 🧭 智能体相对于目标的GPS坐标，用于导航任务
- **格式**: `numpy.ndarray`
- **形状**: `(时间步数, 2)` - 只保存前两个坐标(x,y)
- **数据类型**: `float32`
- **数据处理**:
```python
gps_compass = batch[gps_key][i].cpu().numpy().astype(np.float32)
if len(gps_compass) >= 2:
    imitation_learning_data['other_data']['pointgoal_with_gps_compass'][i].append(gps_compass[:2])
```

#### 4.6 episode_stats - Episode统计摘要
**数据来源**: `extract_scalars_from_info(infos[i])` (falcon_evaluator.py:408)
**生成时机**: episode结束时计算
- **用途**: 🏆 整个episode的最终统计结果和性能指标
- **格式**: `dict`
- **包含字段**:
  - `reward`: 总累积奖励
  - 从最终info中提取的所有标量指标
- **内容**: episode 结束时的汇总统计信息

#### 4.7 全局评估数据（如果提供eval_data_collection）

##### global_actions - 全局动作数据
**数据来源**: `eval_data_collection['actions']` (falcon_evaluator.py:845)
- **数据形状**: `(时间步数, 环境数, 动作维度)`
- **数据处理**: `torch.stack(actions_tensor, dim=0).cpu().numpy()`

##### global_rewards - 全局奖励汇总
**数据来源**: `eval_data_collection['rewards']` (falcon_evaluator.py:856)
- **数据形状**: `(episode数,)`
- **内容**: 每个episode的total_reward

##### global_masks - 全局掩码数据
**数据来源**: `eval_data_collection['masks']` (falcon_evaluator.py:869)

#### 4.8 running_episode_stats - 运行时统计（可选）
**数据来源**: `eval_data_collection['running_episode_stats']` (falcon_evaluator.py:876)
- **用途**: 📈 训练过程中的运行时统计信息
- **格式**: `dict` 或 `None`
- **内容**: 包含轨迹数据等训练过程监控信息
- **说明**: 包含训练过程中的中间统计，**可以选择不保存以节省存储空间**

#### 4.9 元数据字段
- `scene_id`: 🏠 Habitat场景文件路径标识
- `episode_id`: 🆔 数据集中的episode唯一标识
- `checkpoint_index`: 💾 模型检查点索引，等同于训练update次数
- `num_steps`: 👣 总训练步数
- `num_envs`: 🌍 并行环境数量

## 数据分析脚本使用方法

### 脚本功能

`analyze_episode_data.py` 脚本提供以下功能：
1. 📊 自动检测最新数据目录
2. 🔍 详细的数据结构分析
3. 💡 数据字段含义解释
4. 📈 奖励和性能指标分析
5. 🖼️ 第一帧图像可视化
6. 📋 Episode 列表和统计

### 使用方法

#### 1. 🚀 自动分析最新数据
```bash
# 自动检测并分析最新数据
python analyze_episode_data.py
```
**功能说明**:
- 🎯 自动检测最新的数据目录
- 📋 列出所有可用的 episode
- 🔬 自动分析第一个 episode

#### 2. 🎯 分析指定 episode
```bash
# 分析特定的 episode 数据
python analyze_episode_data.py <episode_name>
```
**使用示例**:
```bash
# 分析场景 2azQ1b91cZZ 的第一个 episode
python analyze_episode_data.py 2azQ1b91cZZ_ep000001
```

#### 3. 📁 指定数据目录
```bash
# 从指定目录分析 episode 数据
python analyze_episode_data.py <episode_name> <data_directory>
```
**使用示例**:
```bash
# 从特定时间戳目录分析数据
python analyze_episode_data.py 2azQ1b91cZZ_ep000001 /path/to/falcon_imitation_data/20250807_164151
```

### 输出说明

脚本会输出以下信息：

1. **数据加载状态**: ✅ 成功 / ❌ 失败
2. **数据结构分析**: 包含形状、类型、数值范围
3. **💡 数据含义解释**: 每个字段的具体用途说明
4. **📊 Episode 统计**: 成功率、奖励、步数等
5. **📈 奖励分析**: 总奖励、平均奖励、奖励范围
6. **🖼️ 可视化**: RGB图像、深度图、俯视图

### 📊 示例输出

```
================================================================================
🔍 分析Episode: 2azQ1b91cZZ_ep000001
📁 数据目录: /root/zwj/Falcon/falcon_imitation_data/20250807_164151
================================================================================

============================================================
📂 JAW_RGB_DATA 数据结构分析
============================================================
📁 jaw_rgb_data: dict (包含 1 个键)
   💡 说明: 🎥 机器人头部摄像头的RGB图像数据，用于视觉感知
  🔢 rgb: numpy.ndarray
     💡 说明: 🎥 RGB图像序列，每帧包含彩色视觉信息
     形状: (30, 224, 224, 3)
     数据类型: uint8
     数值范围: [0.000, 255.000]
```

### 🚀 常用命令示例

#### 📋 基础使用

```bash
# 一键分析最新数据（推荐新手使用）
python analyze_episode_data.py

# 查看所有可用 episodes
ls falcon_imitation_data/*/other_data/ | head -10

# 分析指定 episode
python analyze_episode_data.py 2azQ1b91cZZ_ep000001
```

#### 🛠️ 高级用法

```bash
# 批量分析多个 episodes
for episode in $(ls falcon_imitation_data/*/other_data/*.pkl | head -3); do
  echo "分析: $episode"
  python analyze_episode_data.py $(basename $episode .pkl)
done

# 带时间戳的分析
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始数据分析" && python analyze_episode_data.py

# 保存分析结果到文件
python analyze_episode_data.py > analysis_report_$(date +%Y%m%d_%H%M%S).txt 2>&1
```

## 数据优化建议

### 关于 running_episode_stats

**建议**: 可以选择不保存 `running_episode_stats` 以节省存储空间

**原因**:
- 主要用于训练过程中的调试和监控
- 对于模仿学习的核心功能不是必需的
- 可以显著减少数据文件大小

**实现方式**: 在保存数据时将 `running_episode_stats` 设置为 `None`

### 关于 checkpoint_index

**含义**: 💾 模型检查点的索引号，用于标识训练进度

**用途**:
- 标识数据来自哪个训练阶段
- 用于数据版本管理
- 便于追踪模型性能变化
- 在评估和分析时提供时间参考

## 常见问题

### Q: 数据文件很大，如何优化？
A: 
1. 可以不保存 `running_episode_stats`
2. 考虑压缩图像数据
3. 根据需要调整图像分辨率

### Q: 如何添加新的数据字段？
A:
1. 在相应的保存函数中添加新字段
2. 在 `analyze_episode_data.py` 的 `get_data_explanation()` 函数中添加解释
3. 更新本文档

### Q: 数据格式不兼容怎么办？
A:
1. 检查数据保存和加载的代码版本
2. 使用分析脚本查看实际数据结构
3. 根据需要编写数据转换脚本

## 视频生成脚本使用方法

### 脚本功能

`generate_episode_video.py` 脚本提供以下功能：
1. 🎬 生成包含RGB、深度图和Topdown Map的同步视频
2. 📊 实时显示episode指标（奖励、距离、成功率等）
3. 🗺️ 在Topdown图上区分ego和agent的位置（如果有位置数据）
4. 🚀 自动检测最新数据目录
5. ⚙️ 可配置视频帧率和输出路径

### 使用方法

#### 1. 🚀 自动生成最新数据的视频
```bash
# 自动检测并生成第一个episode的视频
python generate_episode_video.py
```
**功能说明**:
- 🎯 自动检测最新的数据目录
- 📋 列出所有可用的 episode
- 🎬 自动生成第一个 episode 的视频

#### 2. 🎯 生成指定 episode 的视频
```bash
# 生成特定episode的视频
python generate_episode_video.py <episode_name>
```
**使用示例**:
```bash
# 生成场景 2azQ1b91cZZ 的第一个 episode 视频
python generate_episode_video.py 2azQ1b91cZZ_ep000001
```

#### 3. 📁 指定数据目录和输出路径
```bash
# 完整参数指定
python generate_episode_video.py <episode_name> <data_directory> <output_directory> <fps>
```
**使用示例**:
```bash
# 从特定目录生成视频，指定输出路径和帧率
python generate_episode_video.py 2azQ1b91cZZ_ep000001 /path/to/data /path/to/output 15
```

### 视频特性

#### 🖼️ 视频布局
- **左上角**: RGB摄像头图像 - 机械臂末端的彩色视觉
- **右上角**: 深度摄像头图像 - 距离感知信息
- **左下角**: Topdown Map - 环境俯视图（带智能体位置标记）
- **右下角**: 实时指标显示 - 奖励、距离、成功率等

#### 📊 实时指标显示
- `Frame`: 当前帧数/总帧数
- `Progress`: 播放进度百分比
- `reward`: 当前帧奖励值
- `cumulative_reward`: 累积奖励
- `distance_to_goal`: 到目标距离
- `success`: 任务成功标志
- `spl`: SPL性能指标
- `num_steps`: 执行步数
- `did_multi_agents_collide`: 智能体碰撞检测

#### 🎨 视觉增强
- **颜色区分**: Topdown图上用不同颜色标记ego和agent位置
- **缺失数据处理**: 用不同颜色的占位符表示缺失的数据类型
- **自适应显示**: 根据数据范围自动调整深度图的颜色映射

### 输出说明

#### 📁 输出文件
- **文件格式**: MP4视频文件
- **命名规则**: `{episode_name}_{timestamp}.mp4`
- **默认位置**: `{data_directory}/videos/`
- **视频质量**: 1800 bitrate，可配置帧率

#### 📊 控制台输出
```
================================================================================
🎬 Falcon Episode视频生成器
📂 数据目录: /root/zwj/Falcon/falcon_imitation_data/20250807_164151
🎯 Episode: 2azQ1b91cZZ_ep000001
================================================================================
✅ 成功加载: jaw_rgb_data/2azQ1b91cZZ_ep000001.pkl
✅ 成功加载: jaw_depth_data/2azQ1b91cZZ_ep000001.pkl
✅ 成功加载: topdown_map/2azQ1b91cZZ_ep000001.pkl
✅ 成功加载: other_data/2azQ1b91cZZ_ep000001.pkl
📊 总帧数: 30
🎥 开始渲染视频...
✅ 视频生成成功: /path/to/output/2azQ1b91cZZ_ep000001_20250107_143022.mp4
📊 视频信息: 30帧, 10fps, 时长3.0秒
```

### 🚀 常用命令示例

#### 📋 基础使用
```bash
# 一键生成最新数据视频（推荐）
python generate_episode_video.py

# 生成指定episode视频
python generate_episode_video.py 33ypawbKCQf.basis_ep000001

# 高帧率视频生成
python generate_episode_video.py 33ypawbKCQf.basis_ep000001 "" "" 20
```

#### 🛠️ 高级用法
```bash
# 批量生成多个episode视频
for episode in $(ls falcon_imitation_data/*/other_data/*.pkl | head -3); do
  echo "生成视频: $episode"
  python generate_episode_video.py $(basename $episode .pkl)
done

# 生成高质量视频（高帧率）
python generate_episode_video.py 2azQ1b91cZZ_ep000001 "" "" 30

# 指定输出目录
mkdir -p ./episode_videos
python generate_episode_video.py 2azQ1b91cZZ_ep000001 "" ./episode_videos
```



## 更新日志

- **2025-01-07**: 初始版本，包含完整的数据格式说明和分析脚本使用方法
- **2025-01-07**: 添加数据字段解释功能，优化分析脚本输出格式
- **2025-01-07**: 新增视频生成脚本，支持多视图同步显示和实时指标展示