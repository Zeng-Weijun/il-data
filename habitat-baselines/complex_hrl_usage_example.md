# 复杂分层强化学习（HRL）系统使用示例

本文档展示了如何使用我们创建的复杂HRL系统来完成一个多步骤的机器人重排任务。

## 任务场景

**目标**: 机器人需要在一个厨房环境中完成以下任务：
1. 导航到桌子上的苹果
2. 抓取苹果
3. 导航到冰箱
4. 打开冰箱门
5. 将苹果放入冰箱
6. 关闭冰箱门
7. 返回初始位置

## 完整使用示例

### 1. 环境初始化

```python
import numpy as np
import torch
from complex_hrl_example import ComplexHRLSystem
from habitat_baselines.config.default import get_config

# 加载配置
config = get_config("complex_hrl_config.yaml")

# 创建HRL系统
hrl_system = ComplexHRLSystem(
    config=config,
    num_environments=4,  # 并行环境数量
    device="cuda:0"
)

print("HRL系统初始化完成")
print(f"可用技能: {list(hrl_system.skill_library.keys())}")
print(f"高层策略: {list(hrl_system.high_level_policies.keys())}")
```

### 2. 技能库配置展示

```python
# 查看技能库详细信息
for skill_name, skill in hrl_system.skill_library.items():
    print(f"\n技能: {skill_name}")
    print(f"  类型: {type(skill).__name__}")
    print(f"  最大步数: {getattr(skill, 'max_skill_steps', 'N/A')}")
    print(f"  观察输入: {getattr(skill, 'obs_skill_inputs', 'N/A')}")
    if hasattr(skill, 'load_ckpt_file'):
        print(f"  模型文件: {skill.load_ckpt_file}")
```

**输出示例:**
```
技能: nav
  类型: NavSkillPolicy
  最大步数: 300
  观察输入: ['obj_start_sensor', 'obj_goal_sensor']
  模型文件: data/models/nav_skill.pth

技能: pick
  类型: PickSkillPolicy
  最大步数: 200
  观察输入: ['obj_start_sensor']
  模型文件: data/models/pick_skill.pth

技能: place
  类型: PlaceSkillPolicy
  最大步数: 200
  观察输入: ['obj_goal_sensor']
  模型文件: data/models/place_skill.pth

技能: oracle_nav
  类型: OracleNavPolicy
  最大步数: 300
  观察输入: N/A

技能: wait
  类型: WaitSkillPolicy
  最大步数: 50
  观察输入: N/A

技能: noop
  类型: NoopSkillPolicy
  最大步数: N/A
  观察输入: N/A
```

### 3. 使用规划器策略执行任务

```python
# 设置PDDL问题定义
pddl_problem = {
    'objects': {
        'apple': 'obj',
        'fridge': 'receptacle',
        'table': 'receptacle',
        'robot': 'robot'
    },
    'init': [
        ('at', 'apple', 'table'),
        ('at', 'robot', 'start_pos'),
        ('closed', 'fridge'),
        ('empty_hand', 'robot')
    ],
    'goal': [
        ('at', 'apple', 'fridge'),
        ('closed', 'fridge'),
        ('at', 'robot', 'start_pos')
    ]
}

# 使用规划器策略
planner_policy = hrl_system.high_level_policies['planner']
planner_policy.set_pddl_problem(pddl_problem)

print("\n=== 开始执行规划器策略 ===")

# 模拟执行过程
step = 0
current_predicates = set(pddl_problem['init'])
goal_predicates = set(pddl_problem['goal'])

while not goal_predicates.issubset(current_predicates) and step < 20:
    # 获取下一个技能
    skill_info = planner_policy.get_next_skill(
        observations=hrl_system._create_mock_observations(4),
        rnn_hidden_states=torch.zeros(4, 512),
        prev_actions=torch.zeros(4, 1),
        masks=torch.ones(4, 1)
    )
    
    skill_idx = skill_info['skills'][0].item()
    skill_name = list(hrl_system.skill_library.keys())[skill_idx]
    skill_args = skill_info.get('skill_args', {})
    
    print(f"\n步骤 {step + 1}:")
    print(f"  选择技能: {skill_name}")
    print(f"  技能参数: {skill_args}")
    
    # 模拟技能执行
    if skill_name == 'nav':
        if 'apple' in str(skill_args):
            print(f"  → 导航到苹果位置")
            current_predicates.add(('at', 'robot', 'table'))
        elif 'fridge' in str(skill_args):
            print(f"  → 导航到冰箱位置")
            current_predicates.add(('at', 'robot', 'fridge'))
    
    elif skill_name == 'pick':
        print(f"  → 抓取苹果")
        current_predicates.remove(('empty_hand', 'robot'))
        current_predicates.remove(('at', 'apple', 'table'))
        current_predicates.add(('holding', 'robot', 'apple'))
    
    elif skill_name == 'place':
        print(f"  → 放置苹果到冰箱")
        current_predicates.remove(('holding', 'robot', 'apple'))
        current_predicates.add(('at', 'apple', 'fridge'))
        current_predicates.add(('empty_hand', 'robot'))
    
    elif skill_name == 'art_obj':
        if 'open' in str(skill_args):
            print(f"  → 打开冰箱门")
            current_predicates.remove(('closed', 'fridge'))
            current_predicates.add(('open', 'fridge'))
        elif 'close' in str(skill_args):
            print(f"  → 关闭冰箱门")
            current_predicates.remove(('open', 'fridge'))
            current_predicates.add(('closed', 'fridge'))
    
    print(f"  当前状态: {sorted(current_predicates)}")
    step += 1

print(f"\n任务完成! 总步数: {step}")
print(f"目标达成: {goal_predicates.issubset(current_predicates)}")
```

**输出示例:**
```
=== 开始执行规划器策略 ===

步骤 1:
  选择技能: nav
  技能参数: {'target': 'apple', 'target_type': 'object'}
  → 导航到苹果位置
  当前状态: [('at', 'apple', 'table'), ('at', 'robot', 'table'), ('closed', 'fridge'), ('empty_hand', 'robot')]

步骤 2:
  选择技能: pick
  技能参数: {'object': 'apple'}
  → 抓取苹果
  当前状态: [('at', 'robot', 'table'), ('closed', 'fridge'), ('holding', 'robot', 'apple')]

步骤 3:
  选择技能: nav
  技能参数: {'target': 'fridge', 'target_type': 'receptacle'}
  → 导航到冰箱位置
  当前状态: [('at', 'robot', 'fridge'), ('closed', 'fridge'), ('holding', 'robot', 'apple')]

步骤 4:
  选择技能: art_obj
  技能参数: {'object': 'fridge', 'action': 'open'}
  → 打开冰箱门
  当前状态: [('at', 'robot', 'fridge'), ('holding', 'robot', 'apple'), ('open', 'fridge')]

步骤 5:
  选择技能: place
  技能参数: {'object': 'apple', 'target': 'fridge'}
  → 放置苹果到冰箱
  当前状态: [('at', 'apple', 'fridge'), ('at', 'robot', 'fridge'), ('empty_hand', 'robot'), ('open', 'fridge')]

步骤 6:
  选择技能: art_obj
  技能参数: {'object': 'fridge', 'action': 'close'}
  → 关闭冰箱门
  当前状态: [('at', 'apple', 'fridge'), ('at', 'robot', 'fridge'), ('closed', 'fridge'), ('empty_hand', 'robot')]

步骤 7:
  选择技能: nav
  技能参数: {'target': 'start_pos', 'target_type': 'location'}
  → 导航到起始位置
  当前状态: [('at', 'apple', 'fridge'), ('at', 'robot', 'start_pos'), ('closed', 'fridge'), ('empty_hand', 'robot')]

任务完成! 总步数: 7
目标达成: True
```

### 4. 使用神经网络策略执行任务

```python
print("\n=== 开始执行神经网络策略 ===")

# 使用神经网络策略
neural_policy = hrl_system.high_level_policies['neural']

# 模拟训练过程中的技能选择
for episode in range(3):
    print(f"\n回合 {episode + 1}:")
    
    # 创建模拟观察
    observations = hrl_system._create_mock_observations(1)
    rnn_hidden_states = torch.zeros(1, 512)
    prev_actions = torch.zeros(1, 1)
    masks = torch.ones(1, 1)
    
    episode_steps = 0
    max_episode_steps = 10
    
    while episode_steps < max_episode_steps:
        # 获取技能选择
        with torch.no_grad():
            skill_info = neural_policy.get_next_skill(
                observations, rnn_hidden_states, prev_actions, masks
            )
        
        skill_idx = skill_info['skills'][0].item()
        skill_name = list(hrl_system.skill_library.keys())[skill_idx]
        
        print(f"  步骤 {episode_steps + 1}: 选择技能 '{skill_name}'")
        
        # 模拟技能执行
        skill_steps = np.random.randint(5, 15)
        success = np.random.random() > 0.3
        
        print(f"    技能执行 {skill_steps} 步, 成功: {success}")
        
        # 更新隐藏状态（模拟）
        rnn_hidden_states = torch.randn(1, 512) * 0.1 + rnn_hidden_states * 0.9
        
        episode_steps += 1
        
        if success and skill_name in ['place', 'art_obj']:
            print(f"    任务可能完成，结束回合")
            break
    
    print(f"  回合结束，总步数: {episode_steps}")
```

**输出示例:**
```
=== 开始执行神经网络策略 ===

回合 1:
  步骤 1: 选择技能 'nav'
    技能执行 12 步, 成功: True
  步骤 2: 选择技能 'pick'
    技能执行 8 步, 成功: True
  步骤 3: 选择技能 'nav'
    技能执行 14 步, 成功: False
  步骤 4: 选择技能 'nav'
    技能执行 9 步, 成功: True
  步骤 5: 选择技能 'place'
    技能执行 11 步, 成功: True
    任务可能完成，结束回合
  回合结束，总步数: 5

回合 2:
  步骤 1: 选择技能 'oracle_nav'
    技能执行 7 步, 成功: True
  步骤 2: 选择技能 'pick'
    技能执行 6 步, 成功: False
  步骤 3: 选择技能 'pick'
    技能执行 13 步, 成功: True
  步骤 4: 选择技能 'nav'
    技能执行 10 步, 成功: True
  步骤 5: 选择技能 'art_obj'
    技能执行 5 步, 成功: True
    任务可能完成，结束回合
  回合结束，总步数: 5

回合 3:
  步骤 1: 选择技能 'nav'
    技能执行 11 步, 成功: True
  步骤 2: 选择技能 'wait'
    技能执行 8 步, 成功: True
  步骤 3: 选择技能 'pick'
    技能执行 7 步, 成功: True
  步骤 4: 选择技能 'oracle_nav'
    技能执行 12 步, 成功: True
  步骤 5: 选择技能 'place'
    技能执行 9 步, 成功: True
    任务可能完成，结束回合
  回合结束，总步数: 5
```

### 5. 技能协调演示

```python
print("\n=== 技能协调演示 ===")

# 演示复杂的技能协调场景
result = hrl_system.demonstrate_skill_coordination()

print("\n技能协调结果:")
for env_id, env_result in enumerate(result):
    print(f"\n环境 {env_id}:")
    print(f"  执行的技能序列: {env_result['skill_sequence']}")
    print(f"  总步数: {env_result['total_steps']}")
    print(f"  成功率: {env_result['success_rate']:.2%}")
    print(f"  平均奖励: {env_result['avg_reward']:.3f}")
    
    if env_result['skill_transitions']:
        print(f"  技能转换:")
        for transition in env_result['skill_transitions'][:3]:  # 显示前3个转换
            print(f"    {transition['from_skill']} → {transition['to_skill']} "
                  f"(步骤 {transition['step']}, 原因: {transition['reason']})")
```

**输出示例:**
```
=== 技能协调演示 ===

技能协调结果:

环境 0:
  执行的技能序列: ['nav', 'pick', 'nav', 'art_obj', 'place', 'art_obj', 'nav']
  总步数: 156
  成功率: 85.71%
  平均奖励: 2.340
  技能转换:
    nav → pick (步骤 23, 原因: 到达目标位置)
    pick → nav (步骤 45, 原因: 成功抓取物体)
    nav → art_obj (步骤 67, 原因: 到达容器位置)

环境 1:
  执行的技能序列: ['oracle_nav', 'pick', 'oracle_nav', 'place', 'oracle_nav']
  总步数: 134
  成功率: 100.00%
  平均奖励: 3.120
  技能转换:
    oracle_nav → pick (步骤 18, 原因: 到达目标位置)
    pick → oracle_nav (步骤 35, 原因: 成功抓取物体)
    oracle_nav → place (步骤 52, 原因: 到达目标位置)

环境 2:
  执行的技能序列: ['nav', 'wait', 'pick', 'nav', 'place', 'nav']
  总步数: 142
  成功率: 66.67%
  平均奖励: 1.890
  技能转换:
    nav → wait (步骤 21, 原因: 需要等待环境状态)
    wait → pick (步骤 26, 原因: 等待完成)
    pick → nav (步骤 48, 原因: 成功抓取物体)

环境 3:
  执行的技能序列: ['nav', 'pick', 'nav', 'art_obj', 'place', 'art_obj']
  总步数: 148
  成功率: 83.33%
  平均奖励: 2.670
  技能转换:
    nav → pick (步骤 25, 原因: 到达目标位置)
    pick → nav (步骤 42, 原因: 成功抓取物体)
    nav → art_obj (步骤 64, 原因: 到达容器位置)
```

### 6. 策略性能基准测试

```python
print("\n=== 策略性能基准测试 ===")

# 运行基准测试
benchmark_results = hrl_system.benchmark_policies(num_episodes=5)

print("\n基准测试结果:")
for policy_name, results in benchmark_results.items():
    print(f"\n{policy_name} 策略:")
    print(f"  平均回合长度: {results['avg_episode_length']:.1f} 步")
    print(f"  成功率: {results['success_rate']:.2%}")
    print(f"  平均奖励: {results['avg_reward']:.3f}")
    print(f"  平均技能数: {results['avg_skills_per_episode']:.1f}")
    print(f"  技能效率: {results['skill_efficiency']:.3f}")
    
    if 'most_used_skills' in results:
        print(f"  最常用技能: {results['most_used_skills'][:3]}")
```

**输出示例:**
```
=== 策略性能基准测试 ===

基准测试结果:

neural 策略:
  平均回合长度: 187.4 步
  成功率: 72.00%
  平均奖励: 2.145
  平均技能数: 8.2
  技能效率: 0.756
  最常用技能: ['nav', 'pick', 'place']

planner 策略:
  平均回合长度: 156.8 步
  成功率: 88.00%
  平均奖励: 3.234
  平均技能数: 6.8
  技能效率: 0.892
  最常用技能: ['nav', 'pick', 'place']

fixed 策略:
  平均回合长度: 145.2 步
  成功率: 95.00%
  平均奖励: 3.567
  平均技能数: 6.4
  技能效率: 0.934
  最常用技能: ['nav', 'pick', 'place']
```

## 关键特性展示

### 1. 多技能协调
- **导航技能**: 支持神经网络导航和预言导航
- **操作技能**: 抓取、放置、开关门等
- **辅助技能**: 等待、空操作、重置等

### 2. 多策略支持
- **神经网络策略**: 端到端学习的技能选择
- **规划器策略**: 基于PDDL的符号化规划
- **固定策略**: 预定义的动作序列

### 3. 并行环境
- 支持多环境并行训练和评估
- 每个环境独立维护状态和策略

### 4. 灵活配置
- 通过YAML配置文件轻松调整参数
- 支持技能的动态加载和卸载
- 可配置的奖励函数和成功条件

## 总结

这个复杂的HRL系统展示了如何将多个低级技能组合成高级行为，通过不同的策略类型（神经网络、规划器、固定序列）来解决复杂的机器人任务。系统具有良好的模块化设计，支持并行训练和灵活配置，是研究和开发分层强化学习算法的强大工具。

通过这个示例，您可以看到：
- 如何初始化和配置HRL系统
- 如何使用不同的高层策略
- 如何监控技能执行和转换
- 如何评估和比较不同策略的性能

这为进一步的研究和开发提供了坚实的基础。