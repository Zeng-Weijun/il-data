# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from collections import deque
from dataclasses import dataclass
from typing import List

import gym.spaces as spaces
import torch

from habitat.tasks.rearrange.multi_task.pddl_action import PddlAction
from habitat.tasks.rearrange.multi_task.pddl_logical_expr import LogicalExpr
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData
from queue import PriorityQueue


@dataclass
class PlanNode:
    """
    表示在搜索问题中规划通向目标的高级动作路径所需的信息。
    """

    cur_pred_state: List[Predicate]  # 当前谓词状态
    parent: "PlanNode"  # 父节点
    depth: int  # 搜索深度
    action: PddlAction  # PDDL动作


class PlannerHighLevelPolicy(HighLevelPolicy):
    """
    高级策略,用于规划对象重排序动作序列。
    'plan_idx'配置参数控制代理将执行什么样的计划。
    代理可以重新排列1个对象或两个对象。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 这必须与'GlobalPredicatesSensor'中的谓词集匹配
        self._predicates_list = self._pddl_prob.get_possible_predicates()

        # 初始化所有可能的动作
        self._all_actions = self._setup_actions()
        self._n_actions = len(self._all_actions)
        self._max_search_depth = self._config.max_search_depth
        self._reactive_planner = self._config.is_reactive

        # 初始化规划相关的张量
        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)
        self._plans: List[List[PddlAction]] = [
            [] for _ in range(self._num_envs)
        ]
        self._should_replan = torch.zeros(self._num_envs, dtype=torch.bool)
        self.gen_plan_ids_batch = torch.zeros(
            self._num_envs, dtype=torch.int32
        )
        self._plan_idx = self._config.plan_idx
        self._select_random_goal = self._config.select_random_goal
        assert self._plan_idx >= 0

        # 我们考虑4种可能的计划,索引从0到3:
        # 0 - 不移动任何对象
        # 1 - 移动第一个对象
        # 2 - 移动第二个对象
        # 3 - 移动两个对象
        # 计划存储在agent_plans数组中
        # 当select_random_goal为False时,代理执行由plan_idx索引的计划
        # 例如: final_plan = agent_plans[plan_idx]
        # 当select_random_goal为True时,代理将从plan_idx指定的子集中随机选择一个计划
        # plan_index = randint(self.low_plan[plan_idx], self.high_plan[plan_idx])
        # final_plan = agent_plans[plan_index]
        # 因此,plan_idx指定代理是否应该:
        # 总是移动相同的对象(0)
        # 随机移动两个对象中的一个(1)
        # 随机移动一个或两个对象(2)
        # 随机移动一个、两个或不移动对象(3)
        self.low_plan = [1, 1, 1, 0]
        self.high_plan = [1, 2, 3, 3]

    def filter_envs(self, curr_envs_to_keep_active):
        """
        清理策略的状态变量,使其与活动环境匹配
        """
        self._should_replan = self._should_replan[curr_envs_to_keep_active]
        self.gen_plan_ids_batch = self.gen_plan_ids_batch[
            curr_envs_to_keep_active
        ]
        self._next_sol_idxs = self._next_sol_idxs[curr_envs_to_keep_active]

    def create_hl_info(self):
        """创建高级信息字典"""
        return {"actions": None}

    def get_policy_action_space(
        self, env_action_space: spaces.Space
    ) -> spaces.Space:
        """
        获取策略的动作空间用于学习。如果我们正在学习HL策略,
        它将返回其用于学习的自定义动作空间。
        """
        return spaces.Discrete(self._n_actions)

    def apply_mask(self, mask):
        """应用掩码来决定是否需要重新规划"""
        if self._reactive_planner:
            # 每一步都重新规划
            self._should_replan = torch.ones(mask.shape[0], dtype=torch.bool)
        else:
            # 仅在步骤0时规划
            self._should_replan = ~mask.cpu().view(-1)

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        """
        获取值函数。我们分配值0。
        这是必需的,以便我们可以在多智能体策略中连接值。
        """
        return torch.zeros(rnn_hidden_states.shape[0], 1).to(
            rnn_hidden_states.device
        )
    def _get_solution_nodes(
        self, pred_vals, pddl_goal: LogicalExpr
    ) -> List[PlanNode]:
        """
        基于pred_vals中的真值,规划一系列PddlActions(以PlanNodes列表返回),
        描述如何从当前状态到达指定的pddl_goal。
        """
        assert pddl_goal is not None, "规划时必须设置PDDL目标。"

        # 当前状态下的真谓词
        start_true_preds = [
            pred
            for is_valid, pred in zip(pred_vals, self._predicates_list)
            if (is_valid == 1.0)
        ]

        def _is_pred_at(pred, robot_type):
            """检查谓词是否表示机器人在特定位置"""
            return (
                pred.name == "robot_at" and pred._arg_values[-1] == robot_type
            )

        def _get_pred_hash(preds):
            """获取谓词的哈希值"""
            return ",".join(sorted([p.compact_str for p in preds]))

        def _update_pred_set(pred_set, action):
            """更新谓词集合,处理动作效果"""
            new_pred_set = list(pred_set)
            
            # 处理导航动作
            # 如果动作名包含"nav",说明这是一个导航动作
            if "nav" in action.name:
                # 获取导航目标机器人的类型
                robot_to_nav = action._param_values[-1]
                # 从谓词集合中移除表示该机器人当前位置的谓词
                new_pred_set = [
                    pred for pred in new_pred_set
                    if not _is_pred_at(pred, robot_to_nav)
                ]

            # 处理动作的后置条件
            for p in action.post_cond:#这个post_cond
                # 如果后置条件是"holding"(持有物体)
                if p.name == "holding":
                    # 移除与之矛盾的"not_holding"谓词
                    new_pred_set = [
                        other_p for other_p in new_pred_set
                        if not (other_p.name == "not_holding" and 
                               other_p._arg_values[0] == p._arg_values[1])
                    ]
                # 如果后置条件是"not_holding"(未持有物体)
                elif p.name == "not_holding":
                    # 移除与之矛盾的"holding"谓词
                    new_pred_set = [
                        other_p for other_p in new_pred_set
                        if not (other_p.name == "holding" and
                               p._arg_values[0] == other_p._arg_values[1])
                    ]
                # 如果后置条件谓词不在当前集合中,则添加它
                if p not in new_pred_set:
                    new_pred_set.append(p)
            # 返回更新后的谓词集合
            return new_pred_set

        # 使用优先队列进行启发式搜索
        frontier = PriorityQueue()
        # 优先级为(启发值, 深度, PlanNode)
        frontier.put((0, 0, PlanNode(start_true_preds, None, 0, None)))
        visited = {_get_pred_hash(start_true_preds)}
        sol_nodes = []

        # 缓存已验证的前提条件结果
        precond_cache = {}
        
        while not frontier.empty():
            _, _, cur_node = frontier.get()

            if cur_node.depth > self._max_search_depth:
                continue

            # 如果已找到解决方案,且当前深度大于最短解,则停止搜索
            if sol_nodes and cur_node.depth >= sol_nodes[0].depth:
                break

            # 按照动作类型对动作进行分组和优先级排序
            actions = sorted(self._all_actions, 
                           key=lambda x: 1 if "nav" in x.name else 0)

            for action in actions:
                # 使用缓存检查前提条件
                state_hash = _get_pred_hash(cur_node.cur_pred_state)
                cache_key = (state_hash, action.name)
                # 检查缓存中是否已有该状态-动作对的前提条件检查结果
                if cache_key in precond_cache:
                    # 如果缓存显示该动作的前提条件不满足,跳过这个动作
                    if not precond_cache[cache_key]:
                        continue
                else:
                    is_satisfied = action.is_precond_satisfied_from_predicates(
                        cur_node.cur_pred_state
                    )
                    precond_cache[cache_key] = is_satisfied
                    if not is_satisfied:
                        continue

                # 更新谓词状态
                new_pred_set = _update_pred_set(cur_node.cur_pred_state, action)
                pred_hash = _get_pred_hash(new_pred_set)

                if pred_hash not in visited:
                    visited.add(pred_hash)
                    # 创建新的规划节点,包含:
                    # - 更新后的谓词集合
                    # - 指向父节点的指针
                    # - 增加的搜索深度
                    # - 执行的动作
                    new_node = PlanNode(
                        new_pred_set, cur_node, cur_node.depth + 1, action
                    )
                    
                    # 计算启发值(到目标的估计距离)
                    # 使用规划图启发式替代简单的后置条件计数
                    
                    # 使用动作后置条件的数量作为启发式估计
                    # 后置条件越多,说明这个动作离目标状态可能越近
                    heuristic = len(action.post_cond)



                    # heuristic = self._planning_graph_heuristic(new_pred_set, pddl_goal)
                    
                    # 检查新状态是否满足目标条件
                    if pddl_goal.is_true_from_predicates(new_pred_set):
                        # 如果满足目标,将此节点加入解决方案列表
                        sol_nodes.append(new_node)
                        # 找到解决方案后跳出当前动作循环
                        break
                    else:
                        # 如果未达到目标,将新节点加入优先队列继续搜索
                        # 优先级由启发值和深度共同决定
                        frontier.put((heuristic, new_node.depth, new_node))

        return sol_nodes

    def _extract_paths(self, sol_nodes):
        """从解决方案节点中提取路径"""
        paths = []
        for sol_node in sol_nodes:
            cur_node = sol_node
            path = []
            while cur_node.parent is not None:
                path.append(cur_node)
                cur_node = cur_node.parent
            paths.append(path[::-1])
        return paths

    def _get_plan(self, pred_vals, pddl_goal: LogicalExpr) -> List[PddlAction]:
        """
        获取计划
        :param pred_vals: 形状 (num_prds,)。非批处理。
        """
        assert len(pred_vals) == len(self._predicates_list)
        sol_nodes = self._get_solution_nodes(pred_vals, pddl_goal)

        # 提取导向目标的动作序列
        paths = self._extract_paths(sol_nodes)

        all_ac_seqs = []
        for path in paths:
            all_ac_seqs.append([node.action for node in path])
        # 按动作序列长度排序
        full_plans = sorted(all_ac_seqs, key=len)
        # 每个完整计划都是其他完整计划的排列
        plans = full_plans[0]
        return plans
########################################################################################
    def _planning_graph_heuristic(self, current_predicates, goal_expr):
        """
        规划图启发式函数实现
        
        原理：
        1. 构建放松的规划图（忽略删除效果）
        2. 计算目标谓词在图中首次出现的层级
        3. 使用最大层级作为启发值
        
        :param current_predicates: 当前状态的谓词列表
        :param goal_expr: 目标逻辑表达式
        :return: 启发值（到目标的估计距离）
        """
        # 如果目标已经满足，返回0
        if goal_expr.is_true_from_predicates(current_predicates):
            return 0
            
        # 提取目标中的所有谓词
        goal_predicates = self._extract_goal_predicates(goal_expr)
        
        # 初始化规划图
        # 第0层：当前状态的谓词
        graph_layers = [set(pred.compact_str for pred in current_predicates)]
        goal_levels = {}  # 记录每个目标谓词首次出现的层级
        
        max_layers = 10  # 限制最大层数，避免无限循环
        
        for layer in range(max_layers):
            current_layer_predicates = graph_layers[layer]
            new_predicates = set(current_layer_predicates)  # 复制当前层
            
            # 检查当前层是否有新的目标谓词出现
            for goal_pred in goal_predicates:
                goal_pred_str = goal_pred.compact_str
                if goal_pred_str in current_layer_predicates and goal_pred_str not in goal_levels:
                    goal_levels[goal_pred_str] = layer
            
            # 如果所有目标谓词都已找到，停止构建
            if len(goal_levels) == len(goal_predicates):
                break
                
            # 应用所有可能的动作（忽略删除效果）
            actions_applied = False
            for action in self._all_actions:
                # 检查动作的前置条件是否满足
                if action.is_precond_satisfied_from_predicates(
                    [pred for pred in current_predicates 
                     if pred.compact_str in current_layer_predicates]
                ):
                    # 添加动作的后置条件（忽略删除效果）
                    for post_cond in action.post_cond:
                        post_cond_str = post_cond.compact_str
                        if post_cond_str not in current_layer_predicates:
                            new_predicates.add(post_cond_str)
                            actions_applied = True
            
            # 如果没有新的谓词产生，停止构建
            if not actions_applied or new_predicates == current_layer_predicates:
                break
                
            graph_layers.append(new_predicates)
        
        # 计算启发值：所有目标谓词的最大出现层级
        if not goal_levels:
            # 如果没有目标谓词被找到，返回一个较大的值
            return max_layers
            
        max_level = max(goal_levels.values())
        
        # 对于未找到的目标谓词，使用最大层数作为惩罚
        missing_goals = len(goal_predicates) - len(goal_levels)
        if missing_goals > 0:
            max_level += missing_goals * 2  # 惩罚未找到的目标
            
        return max_level
    
    def _extract_goal_predicates(self, goal_expr):
        """
        从目标逻辑表达式中提取所有谓词
        
        :param goal_expr: 目标逻辑表达式
        :return: 谓词列表
        """
        predicates = []
        
        if hasattr(goal_expr, 'sub_exprs') and goal_expr.sub_exprs:
            # 递归处理子表达式
            for sub_expr in goal_expr.sub_exprs:
                if hasattr(sub_expr, 'sub_exprs'):
                    # 如果是逻辑表达式，递归提取
                    predicates.extend(self._extract_goal_predicates(sub_expr))
                else:
                    # 如果是谓词，直接添加
                    predicates.append(sub_expr)
        else:
            # 如果是单个谓词
            predicates.append(goal_expr)
            
        return predicates
###################################################################################################
    def _replan(self, pred_vals, gen_plan_idx: int):
        """重新规划"""
        if self._select_random_goal:
            # 随机选择一个计划
            index_plan = gen_plan_idx
        else:
            # 固定使用指定的plan_idx
            index_plan = self._plan_idx - 1

        possible_plans = [
            # 不做任何事
            None,
            # 移动第一个对象
            self._pddl_prob.stage_goals["stage_1_2"],
            # 移动第二个对象
            self._pddl_prob.stage_goals["stage_2_2"],
            # 移动两个对象
            self._pddl_prob.goal,
        ]
        pddl_goal_selected = possible_plans[index_plan]
        if pddl_goal_selected is None:
            # 代理不想做任何事,所以没有需要规划的内容
            plans = []
        else:
            plans = self._get_plan(pred_vals, pddl_goal_selected)

        return plans

    def _get_plan_action(self, pred_vals, batch_idx: int, gen_plan_idx: int):
        """获取计划动作"""
        if self._should_replan[batch_idx]:
            # 重新计算这个批次元素的计划
            self._plans[batch_idx] = self._replan(pred_vals, gen_plan_idx)
            self._next_sol_idxs[batch_idx] = 0
            if self._plans[batch_idx] is None:
                return None

        # 进入计划的下一部分
        cur_plan = self._plans[batch_idx]
        cur_idx = self._next_sol_idxs[batch_idx]
        if cur_idx >= len(cur_plan):
            cur_ac = None
        else:
            cur_ac = cur_plan[cur_idx]

        self._next_sol_idxs[batch_idx] += 1
        return cur_ac

    def get_next_skill(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        plan_masks,
        deterministic,
        log_info,
    ):
        """获取下一个技能"""
        batch_size = masks.shape[0]
        all_pred_vals = observations["all_predicates"]
        next_skill = torch.zeros(batch_size)
        skill_args_data = [None for _ in range(batch_size)]
        immediate_end = torch.zeros(batch_size, dtype=torch.bool)

        # 为新环境生成计划ID
        if (~masks).sum() > 0:
            self.gen_plan_ids_batch[~masks[:, 0].cpu()] = torch.randint(
                low=self.low_plan[self._plan_idx - 1],
                high=self.high_plan[self._plan_idx - 1] + 1,
                size=[(~masks).int().sum().item()],
                dtype=torch.int32,
            )

        # 处理每个批次
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan != 1.0:
                continue
            gen_plan_idx = self.gen_plan_ids_batch[batch_idx]
            cur_ac = self._get_plan_action(
                all_pred_vals[batch_idx], batch_idx, gen_plan_idx
            )
            if cur_ac is not None:
                next_skill[batch_idx] = self._skill_name_to_idx[cur_ac.name]
                skill_args_data[batch_idx] = [param.name for param in cur_ac.param_values]  # type: ignore[call-overload]
            else:
                # 如果没有下一个动作,什么都不做
                next_skill[batch_idx] = self._skill_name_to_idx["wait"]
                # 等待1步
                skill_args_data[batch_idx] = ["1"]  # type: ignore[call-overload]
        return (
            next_skill,
            skill_args_data,
            immediate_end,
            PolicyActionData(),
        )
