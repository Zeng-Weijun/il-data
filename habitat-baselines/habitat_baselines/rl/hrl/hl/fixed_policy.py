# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 导入所需的类型提示模块
from typing import List, Tuple

# 导入PyTorch
import torch

# 导入相关的功能模块
from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData


class FixedHighLevelPolicy(HighLevelPolicy):
    """
    执行PDDL问题文件中'solution'字段指定的固定高级动作序列。
    :property _solution_actions: 元组列表，其中第一个元组元素是动作名称，
        第二个是动作参数。为每个环境存储一个计划。
    """

    # 定义解决方案动作的类型注解
    _solution_actions: List[List[Tuple[str, List[str]]]]

    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

        # 为每个环境更新解决方案动作
        self._update_solution_actions(
            [self._parse_solution_actions() for _ in range(self._num_envs)]
        )

        # 初始化下一个解决方案索引为零向量
        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)

    def _update_solution_actions(
        self, solution_actions: List[List[Tuple[str, List[str]]]]
    ) -> None:
        # 检查解决方案动作列表是否为空
        if len(solution_actions) == 0:
            raise ValueError(
                "Solution actions must be non-empty (if want to execute no actions, just include a no-op)"
            )
        self._solution_actions = solution_actions

    def _parse_solution_actions(self) -> List[Tuple[str, List[str]]]:
        """
        返回要执行的动作序列，包含：
        - 动作名称
        - 动作参数列表
        """
        # 获取PDDL问题的解决方案
        solution = self._pddl_prob.solution

        # 初始化解决方案动作列表
        solution_actions = []
        for i, hl_action in enumerate(solution):
            # 构建解决方案动作元组
            sol_action = (
                hl_action.name,
                [x.name for x in hl_action.param_values],
            )
            solution_actions.append(sol_action)

            # 如果配置了机械臂休息且不是最后一个动作
            if self._config.add_arm_rest and i < (len(solution) - 1):
                solution_actions.append(parse_func("reset_arm(0)"))

        # 在末尾添加等待动作
        solution_actions.append(parse_func("wait(30)"))

        return solution_actions

    def apply_mask(self, mask):
        """
        将给定的掩码应用于下一个技能索引。

        参数:
            mask: 形状为(num_envs,)的二进制掩码，将被应用到下一个技能索引。
        """
        # 应用掩码到下一个解决方案索引
        self._next_sol_idxs *= mask.cpu().view(-1)

    def _get_next_sol_idx(self, batch_idx, immediate_end):
        """
        从解决方案动作列表中获取下一个要使用的索引。

        参数:
            batch_idx: 当前环境的索引。

        返回:
            从解决方案动作列表中获取的下一个要使用的索引。
        """
        # 检查是否已经执行完所有动作
        if self._next_sol_idxs[batch_idx] >= len(
            self._solution_actions[batch_idx]
        ):
            baselines_logger.info(
                f"Calling for immediate end with {self._next_sol_idxs[batch_idx]}"
            )
            immediate_end[batch_idx] = True
            # 重复最后一个动作
            return len(self._solution_actions[batch_idx]) - 1
        else:
            return self._next_sol_idxs[batch_idx].item()

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        # 返回零值，用于多智能体策略中的值连接
        return torch.zeros(rnn_hidden_states.shape[0], 1).to(
            rnn_hidden_states.device
        )

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
        # 获取批次大小
        batch_size = masks.shape[0]
        # 初始化下一个技能张量
        next_skill = torch.zeros(batch_size)
        # 初始化技能参数数据列表
        skill_args_data = [None for _ in range(batch_size)]
        # 初始化立即结束标志
        immediate_end = torch.zeros(batch_size, dtype=torch.bool)
        
        # 遍历每个批次
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan == 1.0:
                # 获取下一个解决方案索引
                use_idx = self._get_next_sol_idx(batch_idx, immediate_end)

                # 获取技能名称和参数
                skill_name, skill_args = self._solution_actions[batch_idx][
                    use_idx
                ]
                baselines_logger.info(
                    f"Got next element of the plan with {skill_name}, {skill_args}"
                )
                
                # 检查技能名称是否存在
                if skill_name not in self._skill_name_to_idx:
                    raise ValueError(
                        f"Could not find skill named {skill_name} in {self._skill_name_to_idx}"
                    )
                next_skill[batch_idx] = self._skill_name_to_idx[skill_name]

                # 设置技能参数
                skill_args_data[batch_idx] = skill_args  # type: ignore[call-overload]

                # 更新下一个解决方案索引
                self._next_sol_idxs[batch_idx] += 1

        return next_skill, skill_args_data, immediate_end, PolicyActionData()

    def filter_envs(self, curr_envs_to_keep_active):
        """
        清理策略的状态变量，使其与活动环境匹配
        """
        # 根据活动环境过滤下一个解决方案索引
        self._next_sol_idxs = self._next_sol_idxs[curr_envs_to_keep_active]
        # 为每个环境解析解决方案动作
        parse_solution_actions = [
            self._parse_solution_actions() for _ in range(self._num_envs)
        ]
        # 更新解决方案动作
        self._update_solution_actions(
            [
                parse_solution_actions[i]
                for i in range(curr_envs_to_keep_active.shape[0])
                if curr_envs_to_keep_active[i]
            ]
        )
