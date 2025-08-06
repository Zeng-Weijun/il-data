#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 导入必要的库
import collections
import inspect
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from habitat import logger
from habitat.utils import profiling_wrapper
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ppo.policy import NetPolicy
from habitat_baselines.rl.ppo.updater import Updater
from habitat_baselines.rl.ver.ver_rollout_storage import VERRolloutStorage
from habitat_baselines.utils.common import (
    LagrangeInequalityCoefficient,
    inference_mode,
)
from habitat_baselines.utils.timing import g_timer

from habitat_baselines.rl.ppo.belief_policy import AttentiveBeliefPolicy

# PPO算法的epsilon值
EPS_PPO = 1e-5


@baseline_registry.register_updater
class PPO(nn.Module, Updater):
    # 定义熵系数类型
    entropy_coef: Union[float, LagrangeInequalityCoefficient]

    @classmethod
    def from_config(cls, actor_critic: NetPolicy, config, aux_tasks = [], aux_names = [], aux_cfg = None):
        # 从配置创建PPO实例
        return cls(
            actor_critic=actor_critic,  # Actor-Critic网络
            clip_param=config.clip_param,  # PPO裁剪参数
            ppo_epoch=config.ppo_epoch,  # PPO训练轮数
            num_mini_batch=config.num_mini_batch,  # mini-batch数量
            value_loss_coef=config.value_loss_coef,  # 价值损失系数
            entropy_coef=config.entropy_coef,  # 熵损失系数
            aux_loss_coef=0.0,  # 辅助损失系数
            lr=config.lr,  # 学习率
            eps=config.eps,  # epsilon值
            max_grad_norm=config.max_grad_norm,  # 最大梯度范数
            use_clipped_value_loss=config.use_clipped_value_loss,  # 是否使用裁剪的价值损失
            use_normalized_advantage=config.use_normalized_advantage,  # 是否使用标准化优势函数
            entropy_target_factor=config.entropy_target_factor,  # 熵目标因子
            use_adaptive_entropy_pen=config.use_adaptive_entropy_pen,  # 是否使用自适应熵惩罚
            aux_tasks=aux_tasks,  # 辅助任务列表
            aux_names=aux_names,  # 辅助任务名称列表
            aux_cfg=aux_cfg,  # 辅助任务配置
        )

    def __init__(
        self,
        actor_critic: NetPolicy,  # Actor-Critic网络
        clip_param: float,  # PPO裁剪参数
        ppo_epoch: int,  # PPO训练轮数
        num_mini_batch: int,  # mini-batch数量
        value_loss_coef: float,  # 价值损失系数
        entropy_coef: float,  # 熵损失系数
        aux_loss_coef: float = 0.0,  # 辅助损失系数
        lr: Optional[float] = None,  # 学习率
        eps: Optional[float] = None,  # epsilon值
        max_grad_norm: Optional[float] = None,  # 最大梯度范数
        use_clipped_value_loss: bool = False,  # 是否使用裁剪的价值损失
        use_normalized_advantage: bool = True,  # 是否使用标准化优势函数
        entropy_target_factor: float = 0.0,  # 熵目标因子
        use_adaptive_entropy_pen: bool = False,  # 是否使用自适应熵惩罚
        aux_tasks=[],  # 辅助任务列表
        aux_names=[],  # 辅助任务名称列表
        aux_cfg=None,  # 辅助任务配置
    ) -> None:
        super().__init__()

        # 初始化各种参数
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.aux_loss_coef = aux_loss_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # 获取设备信息
        self.device = next(actor_critic.parameters()).device

        # 如果使用自适应熵惩罚且满足特定条件
        # use_adaptive_entropy_pen: 是否使用自适应熵惩罚
        # num_actions: actor_critic网络的动作空间维度
        # action_distribution_type: 动作分布类型,这里要求是高斯分布
        if (
            use_adaptive_entropy_pen
            and hasattr(self.actor_critic, "num_actions") 
            and getattr(self.actor_critic, "action_distribution_type", None)
            == "gaussian"
        ):
            num_actions = self.actor_critic.num_actions

            # 创建Lagrange不等式系数
            # LagrangeInequalityCoefficient用于自适应调整熵惩罚系数
            # entropy_target_factor * num_actions: 熵的目标值
            # init_alpha: 初始熵系数
            # alpha_max/min: 熵系数的上下界
            # greater_than: 表示熵要大于目标值
            self.entropy_coef = LagrangeInequalityCoefficient(
                -float(entropy_target_factor) * num_actions,
                init_alpha=entropy_coef,
                alpha_max=1.0,
                alpha_min=1e-4,
                greater_than=True,
            ).to(device=self.device)
        # 初始化辅助任务相关参数
        self._aux_tasks=[]
        self._aux_names=[]
        if aux_cfg:
            self.aux_cfg = aux_cfg
            self._aux_tasks = aux_tasks
            self._aux_names = aux_names

        # 设置是否使用标准化优势函数
        self.use_normalized_advantage = use_normalized_advantage
        # 创建优化器
        self.optimizer = self._create_optimizer(lr, eps, self._aux_tasks)

        # 获取非actor-critic参数
        self.non_ac_params = [
            p
            for name, p in self.named_parameters()
            if not name.startswith("actor_critic.")
        ]

    def _create_optimizer(self, lr, eps, aux_tasks=None):
        # 创建优化器
        # 获取需要训练的参数
        # 获取所有需要梯度更新的参数
        # filter函数过滤出requires_grad=True的参数
        # self.parameters()返回模型中的所有参数
        params = list(filter(lambda p: p.requires_grad, self.parameters()))
        logger.info(
            f"(No Aux) Main Number of params to train: {sum(param.numel() for param in params)}"
        )

        # 如果有辅助任务，添加辅助任务的参数
        if len(aux_tasks) > 0:
            for aux_t in aux_tasks:
                params += list(filter(lambda p: p.requires_grad, aux_t.parameters()))

        logger.info(
            f"Total Number of params to train: {sum(param.numel() for param in params)}"
        )
        
        # 如果有参数需要训练
        if len(params) > 0:
            optim_cls = optim.Adam
            optim_kwargs = dict(
                params=params,
                lr=lr,
                eps=eps,
            )
            # 检查优化器是否支持foreach参数
            # foreach=True可以让优化器对多个张量同时进行操作,提高性能
            signature = inspect.signature(optim_cls.__init__)
            if "foreach" in signature.parameters:
                optim_kwargs["foreach"] = True
            else:
                # 如果不支持foreach,尝试使用multi_tensor优化器
                # multi_tensor优化器可以将多个小张量合并成一个大张量进行操作,提高计算效率
                try:
                    import torch.optim._multi_tensor
                except ImportError:
                    pass
                else:
                    optim_cls = torch.optim._multi_tensor.Adam

            return optim_cls(**optim_kwargs)
        else:
            return None

    def get_advantages(self, rollouts: RolloutStorage) -> Tensor:
        # 计算优势函数
        advantages = (
            rollouts.buffers["returns"]  # type: ignore
            - rollouts.buffers["value_preds"]
        )
        if not self.use_normalized_advantage:
            return advantages

        # 计算方差和均值
        var, mean = self._compute_var_mean(
            advantages[torch.isfinite(advantages)]
        )

        # 标准化优势函数
        advantages -= mean
        return advantages.mul_(torch.rsqrt(var + EPS_PPO))

    @staticmethod
    def _compute_var_mean(x):
        # 计算方差和均值
        return torch.var_mean(x)

    def _set_grads_to_none(self):
        # 将梯度设置为None
        for pg in self.optimizer.param_groups:
            for p in pg["params"]:
                p.grad = None

    @g_timer.avg_time("ppo.update_from_batch", level=1)
    def _update_from_batch(self, batch, epoch, rollouts, learner_metrics):
        """
        从mini-batch更新模型参数
        """

        def record_min_mean_max(t: torch.Tensor, prefix: str):
            # 记录最小值、均值和最大值
            for name, op in (
                ("min", torch.min),
                ("mean", torch.mean),
                ("max", torch.max),
            ):
                learner_metrics[f"{prefix}_{name}"].append(op(t))

        # 将梯度设置为None
        self._set_grads_to_none()
        aux_dist_entropy = None
        # AttentiveBeliefPolicy是一种特殊的策略网络,它使用注意力机制来处理信念状态
        # 它可以从历史观察中提取重要信息,并用于当前决策
        if isinstance(self.actor_critic, AttentiveBeliefPolicy):
            # 对于AttentiveBeliefPolicy,evaluate_actions会返回更多信息:
            # values: 状态值估计
            # action_log_probs: 动作的对数概率
            # dist_entropy: 动作分布的熵
            # final_rnn_state: RNN的最终隐藏状态
            # rnn_features: RNN提取的特征
            # individual_rnn_features: 每个时间步的RNN特征
            # aux_dist_entropy: 辅助任务的分布熵
            # aux_weights: 辅助任务的权重
            (
                values,
                action_log_probs,
                dist_entropy,
                final_rnn_state,
                rnn_features,
                individual_rnn_features,
                aux_dist_entropy,
                aux_weights,
            ) = self._evaluate_actions(
                batch["observations"],
                batch["recurrent_hidden_states"],
                batch["prev_actions"],
                batch["masks"],
                batch["actions"],
                batch.get("rnn_build_seq_info", None),
            )
        else:
            # 对于普通的策略网络,只返回基本的评估结果
            (
                values,
                action_log_probs,
                dist_entropy,
                final_rnn_state,
                aux_loss_res,
            ) = self._evaluate_actions(
                batch["observations"],
                batch["recurrent_hidden_states"],
                batch["prev_actions"],
                batch["masks"],
                batch["actions"],
                batch.get("rnn_build_seq_info", None),
            )
        # 计算概率比率
        ratio = torch.exp(action_log_probs - batch["action_log_probs"])

        # 计算PPO目标函数的两部分
        surr1 = batch["advantages"] * ratio
        surr2 = batch["advantages"] * (
            torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
        )
        # 取两者的最小值作为动作损失
        action_loss = -torch.min(surr1, surr2)

        # 转换值函数预测为浮点数
        values = values.float()
        orig_values = values

        # 如果使用裁剪的值函数损失
        if self.use_clipped_value_loss:
            delta = values.detach() - batch["value_preds"]
            value_pred_clipped = batch["value_preds"] + delta.clamp(
                -self.clip_param, self.clip_param
            )

            values = torch.where(
                delta.abs() < self.clip_param,
                values,
                value_pred_clipped,
            )

        # 计算值函数损失
        value_loss = 0.5 * F.mse_loss(
            values, batch["returns"], reduction="none"
        )

        # 如果使用重要性采样系数
        if "is_coeffs" in batch:
            assert isinstance(batch["is_coeffs"], torch.Tensor)
            ver_is_coeffs = batch["is_coeffs"].clamp(max=1.0)
            mean_fn = lambda t: torch.mean(ver_is_coeffs * t)
        else:
            mean_fn = torch.mean

        # 计算各种损失的平均值
        action_loss, value_loss, dist_entropy = map(
            mean_fn,
            (action_loss, value_loss, dist_entropy),
        )
        
        # 计算辅助任务的损失
        total_aux_loss = 0
        aux_losses = []
        # 检查策略网络是否为AttentiveBeliefPolicy类型且存在辅助任务
        if isinstance(self.actor_critic, AttentiveBeliefPolicy) and len(self._aux_tasks) > 0:
            # 调用策略网络的evaluate_aux_losses方法计算辅助任务的原始损失
            # 输入包括:batch数据、RNN最终状态、RNN特征、每个时间步的RNN特征
            aux_raw_losses = self.actor_critic.evaluate_aux_losses(batch, final_rnn_state, rnn_features, individual_rnn_features)
            
            # 将所有辅助任务的损失堆叠成一个张量
            aux_losses = torch.stack(aux_raw_losses)
            
            # 计算所有辅助任务损失的总和
            # dim=0表示沿着第一个维度(任务维度)求和
            total_aux_loss = torch.sum(aux_losses, dim=0)

        # 组合所有损失
        all_losses = [
            self.value_loss_coef * value_loss,  # 值函数损失
            action_loss,  # 动作损失
        ]

        # 添加辅助任务损失
        if isinstance(self.actor_critic, AttentiveBeliefPolicy):
            all_losses.append(total_aux_loss * self.aux_loss_coef)
            
        # 添加熵损失
        if isinstance(self.entropy_coef, float):
            all_losses.append(-self.entropy_coef * dist_entropy)
        else:
            all_losses.append(self.entropy_coef.lagrangian_loss(dist_entropy))

        # 添加辅助分布熵
        if aux_dist_entropy is not None:
            all_losses.append(aux_dist_entropy * self.aux_cfg.entropy_coef)

        # 如果没有辅助任务，添加其他辅助损失
        if len(self._aux_tasks) == 0:
            all_losses.extend(v["loss"] for v in aux_loss_res.values())

        # 计算总损失
        total_loss = torch.stack(all_losses).sum()

        # 反向传播前的处理
        total_loss = self.before_backward(total_loss)
        total_loss.backward()
        self.after_backward(total_loss)

        # 更新参数
        grad_norm = self.before_step()
        self.optimizer.step()
        self.after_step()

        # 记录指标
        with inference_mode():
            if "is_coeffs" in batch:
                record_min_mean_max(batch["is_coeffs"], "ver_is_coeffs")
            record_min_mean_max(orig_values, "value_pred")
            record_min_mean_max(ratio, "prob_ratio")

            learner_metrics["value_loss"].append(value_loss)
            learner_metrics["action_loss"].append(action_loss)
            learner_metrics["dist_entropy"].append(dist_entropy)

            if epoch == (self.ppo_epoch - 1):
                learner_metrics["ppo_fraction_clipped"].append(
                    (ratio > (1.0 + self.clip_param)).float().mean()
                    + (ratio < (1.0 - self.clip_param)).float().mean()
                )

            learner_metrics["grad_norm"].append(grad_norm)
            if isinstance(self.entropy_coef, LagrangeInequalityCoefficient):
                learner_metrics["entropy_coef"].append(
                    self.entropy_coef().detach()
                )

            # 记录辅助任务相关指标
            if len(self._aux_tasks) == 0:
                for name, res in aux_loss_res.items():
                    for k, v in res.items():
                        learner_metrics[f"aux_{name}_{k}"].append(v.detach())
            else:
                learner_metrics["aux_entropy"].append(aux_dist_entropy)
                for i, aux_loss in enumerate(aux_losses):
                    learner_metrics[f"aux_entropy_{self._aux_names[i]}"].append(aux_loss.item())
                for i, aux_weight in enumerate(aux_weights):
                    learner_metrics[f"aux_weights_{self._aux_names[i]}"].append(aux_weight.item())

            # 记录其他指标
            if "is_stale" in batch:
                assert isinstance(batch["is_stale"], torch.Tensor)
                learner_metrics["fraction_stale"].append(
                    batch["is_stale"].float().mean()
                )

            if isinstance(rollouts, VERRolloutStorage):
                assert isinstance(batch["policy_version"], torch.Tensor)
                record_min_mean_max(
                    (
                        rollouts.current_policy_version
                        - batch["policy_version"]
                    ).float(),
                    "policy_version_difference",
                )

    def update(
        self,
        rollouts: RolloutStorage,
    ) -> Dict[str, float]:
        # 获取优势函数
        advantages = self.get_advantages(rollouts)

        # 初始化指标字典
        learner_metrics: Dict[str, List[Any]] = collections.defaultdict(list)

        # 进行多轮PPO更新
        for epoch in range(self.ppo_epoch):
            profiling_wrapper.range_push("PPO.update epoch")
            # 获取数据生成器
            data_generator = rollouts.data_generator(
                advantages, self.num_mini_batch
            )

            # 对每个mini-batch进行更新
            for _bid, batch in enumerate(data_generator):
                self._update_from_batch(
                    batch, epoch, rollouts, learner_metrics
                )

            profiling_wrapper.range_pop()  # PPO.update epoch

        # 清空梯度
        self._set_grads_to_none()

        # 返回平均指标
        with inference_mode():
            return {
                k: float(
                    torch.stack(
                        [torch.as_tensor(v, dtype=torch.float32) for v in vs]
                    ).mean()
                )
                for k, vs in learner_metrics.items()
            }

    @g_timer.avg_time("ppo.eval_actions", level=1)
    def _evaluate_actions(self, *args, **kwargs):
        """
        内部方法，用于调用Policy.evaluate_actions
        这样设计是为了可以通过继承来重写该调用
        """
        return self.actor_critic.evaluate_actions(*args, **kwargs)

    def before_backward(self, loss: Tensor) -> Tensor:
        # 反向传播前的处理
        return loss

    def after_backward(self, loss: Tensor) -> None:
        # 反向传播后的处理
        pass

    def before_step(self) -> torch.Tensor:
        # 优化器步骤前的处理
        handles = []
        # 如果使用分布式训练
        if torch.distributed.is_initialized():
            for p in self.non_ac_params:
                if p.grad is not None:
                    p.grad.data.detach().div_(
                        torch.distributed.get_world_size()
                    )
                    handles.append(
                        torch.distributed.all_reduce(
                            p.grad.data.detach(), async_op=True
                        )
                    )

        # 梯度裁剪
        grad_norm = nn.utils.clip_grad_norm_(
            self.actor_critic.policy_parameters(),
            self.max_grad_norm,
        )

        # 对辅助损失参数进行梯度裁剪
        for v in self.actor_critic.aux_loss_parameters().values():
            nn.utils.clip_grad_norm_(v, self.max_grad_norm)

        # 等待所有异步操作完成
        [h.wait() for h in handles]

        return grad_norm

    def after_step(self) -> None:
        # 优化器步骤后的处理
        if isinstance(self.entropy_coef, LagrangeInequalityCoefficient):
            self.entropy_coef.project_into_bounds()

    def get_resume_state(self):
        # 获取恢复状态
        return {
            "optim_state": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        # 加载状态字典
        if "optim_state" in state:
            self.optimizer.load_state_dict(state["optim_state"])
