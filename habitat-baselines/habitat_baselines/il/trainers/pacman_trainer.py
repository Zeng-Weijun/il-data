#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import time
from datetime import datetime
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import habitat
from habitat import logger
from habitat.datasets.utils import VocabDict
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.il.data.nav_data import NavDataset
from habitat_baselines.il.metrics import NavMetric
from habitat_baselines.il.models.models import (
    MaskedNLLCriterion,
    NavPlannerControllerModel,
)
from habitat_baselines.utils.common import generate_video


@baseline_registry.register_trainer(name="pacman")
class PACMANTrainer(BaseILTrainer):
    """PACMAN(规划器和控制器模块)导航模型的训练器类
    用于EmbodiedQA (Das et. al.;CVPR 2018)
    论文: https://embodiedqa.org/paper.pdf.
    """
    supported_tasks = ["EQA-v0"]  # 支持的任务类型

    def __init__(self, config=None):
        """初始化训练器
        Args:
            config: 配置参数
        """
        super().__init__(config)

        # 设置设备(GPU/CPU)
        self.device = (
            torch.device("cuda", self.config.habitat_baselines.torch_gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if config is not None:
            logger.info(f"config: {config}")

    def _save_nav_results(
        self,
        ckpt_path: str,
        ep_id: int, 
        questions: torch.Tensor,
        imgs: List[np.ndarray],
        q_vocab_dict: VocabDict,
        results_dir: str,
        writer: TensorboardWriter,
        video_option: list,
    ) -> None:
        """保存导航评估结果
        Args:
            ckpt_path: 被评估的检查点路径
            ep_id: episode ID(批次索引)
            questions: 输入到模型的问题
            imgs: 包含输入帧的图像张量
            q_vocab_dict: 问题词汇表字典
            results_dir: 保存结果的目录
            writer: tensorboard写入器
            video_option: ["disk", "tb"]
        Returns:
            None
        """

        question = questions[0]

        # 获取检查点epoch
        ckpt_epoch = ckpt_path[ckpt_path.rfind("/") + 1 : -5]
        results_dir = os.path.join(results_dir, ckpt_epoch)
        ckpt_no = int(ckpt_epoch[6:])

        # 将问题token转换为字符串
        q_string = q_vocab_dict.token_idx_2_string(question)
        frames_with_text: List[np.ndarray] = []
        
        # 为每一帧添加文字
        for frame in imgs:
            border_width = 32
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (0, 0, 0)
            scale = 0.3
            thickness = 1

            # 添加白色边框
            frame = cv2.copyMakeBorder(
                frame,
                border_width,
                border_width,
                border_width,
                border_width,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )

            # 添加问题文字
            frame = cv2.putText(
                frame,
                "Question: " + q_string,
                (10, 15),
                font,
                scale,
                color,
                thickness,
            )

            frames_with_text.append(np.ndarray(frame))
            
        # 生成视频
        generate_video(
            video_option,
            results_dir,
            frames_with_text,
            ep_id,
            ckpt_no,
            {},
            writer,
            fps=5,
        )

    def train(self) -> None:
        """训练EQA导航模型的主方法
        Returns:
            None
        """
        config = self.config

        with habitat.Env(config.habitat) as env:
            # 创建导航数据集
            nav_dataset = (
                NavDataset(
                    config,
                    env,
                    self.device,
                )
                .shuffle(1000)
                .decode("rgb")
            )

            nav_dataset = nav_dataset.map(nav_dataset.map_dataset_sample)

            # 创建数据加载器
            train_loader = DataLoader(
                nav_dataset,
                batch_size=config.habitat_baselines.il.nav.batch_size,
            )

            logger.info("train_loader has {} samples".format(len(nav_dataset)))

            # 获取词汇表
            q_vocab_dict, _ = nav_dataset.get_vocab_dicts()

            # 初始化模型
            model_kwargs = {"q_vocab": q_vocab_dict.word2idx_dict}
            model = NavPlannerControllerModel(**model_kwargs)

            # 定义损失函数
            planner_loss_fn = MaskedNLLCriterion()
            controller_loss_fn = MaskedNLLCriterion()

            # 定义优化器
            optim = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=float(config.habitat_baselines.il.nav.lr),
            )

            # 初始化指标
            metrics = NavMetric(
                info={"split": "train"},
                metric_names=["planner_loss", "controller_loss"],
                log_json=os.path.join(
                    config.habitat_baselines.il.output_log_dir, "train.json"
                ),
            )

            epoch = 1
            avg_p_loss = 0.0  # 平均规划器损失
            avg_c_loss = 0.0  # 平均控制器损失

            logger.info(model)
            model.train().to(self.device)

            # 训练循环
            with TensorboardWriter(
                "train_{}/{}".format(
                    config.habitat_baselines.tensorboard_dir,
                    datetime.today().strftime("%Y-%m-%d-%H:%M"),
                ),
                flush_secs=self.flush_secs,
            ) as writer:
                while epoch <= config.habitat_baselines.il.nav.max_epochs:
                    start_time = time.time()
                    for t, batch in enumerate(train_loader):
                        # 将数据移到设备上
                        batch = (
                            item.to(self.device, non_blocking=True)
                            for item in batch
                        )
                        
                        # 解包批次数据
                        (
                            idx,
                            questions,
                            _,
                            planner_img_feats,
                            planner_actions_in,
                            planner_actions_out,
                            planner_action_lengths,
                            planner_masks,
                            controller_img_feats,
                            controller_actions_in,
                            planner_hidden_idx,
                            controller_outs,
                            controller_action_lengths,
                            controller_masks,
                        ) = batch

                        # 按长度排序
                        (
                            planner_action_lengths,
                            perm_idx,
                        ) = planner_action_lengths.sort(0, descending=True)
                        questions = questions[perm_idx]

                        # 重排数据
                        planner_img_feats = planner_img_feats[perm_idx]
                        planner_actions_in = planner_actions_in[perm_idx]
                        planner_actions_out = planner_actions_out[perm_idx]
                        planner_masks = planner_masks[perm_idx]

                        controller_img_feats = controller_img_feats[perm_idx]
                        controller_actions_in = controller_actions_in[perm_idx]
                        controller_outs = controller_outs[perm_idx]
                        planner_hidden_idx = planner_hidden_idx[perm_idx]
                        controller_action_lengths = controller_action_lengths[
                            perm_idx
                        ]
                        controller_masks = controller_masks[perm_idx]

                        # 前向传播
                        (
                            planner_scores,
                            controller_scores,
                            planner_hidden,
                        ) = model(
                            questions,
                            planner_img_feats,
                            planner_actions_in,
                            planner_action_lengths.cpu().numpy(),
                            planner_hidden_idx,
                            controller_img_feats,
                            controller_actions_in,
                            controller_action_lengths,
                        )

                        # 计算损失
                        planner_logprob = F.log_softmax(planner_scores, dim=1)
                        controller_logprob = F.log_softmax(
                            controller_scores, dim=1
                        )

                        planner_loss = planner_loss_fn(
                            planner_logprob,
                            planner_actions_out[
                                :, : planner_action_lengths.max()
                            ].reshape(-1, 1),
                            planner_masks[
                                :, : planner_action_lengths.max()
                            ].reshape(-1, 1),
                        )

                        controller_loss = controller_loss_fn(
                            controller_logprob,
                            controller_outs[
                                :, : controller_action_lengths.max()
                            ].reshape(-1, 1),
                            controller_masks[
                                :, : controller_action_lengths.max()
                            ].reshape(-1, 1),
                        )

                        # 梯度清零
                        optim.zero_grad()

                        # 更新指标
                        metrics.update(
                            [planner_loss.item(), controller_loss.item()]
                        )

                        # 反向传播
                        (planner_loss + controller_loss).backward()

                        # 优化器步进
                        optim.step()

                        # 获取损失统计
                        (planner_loss, controller_loss) = metrics.get_stats()

                        avg_p_loss += planner_loss
                        avg_c_loss += controller_loss

                        # 记录日志
                        if t % config.habitat_baselines.log_interval == 0:
                            logger.info("Epoch: {}".format(epoch))
                            logger.info(metrics.get_stat_string())

                            writer.add_scalar("planner loss", planner_loss, t)
                            writer.add_scalar(
                                "controller loss", controller_loss, t
                            )

                            metrics.dump_log()

                    # 计算每个epoch的平均损失
                    num_batches = math.ceil(
                        len(nav_dataset)
                        / config.habitat_baselines.il.nav.batch_size
                    )

                    avg_p_loss /= num_batches
                    avg_c_loss /= num_batches

                    # 记录epoch统计信息
                    end_time = time.time()
                    time_taken = "{:.1f}".format((end_time - start_time) / 60)

                    logger.info(
                        "Epoch {} completed. Time taken: {} minutes.".format(
                            epoch, time_taken
                        )
                    )

                    logger.info(
                        "Average planner loss: {:.2f}".format(avg_p_loss)
                    )
                    logger.info(
                        "Average controller loss: {:.2f}".format(avg_c_loss)
                    )

                    print("-----------------------------------------")

                    # 保存检查点
                    if (
                        epoch % config.habitat_baselines.checkpoint_interval
                        == 0
                    ):
                        self.save_checkpoint(
                            model.state_dict(), "epoch_{}.ckpt".format(epoch)
                        )

                    epoch += 1

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        """评估单个检查点
        Args:
            checkpoint_path: 检查点路径
            writer: tensorboard写入器对象,用于记录到tensorboard
            checkpoint_index: 当前检查点的索引,用于记录
        Returns:
            None
        """
        config = self.config

        # 设置评估数据集
        with habitat.config.read_write(config):
            config.habitat.dataset.split = (
                self.config.habitat_baselines.eval.split
            )

        with habitat.Env(config.habitat) as env:
            # 创建评估数据集和加载器
            nav_dataset = NavDataset(
                config,
                env,
                self.device,
            ).decode("rgb")

            nav_dataset = nav_dataset.map(nav_dataset.map_dataset_sample)

            eval_loader = DataLoader(nav_dataset)

            logger.info("eval_loader has {} samples".format(len(nav_dataset)))

            # 获取词汇表
            q_vocab_dict, ans_vocab_dict = nav_dataset.get_vocab_dicts()

            # 初始化模型
            model_kwargs = {"q_vocab": q_vocab_dict.word2idx_dict}
            model = NavPlannerControllerModel(**model_kwargs)

            invalids = []  # 记录无效样本

            # 加载模型权重
            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict)
            model.eval().to(self.device)

            # 设置结果保存目录
            results_dir = config.habitat_baselines.il.results_dir.format(
                split="val"
            )
            video_option = self.config.habitat_baselines.eval.video_option

            # 初始化评估指标
            metrics = NavMetric(
                info={"split": "val"},
                metric_names=[
                    "{}_{}".format(y, x)
                    for x in [10, 30, 50, "rand_init"]
                    for z in ["", "_f"]
                    for y in [
                        *["d_{}{}".format(k, z) for k in [0, "T", "D", "min"]],
                        *[w for w in ["stop", "ep_len"] if z == ""],
                    ]
                ],
                log_json=os.path.join(
                    config.habitat_baselines.il.output_log_dir, "eval.json"
                ),
            )

            # 评估循环
            for t, batch in enumerate(eval_loader):
                idx, question, answer, actions, action_length, goal_pos = batch

                metrics_slug = {}
                imgs = []  # type:ignore
                
                # 对不同步数进行评估
                for i in [10, 30, 50, "rand_init"]:
                    for j in ["pred", "fwd-only"]:
                        question = question.to(self.device)

                        controller_step = False
                        planner_hidden = model.planner_nav_rnn.init_hidden(1)

                        # 获取分层动作历史
                        (
                            planner_actions_in,
                            planner_img_feats,
                            controller_step,
                            controller_action_in,
                            controller_img_feats,
                            init_pos,
                            controller_action_counter,
                        ) = nav_dataset.get_hierarchical_features_till_spawn(
                            idx.item(),
                            actions[0, : action_length.item()].numpy(),
                            i if i != "rand_init" else action_length.item(),
                            config.habitat_baselines.il.nav.max_controller_actions,
                        )
                        
                        # 预测模式
                        if j == "pred":
                            planner_actions_in = planner_actions_in.to(
                                self.device
                            )
                            planner_img_feats = planner_img_feats.to(
                                self.device
                            )

                            # 规划器前向传播
                            for step in range(planner_actions_in.size(0)):
                                (
                                    planner_scores,
                                    planner_hidden,
                                ) = model.planner_step(
                                    question,
                                    planner_img_feats[step][
                                        (None,) * 2
                                    ],  # 双重unsqueeze
                                    planner_actions_in[step].view(1, 1),
                                    planner_hidden,
                                )

                        # 设置智能体初始状态
                        env.sim.set_agent_state(
                            init_pos.position, init_pos.rotation
                        )
                        init_dist_to_target = env.sim.geodesic_distance(
                            init_pos.position, goal_pos
                        )

                        # 检查目标是否可达
                        if (
                            init_dist_to_target < 0
                            or init_dist_to_target == float("inf")
                        ):  # 不可达
                            invalids.append([idx.item(), i])
                            continue

                        # 初始化距离和位置队列
                        dists_to_target, pos_queue = [init_dist_to_target], [
                            init_pos
                        ]
                        
                        # 预测模式设置
                        if j == "pred":
                            planner_actions, controller_actions = [], []

                            # 控制器动作计数器设置
                            if (
                                config.habitat_baselines.il.nav.max_controller_actions
                                > 1
                            ):
                                controller_action_counter = (
                                    controller_action_counter
                                    % config.habitat_baselines.il.nav.max_controller_actions
                                )
                                controller_action_counter = max(
                                    controller_action_counter - 1, 0
                                )
                            else:
                                controller_action_counter = 0

                            first_step = True
                            first_step_is_controller = controller_step
                            planner_step = True
                            action = int(controller_action_in)

                        # 执行episode
                        img = None
                        for episode_length in range(
                            config.habitat_baselines.il.nav.max_episode_length
                        ):
                            if j == "pred":
                                if not first_step:
                                    # 保存30步回退案例的结果
                                    if (
                                        i == 30
                                    ):
                                        imgs.append(img)
                                    img_feat = (
                                        eval_loader.dataset.get_img_features(
                                            img, preprocess=True
                                        ).view(1, 1, 4608)
                                    )
                                else:
                                    img_feat = controller_img_feats.to(
                                        self.device
                                    ).view(1, 1, 4608)

                                # 控制器决策
                                if not first_step or first_step_is_controller:
                                    controller_action_in = (
                                        torch.LongTensor(1, 1)
                                        .fill_(action)
                                        .to(self.device)
                                    )
                                    controller_scores = model.controller_step(
                                        img_feat,
                                        controller_action_in,
                                        planner_hidden[0],
                                    )

                                    prob = F.softmax(controller_scores, dim=1)
                                    controller_action = int(
                                        prob.max(1)[1].data.cpu().numpy()[0]
                                    )

                                    # 控制器动作处理
                                    if (
                                        controller_action == 1
                                        and controller_action_counter
                                        < config.habitat_baselines.il.nav.max_controller_actions
                                        - 1
                                    ):
                                        controller_action_counter += 1
                                        planner_step = False
                                    else:
                                        controller_action_counter = 0
                                        planner_step = True
                                        controller_action = 0

                                    controller_actions.append(
                                        controller_action
                                    )
                                    first_step = False

                                # 规划器决策
                                if planner_step:
                                    if not first_step:
                                        action_in = (
                                            torch.LongTensor(1, 1)
                                            .fill_(action + 1)
                                            .to(self.device)
                                        )
                                        (
                                            planner_scores,
                                            planner_hidden,
                                        ) = model.planner_step(
                                            question,
                                            img_feat,
                                            action_in,
                                            planner_hidden,
                                        )
                                    prob = F.softmax(planner_scores, dim=1)
                                    action = int(
                                        prob.max(1)[1].data.cpu().numpy()[0]
                                    )
                                    planner_actions.append(action)

                            else:
                                action = 0

                            # 检查episode是否结束
                            episode_done = (
                                action == 3
                                or episode_length
                                >= config.habitat_baselines.il.nav.max_episode_length
                            )

                            # 获取智能体位置
                            agent_pos = env.sim.get_agent_state().position

                            # 记录到目标的距离
                            dists_to_target.append(
                                env.sim.geodesic_distance(agent_pos, goal_pos)
                            )
                            pos_queue.append([agent_pos])

                            if episode_done:
                                break

                            # 动作映射
                            if action == 0:
                                my_action = 1  # 前进
                            elif action == 1:
                                my_action = 2  # 左转
                            elif action == 2:
                                my_action = 3  # 右转
                            elif action == 3:
                                my_action = 0  # 停止

                            # 执行动作
                            observations = env.sim.step(my_action)
                            img = observations["rgb"]
                            first_step = False

                        # 计算统计指标
                        m = "" if j == "pred" else "_f"
                        metrics_slug[
                            "d_T{}_{}".format(m, i)
                        ] = dists_to_target[-1]
                        metrics_slug["d_D{}_{}".format(m, i)] = (
                            dists_to_target[0] - dists_to_target[-1]
                        )
                        metrics_slug["d_min{}_{}".format(m, i)] = np.array(
                            dists_to_target
                        ).min()

                        if j != "fwd-only":
                            metrics_slug[
                                "ep_len_{}".format(i)
                            ] = episode_length
                            if action == 3:
                                metrics_slug["stop_{}".format(i)] = 1
                            else:
                                metrics_slug["stop_{}".format(i)] = 0

                            metrics_slug["d_0_{}".format(i)] = dists_to_target[
                                0
                            ]

                # 整理并更新指标
                metrics_list = []
                for ind, i in enumerate(metrics.metric_names):
                    if i not in metrics_slug:
                        metrics_list.append(metrics.metrics[ind][0])
                    else:
                        metrics_list.append(metrics_slug[i])

                # 更新指标
                metrics.update(metrics_list)

                # 记录日志
                if t % config.habitat_baselines.log_interval == 0:
                    logger.info(
                        "Valid cases: {}; Invalid cases: {}".format(
                            (t + 1) * 8 - len(invalids), len(invalids)
                        )
                    )
                    logger.info(
                        "eval: Avg metrics: {}".format(
                            metrics.get_stat_string(mode=0)
                        )
                    )
                    print(
                        "-----------------------------------------------------"
                    )

                # 保存评估结果
                if (
                    config.habitat_baselines.il.eval_save_results
                    and t
                    % config.habitat_baselines.il.eval_save_results_interval
                    == 0
                ):
                    q_string = q_vocab_dict.token_idx_2_string(question[0])
                    logger.info("Question: {}".format(q_string))

                    self._save_nav_results(
                        checkpoint_path,
                        t,
                        question,
                        imgs,
                        q_vocab_dict,
                        results_dir,
                        writer,
                        video_option,
                    )
