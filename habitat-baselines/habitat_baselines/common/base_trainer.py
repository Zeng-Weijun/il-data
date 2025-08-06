#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from typing import TYPE_CHECKING, ClassVar, Dict, List

import torch

from habitat import logger
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer,
)
from habitat_baselines.rl.ddppo.ddp_utils import (
    SAVE_STATE,
    add_signal_handlers,
    is_slurm_batch_job,
    load_resume_state,
    save_resume_state,
)
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BaseTrainer:
    r"""通用训练器类,作为更具体的训练器类(如RL训练器、SLAM或模仿学习器)的基础模板。
    仅包含最基本的功能。
    """
    config: "DictConfig"
    flush_secs: float
    supported_tasks: ClassVar[List[str]]

    def train(self) -> None:
        raise NotImplementedError

    def _get_resume_state_config_or_new_config(
        self, resume_state_config: "DictConfig"
    ):
        # 根据配置决定是使用恢复状态的配置还是新配置
        if self.config.habitat_baselines.load_resume_state_config:
            if self.config != resume_state_config:
                logger.warning(
                    "\n##################\n"
                    "You are attempting to resume training with a different "
                    "configuration than the one used for the original training run. "
                    "Since load_resume_state_config=True, the ORIGINAL configuration "
                    "will be used and the new configuration will be IGNORED."
                    "##################\n"
                )
            return resume_state_config
        return self.config.copy()

    def _add_preemption_signal_handlers(self):
        # 添加抢占信号处理器
        if is_slurm_batch_job():
            add_signal_handlers()

    def eval(self) -> None:
        r"""训练器评估的主要方法。调用在BaseRLTrainer或BaseILTrainer中
        指定的_eval_checkpoint()方法

        Returns:
            None
        """

        self._add_preemption_signal_handlers()

        resume_state = load_resume_state(self.config, filename_key="eval")
        if resume_state is not None:
            # 如果有保存的恢复状态,说明我们正在恢复一个被抢占的评估会话
            # 我们获取配置和prev_ckpt_ind,以便从上次中断的检查点继续
            self.config = self._get_resume_state_config_or_new_config(
                resume_state["config"]
            )
            prev_ckpt_ind = resume_state["prev_ckpt_ind"]
        else:
            prev_ckpt_ind = -1

        self.device = (
            torch.device("cuda", self.config.habitat_baselines.torch_gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.habitat_baselines.eval.video_option:
            assert (
                len(self.config.habitat_baselines.tensorboard_dir) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(
                self.config.habitat_baselines.tensorboard_dir, exist_ok=True
            )
        if "disk" in self.config.habitat_baselines.eval.video_option:
            assert (
                len(self.config.habitat_baselines.video_dir) > 0
            ), "Must specify a directory for storing videos on disk"

        with get_writer(self.config, flush_secs=self.flush_secs) as writer:
            if (
                os.path.isfile(
                    self.config.habitat_baselines.eval_ckpt_path_dir
                )
                or not self.config.habitat_baselines.eval.should_load_ckpt
            ):
                # 评估单个检查点。如果should_load_ckpt=False,
                # 则eval_ckpt_path_dir将被忽略

                if self.config.habitat_baselines.eval.should_load_ckpt:
                    proposed_index = get_checkpoint_id(
                        self.config.habitat_baselines.eval_ckpt_path_dir
                    )
                else:
                    proposed_index = None

                if proposed_index is not None:
                    ckpt_idx = proposed_index
                else:
                    ckpt_idx = 0
                self._eval_checkpoint(
                    self.config.habitat_baselines.eval_ckpt_path_dir,
                    writer,
                    checkpoint_index=ckpt_idx,
                )
            else:
                # 按顺序评估多个检查点
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        current_ckpt = poll_checkpoint_folder(
                            self.config.habitat_baselines.eval_ckpt_path_dir,
                            prev_ckpt_ind,
                        )
                        time.sleep(2)  # 轮询前休眠2秒
                    logger.info(f"=======current_ckpt: {current_ckpt}=======")  # type: ignore
                    prev_ckpt_ind += 1
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind,
                    )

                    # 在评估过程中保存恢复状态,以便在作业被抢占时可以恢复评估
                    save_resume_state(
                        {
                            "config": self.config,
                            "prev_ckpt_ind": prev_ckpt_ind,
                        },
                        self.config,
                        filename_key="eval",
                    )

                    if (
                        prev_ckpt_ind + 1
                    ) == self.config.habitat_baselines.num_checkpoints:
                        break

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError


class BaseRLTrainer(BaseTrainer):
    r"""RL训练器的基类。未来的RL特定方法应该托管在这里。

    属性说明:
        device: PyTorch设备对象,用于指定模型运行的硬件设备(CPU/GPU)
        config: 配置字典,包含所有训练相关的配置参数
        video_option: 视频录制选项列表
        num_updates_done: 已完成的更新次数
        num_steps_done: 已完成的训练步数
        _flush_secs: tensorboard写入磁盘的刷新间隔(秒)
        _last_checkpoint_percent: 上一次保存检查点时的训练进度百分比
    """
    device: torch.device  # type: ignore
    config: "DictConfig"
    video_option: List[str]
    num_updates_done: int
    num_steps_done: int
    _flush_secs: int
    _last_checkpoint_percent: float

    def __init__(self, config: "DictConfig") -> None:
        super().__init__()
        assert config is not None, "needs config file to initialize trainer"
        self.config = config
        self._flush_secs = 30  # 默认30秒刷新一次tensorboard
        self.num_updates_done = 0  # 初始化更新次数为0
        self.num_steps_done = 0  # 初始化训练步数为0
        self._last_checkpoint_percent = -1.0  # 初始化检查点进度为-1
        # 检查更新次数和总步数的配置
        # 检查训练更新次数和总步数的配置
        # 两个参数不能同时指定,必须有一个为-1
        if (
            config.habitat_baselines.num_updates != -1
            and config.habitat_baselines.total_num_steps != -1
        ):
            raise RuntimeError(
                "num_updates和total_num_steps不能同时指定,必须有一个为-1\n"
                " num_updates: {} total_num_steps: {}".format(
                    config.habitat_baselines.num_updates,
                    config.habitat_baselines.total_num_steps,
                )
            )

        # 两个参数不能同时为-1,必须指定其中一个
        if (
            config.habitat_baselines.num_updates == -1
            and config.habitat_baselines.total_num_steps == -1
        ):
            raise RuntimeError(
                "num_updates和total_num_steps必须指定其中一个\n"
                " num_updates: {} total_num_steps: {}".format(
                    config.habitat_baselines.num_updates,
                    config.habitat_baselines.total_num_steps,
                )
            )

        # 检查检查点相关配置
        # num_checkpoints和checkpoint_interval不能同时指定
        if (
            config.habitat_baselines.num_checkpoints != -1
            and config.habitat_baselines.checkpoint_interval != -1
        ):
            raise RuntimeError(
                "num_checkpoints和checkpoint_interval不能同时指定,必须有一个为-1\n"
                " num_checkpoints: {} checkpoint_interval: {}".format(
                    config.habitat_baselines.num_checkpoints,
                    config.habitat_baselines.checkpoint_interval,
                )
            )

        # 必须指定num_checkpoints和checkpoint_interval其中一个
        if (
            config.habitat_baselines.num_checkpoints == -1
            and config.habitat_baselines.checkpoint_interval == -1
        ):
            raise RuntimeError(
                "必须指定num_checkpoints和checkpoint_interval其中一个\n"
                " num_checkpoints: {} checkpoint_interval: {}".format(
                    config.habitat_baselines.num_checkpoints,
                    config.habitat_baselines.checkpoint_interval,
                )
            )

    def percent_done(self) -> float:
        # 根据配置的参数计算训练进度百分比
        # 如果指定了num_updates,使用更新次数计算
        if self.config.habitat_baselines.num_updates != -1:
            return (
                self.num_updates_done
                / self.config.habitat_baselines.num_updates
            )
        # 否则使用步数计算
        else:
            return (
                self.num_steps_done
                / self.config.habitat_baselines.total_num_steps
            )

    def is_done(self) -> bool:
        # 通过计算进度百分比判断训练是否完成
        return self.percent_done() >= 1.0

    def should_checkpoint(self) -> bool:
        # 判断当前是否需要保存检查点
        needs_checkpoint = False
        # 如果指定了检查点数量,按百分比间隔保存
        if self.config.habitat_baselines.num_checkpoints != -1:
            checkpoint_every = (
                1 / self.config.habitat_baselines.num_checkpoints
            )
            if (
                self._last_checkpoint_percent + checkpoint_every
                < self.percent_done()
            ):
                needs_checkpoint = True
                self._last_checkpoint_percent = self.percent_done()
        # 否则按更新次数间隔保存
        else:
            needs_checkpoint = (
                self.num_updates_done
                % self.config.habitat_baselines.checkpoint_interval
            ) == 0

        return needs_checkpoint

    def _should_save_resume_state(self) -> bool:
        # 判断是否需要保存恢复状态
        # 当SAVE_STATE标志被设置,或者满足批处理作业和保存间隔条件时保存
        return SAVE_STATE.is_set() or (
            (
                not self.config.habitat_baselines.rl.preemption.save_state_batch_only
                or is_slurm_batch_job()
            )
            and (
                (
                    int(self.num_updates_done + 1)
                    % self.config.habitat_baselines.rl.preemption.save_resume_state_interval
                )
                == 0
            )
        )

    @property
    def flush_secs(self):
        # 获取tensorboard刷新间隔
        return self._flush_secs

    @flush_secs.setter 
    def flush_secs(self, value: int):
        # 设置tensorboard刷新间隔
        self._flush_secs = value

    def train(self) -> None:
        # 训练方法,需要在子类中实现
        raise NotImplementedError

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""评估单个检查点。训练器算法应该实现这个方法。

        Args:
            checkpoint_path: 检查点路径
            writer: 用于记录到tensorboard的writer对象
            checkpoint_index: 当前检查点的索引,用于日志记录

        Returns:
            None
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        # 保存检查点方法,需要在子类中实现
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        # 加载检查点方法,需要在子类中实现
        raise NotImplementedError
