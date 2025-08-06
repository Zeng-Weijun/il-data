#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import contextlib
import os
import random
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import habitat_baselines.rl.multi_agent  # noqa: F401.
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.utils import profiling_wrapper
from habitat_baselines.common import VectorEnvFactory
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer,
)
from habitat_baselines.rl.ddppo.algo import DDPPO  # noqa: F401.
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr
from habitat_baselines.rl.ppo.evaluator import Evaluator
from habitat_baselines.rl.ppo.single_agent_access_mgr import (  # noqa: F401.
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import (
    batch_obs,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    NON_SCALAR_METRICS,
    extract_scalars_from_infos,
)
from habitat_baselines.utils.timing import g_timer

def contains_inf_or_nan(observations):
    """检查观测值中是否包含无穷大或NaN值
    Args:
        observations: 观测值字典
    Returns:
        bool: 是否包含无穷大或NaN值
    """
    for key, value in observations.items():
        if isinstance(value, (float, int)):
            # 如果是标量，检查是否为 NaN 或 inf
            if math.isinf(value) or math.isnan(value):
                print(f"Key {key} contains inf or nan: {value}")
                return True
        elif isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
            # 如果是列表、数组或张量，检查每个元素是否为 NaN 或 inf
            if isinstance(value, torch.Tensor):
                if torch.isinf(value).any() or torch.isnan(value).any():
                    print(f"Key {key} contains inf or nan in tensor")
                    return True
            elif isinstance(value, np.ndarray):
                if np.isinf(value).any() or np.isnan(value).any():
                    print(f"Key {key} contains inf or nan in numpy array")
                    return True
            else:
                for element in value:
                    if isinstance(element, (float, int)) and (math.isinf(element) or math.isnan(element)):
                        print(f"Key {key} contains inf or nan in list/tuple: {element}")
                        return True
    return False
# 作用:用于检测输入的观测值字典中是否包含无穷大或NaN值,帮助调试训练过程中的数值问题

@baseline_registry.register_trainer(name="falcon_trainer") 
class FalconTrainer(BaseRLTrainer):
    """Falcon算法的训练器类"""
    supported_tasks = ["Nav-v0"]  # 支持的任务类型

    SHORT_ROLLOUT_THRESHOLD: float = 0.25  # 短rollout的阈值
    _is_distributed: bool  # 是否为分布式训练
    envs: VectorEnv  # 向量化环境
    _env_spec: Optional[EnvironmentSpec]  # 环境规格

    def __init__(self, config=None):
        """初始化FalconTrainer
        Args:
            config: 配置对象
        """
        super().__init__(config)

        self._agent = None  # 智能体
        self.envs = None  # 环境
        self.obs_transforms = []  # 观测值变换
        self._is_static_encoder = False  # 是否使用静态编码器
        self._encoder = None  # 编码器
        self._env_spec = None  # 环境规格

        # 如果world size大于1则为分布式
        self._is_distributed = get_distrib_size()[2] > 1
        
        # 模仿学习数据保存配置
        self._save_imitation_data = getattr(config.habitat_baselines, 'save_imitation_data', False)
        self._imitation_data_dir = getattr(config.habitat_baselines, 'imitation_data_dir', './imitation_data')
        self._save_frequency = getattr(config.habitat_baselines, 'save_frequency', 10)  # 每10个更新保存一次
        
        if self._save_imitation_data:
            os.makedirs(self._imitation_data_dir, exist_ok=True)
            print(f"[DEBUG] 启用模仿学习数据保存，保存目录: {self._imitation_data_dir}")
# 作用:Falcon算法的训练器类,继承自BaseRLTrainer,实现了训练循环和评估等核心功能

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        """分布式训练中的all_reduce操作辅助方法
        Args:
            t: 需要进行all_reduce的张量
        Returns:
            处理后的张量
        """
        if not self._is_distributed:
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)
# 作用:在分布式训练中对张量进行all_reduce操作,用于同步各进程间的梯度等信息

    def _save_rollout_data_for_imitation(self, update_count: int):
        """保存rollout数据用于模仿学习预训练
        Args:
            update_count: 当前更新次数
        """
        if not self._save_imitation_data or not rank0_only():
            return
            
        if update_count % self._save_frequency != 0:
            return
            
        try:
            # 获取rollouts数据
            rollouts = self._agent.rollouts
            print(f"[DEBUG] Rollouts类型: {type(rollouts).__name__}")
            
            if hasattr(rollouts, '_active_storages'):
                # 多智能体情况
                print(f"[DEBUG] 检测到MultiStorage，活跃存储数量: {len(rollouts._active_storages)}")
                active_storage = rollouts._active_storages[0]
                print(f"[DEBUG] 活跃存储类型: {type(active_storage).__name__}")
            else:
                # 单智能体情况
                print(f"[DEBUG] 单智能体存储")
                active_storage = rollouts
                
            # 检查buffers是否存在
            if not hasattr(active_storage, 'buffers'):
                print(f"[ERROR] 存储对象没有buffers属性")
                return
                
            buffers = active_storage.buffers
            print(f"[DEBUG] Buffers类型: {type(buffers).__name__}")
            print(f"[DEBUG] Buffers包含的键: {list(buffers.keys())}")
            
            # 准备保存的数据
            imitation_data = {
                'observations': {},
                'actions': None,
                'rewards': None,
                'masks': None,
                'info_data': None,  # 新增：完整的info信息
                'running_episode_stats': None,  # 新增：运行时统计信息
                'update_count': update_count,
                'num_steps': self._ppo_cfg.num_steps,
                'num_envs': self.envs.num_envs
            }
            
            # 保存观测数据
            if 'observations' in buffers:
                print(f"[DEBUG] 观测数据包含传感器: {list(buffers['observations'].keys())}")
                for sensor_name, sensor_data in buffers['observations'].items():
                    print(f"[DEBUG] 处理传感器 {sensor_name}，原始形状: {sensor_data.shape}")
                    # 只保存有效的rollout步数，去掉padding
                    valid_data = sensor_data[:self._ppo_cfg.num_steps].cpu().clone()
                    imitation_data['observations'][sensor_name] = valid_data
                    print(f"[DEBUG] 保存传感器 {sensor_name} 数据，形状: {valid_data.shape}")
            else:
                print(f"[DEBUG] 警告: buffers中没有observations键")
            
            # 保存动作数据
            if 'actions' in buffers:
                print(f"[DEBUG] 动作数据原始形状: {buffers['actions'].shape}")
                imitation_data['actions'] = buffers['actions'][:self._ppo_cfg.num_steps].cpu().clone()
                print(f"[DEBUG] 保存动作数据，形状: {imitation_data['actions'].shape}")
            else:
                print(f"[DEBUG] 警告: buffers中没有actions键")
            
            # 保存奖励数据
            if 'rewards' in buffers:
                print(f"[DEBUG] 奖励数据原始形状: {buffers['rewards'].shape}")
                imitation_data['rewards'] = buffers['rewards'][:self._ppo_cfg.num_steps].cpu().clone()
                print(f"[DEBUG] 保存奖励数据，形状: {imitation_data['rewards'].shape}")
            else:
                print(f"[DEBUG] 警告: buffers中没有rewards键")
            
            # 保存mask数据（用于标识episode结束）
            if 'masks' in buffers:
                print(f"[DEBUG] Mask数据原始形状: {buffers['masks'].shape}")
                imitation_data['masks'] = buffers['masks'][:self._ppo_cfg.num_steps].cpu().clone()
                print(f"[DEBUG] 保存mask数据，形状: {imitation_data['masks'].shape}")
            else:
                print(f"[DEBUG] 警告: buffers中没有masks键")
            
            # 直接保存完整的info信息
            if hasattr(self, '_single_proc_infos') and self._single_proc_infos:
                print(f"[DEBUG] 保存完整的_single_proc_infos: {list(self._single_proc_infos.keys())}")
                # 将所有info数据转换为tensor并保存
                info_data = {}
                for key, value in self._single_proc_infos.items():
                    if isinstance(value, list):
                        info_data[key] = torch.tensor(value)
                    else:
                        info_data[key] = torch.tensor([value])
                    print(f"[DEBUG] 保存info数据 {key}，形状: {info_data[key].shape}")
                imitation_data['info_data'] = info_data
            
            # 保存运行时统计信息
            if hasattr(self, 'running_episode_stats') and self.running_episode_stats:
                print(f"[DEBUG] 保存完整的running_episode_stats: {list(self.running_episode_stats.keys())}")
                # 将所有运行时统计数据转换为CPU tensor并保存
                running_stats = {}
                for key, value in self.running_episode_stats.items():
                    if torch.is_tensor(value):
                        running_stats[key] = value.cpu().clone()
                    else:
                        running_stats[key] = torch.tensor(value)
                    print(f"[DEBUG] 保存运行时统计 {key}，形状: {running_stats[key].shape}")
                imitation_data['running_episode_stats'] = running_stats
            
            # 检查是否有任何数据被保存
            has_data = (len(imitation_data['observations']) > 0 or 
                       imitation_data['actions'] is not None or 
                       imitation_data['rewards'] is not None or 
                       imitation_data['masks'] is not None)
            
            if not has_data:
                print(f"[WARNING] 没有找到任何可保存的数据，跳过保存")
                return
            
            # 为每次更新创建独立的文件夹
            update_folder = os.path.join(self._imitation_data_dir, f'update_{update_count:04d}')
            os.makedirs(update_folder, exist_ok=True)
            print(f"[DEBUG] 创建更新文件夹: {update_folder}")
            
            # 保存到文件
            save_path = os.path.join(update_folder, f'rollout_data_update_{update_count}.pt')
            torch.save(imitation_data, save_path)
            print(f"[DEBUG] 模仿学习数据已保存到: {save_path}")
            
            # 保存数据统计信息
            stats_path = os.path.join(update_folder, f'rollout_stats_update_{update_count}.txt')
            with open(stats_path, 'w') as f:
                f.write(f"Update: {update_count}\n")
                f.write(f"Num steps: {self._ppo_cfg.num_steps}\n")
                f.write(f"Num envs: {self.envs.num_envs}\n")
                f.write(f"Observations sensors: {list(imitation_data['observations'].keys())}\n")
                if imitation_data['actions'] is not None:
                    f.write(f"Actions shape: {imitation_data['actions'].shape}\n")
                if imitation_data['rewards'] is not None:
                    f.write(f"Rewards shape: {imitation_data['rewards'].shape}\n")
                    f.write(f"Mean reward: {imitation_data['rewards'].mean().item():.4f}\n")
                    f.write(f"Total reward: {imitation_data['rewards'].sum().item():.4f}\n")
                if imitation_data['info_data'] is not None:
                    f.write(f"Info data keys: {list(imitation_data['info_data'].keys())}\n")
                    for key, value in imitation_data['info_data'].items():
                        f.write(f"  {key} shape: {value.shape}\n")
                if imitation_data['running_episode_stats'] is not None:
                    f.write(f"Running episode stats keys: {list(imitation_data['running_episode_stats'].keys())}\n")
                    for key, value in imitation_data['running_episode_stats'].items():
                        f.write(f"  {key} shape: {value.shape}\n")
            
        except Exception as e:
            print(f"[ERROR] 保存模仿学习数据时出错: {e}")
            import traceback
            traceback.print_exc()

    def _create_obs_transforms(self):
        """创建观测值变换"""
        self.obs_transforms = get_active_obs_transforms(self.config)
        self._env_spec.observation_space = apply_obs_transforms_obs_space(
            self._env_spec.observation_space, self.obs_transforms
        )
# 作用:创建并应用观测值变换,修改环境的观测空间

    def _create_agent(self, resume_state, **kwargs) -> AgentAccessMgr:
        """设置AgentAccessMgr
        Args:
            resume_state: 恢复状态
            **kwargs: 额外参数
        Returns:
            AgentAccessMgr实例
        """
        self._create_obs_transforms()
        return baseline_registry.get_agent_access_mgr(
            self.config.habitat_baselines.rl.agent.type
        )(
            config=self.config,
            env_spec=self._env_spec,
            is_distrib=self._is_distributed,
            device=self.device,
            resume_state=resume_state,
            num_envs=self.envs.num_envs,
            percent_done_fn=self.percent_done,
            **kwargs,
        )
# 作用:创建智能体访问管理器,用于管理智能体的训练和推理

    def _init_envs(self, config=None, is_eval: bool = False):
        """初始化环境
        Args:
            config: 配置对象
            is_eval: 是否为评估模式
        """
        print(f"[DEBUG] 开始初始化环境: {'评估模式' if is_eval else '训练模式'}")
        if config is None:
            config = self.config

        env_factory: VectorEnvFactory = hydra.utils.instantiate(
            config.habitat_baselines.vector_env_factory
        )
        print(f"[DEBUG] 创建向量化环境工厂: {type(env_factory).__name__}")
        
        self.envs = env_factory.construct_envs(
            config,
            workers_ignore_signals=is_slurm_batch_job(),
            enforce_scenes_greater_eq_environments=is_eval,
            is_first_rank=(
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ),
        )
        print(f"[DEBUG] 环境初始化完成, 环境数量: {self.envs.num_envs}")
        print(f"[DEBUG] 观测空间: {self.envs.observation_spaces[0]}")
        print(f"[DEBUG] 动作空间: {self.envs.action_spaces[0]}")

        self._env_spec = EnvironmentSpec(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            orig_action_space=self.envs.orig_action_spaces[0],
        )

        # 只在rank0上记录的测量指标
        self._rank0_keys: Set[str] = set(
            list(self.config.habitat.task.rank0_env0_measure_names)
            + list(self.config.habitat.task.rank0_measure_names)
        )

        # rank0上单独记录的信息
        self._single_proc_infos: Dict[str, List[float]] = {}
# 作用:初始化训练/评估环境,设置环境规格和测量指标
    def _init_train(self, resume_state=None):
        """初始化训练
        Args:
            resume_state: 恢复训练的状态,默认为None
        """
        print(f"[DEBUG] 开始初始化训练环境和智能体")
        # 如果没有传入resume_state,则尝试从配置中加载
        if resume_state is None:
            resume_state = load_resume_state(self.config)
            print(f"[DEBUG] 尝试从配置加载恢复状态: {'成功' if resume_state is not None else '无恢复状态'}")

        # 如果存在恢复状态但配置不允许加载,则报错
        if resume_state is not None:
            if not self.config.habitat_baselines.load_resume_state_config:
                print(f"[DEBUG] 错误: 配置不允许加载恢复状态但找到了之前的训练状态")
                raise FileExistsError(
                    f"配置中设置了不加载恢复状态(load_resume_state_config=False),但存在之前的训练运行。您可以删除检查点文件夹 {self.config.habitat_baselines.checkpoint_folder}, 或在新运行中更改配置键 habitat_baselines.checkpoint_folder。"
                )

            # 获取恢复状态的配置或新配置
            print(f"[DEBUG] 从恢复状态获取配置")
            self.config = self._get_resume_state_config_or_new_config(
                resume_state["config"]
            )

        # 如果配置强制使用分布式,则设置分布式标志
        if self.config.habitat_baselines.rl.ddppo.force_distributed:
            self._is_distributed = True

        # 添加抢占信号处理器
        self._add_preemption_signal_handlers()

        # 如果是分布式训练,进行分布式初始化
        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.habitat_baselines.rl.ddppo.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "使用 {} 个工作进程初始化 DD-PPO".format(
                        torch.distributed.get_world_size()
                    )
                )

            # 设置分布式相关配置
            with read_write(self.config):
                self.config.habitat_baselines.torch_gpu_id = local_rank
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
                    local_rank
                )
                # 乘以模拟器数量以确保它们也获得唯一的种子
                self.config.habitat.seed += (
                    torch.distributed.get_rank()
                    * self.config.habitat_baselines.num_environments
                )

            # 设置随机种子
            random.seed(self.config.habitat.seed)
            np.random.seed(self.config.habitat.seed)
            torch.manual_seed(self.config.habitat.seed)
            # 创建rollout追踪器
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        # 如果是rank0且verbose模式,打印配置信息
        if rank0_only() and self.config.habitat_baselines.verbose:
            logger.info(f"配置: {OmegaConf.to_yaml(self.config)}")

        # 配置性能分析包装器
        profiling_wrapper.configure(
            capture_start_step=self.config.habitat_baselines.profiling.capture_start_step,
            num_steps_to_capture=self.config.habitat_baselines.profiling.num_steps_to_capture,
        )

        # 移除非标量度量,因为它们只能在评估中使用
        for non_scalar_metric in NON_SCALAR_METRICS:
            non_scalar_metric_root = non_scalar_metric.split(".")[0]
            if non_scalar_metric_root in self.config.habitat.task.measurements:
                with read_write(self.config):
                    OmegaConf.set_struct(self.config, False)
                    self.config.habitat.task.measurements.pop(
                        non_scalar_metric_root
                    )
                    OmegaConf.set_struct(self.config, True)
                if self.config.habitat_baselines.verbose:
                    logger.info(
                        f"从度量中移除 {non_scalar_metric_root},因为它不能在训练期间使用。"
                    )

        # 初始化环境
        self._init_envs()

        # 获取设备
        self.device = get_device(self.config)

        # 如果是rank0且检查点文件夹不存在,则创建
        if rank0_only() and not os.path.isdir(
            self.config.habitat_baselines.checkpoint_folder
        ):
            os.makedirs(self.config.habitat_baselines.checkpoint_folder)

        # 添加日志文件处理器
        logger.add_filehandler(self.config.habitat_baselines.log_file)

        # 创建智能体
        self._agent = self._create_agent(resume_state)
        if self._is_distributed:
            self._agent.init_distributed(find_unused_params=False)
        self._agent.post_init()

        # 设置静态编码器标志和PPO配置
        self._is_static_encoder = (
            not self.config.habitat_baselines.rl.ddppo.train_encoder
        )
        self._ppo_cfg = self.config.habitat_baselines.rl.ppo

        # 重置环境并获取初始观察
        observations = self.envs.reset()
        observations = self.envs.post_step(observations)
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        
        # 初始化时检查RGB数据配置 - 已注释掉调试输出
        # rgb_keys = [key for key in batch.keys() if 'rgb' in key.lower()]
        # if rgb_keys:
        #     print(f"[初始化RGB检查] 成功配置RGB数据，包含以下RGB观察键: {rgb_keys}")
        #     for rgb_key in rgb_keys:
        #         rgb_data = batch[rgb_key]
        #         print(f"[初始化RGB检查] {rgb_key}: shape={rgb_data.shape}, dtype={rgb_data.dtype}, device={rgb_data.device}")
        # else:
        #     print("[初始化RGB检查] 警告: 未配置RGB数据，当前观察键包括:", list(batch.keys()))
        #     print("[初始化RGB检查] 提示: 如需RGB数据，请在配置文件中的obs_keys中添加RGB传感器键，如'agent_0_articulated_agent_jaw_rgb'")

        # 训练器和原始PPO训练器之间的关键修改
        # 如果使用静态编码器
        if self._is_static_encoder:
            # 获取智能体的视觉编码器
            self._encoder = self._agent.actor_critic.visual_encoder
            
            # 如果编码器为空
            if self._encoder is None:
                # 从第一个智能体获取视觉编码器
                self._encoder = self._agent._agents[0].actor_critic.visual_encoder
                # 使用推理模式
                with inference_mode():
                    # 移除batch中的"agent_0_"前缀
                    batch_temp = {key.replace('agent_0_', ''): value for key, value in batch.items()}
                    # 使用编码器处理batch_temp,并将结果存入batch中
                    batch[
                        'agent_0_' + PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch_temp)
            else:
                # 如果编码器不为空,直接使用推理模式处理batch
                with inference_mode():
                    batch[
                        PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch)
        
        # 插入第一个观察
        self._agent.rollouts.insert_first_observations(batch)

        # 初始化奖励和统计信息
        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=self._ppo_cfg.reward_window_size)
        )

        # 记录开始时间
        self.t_start = time.time()
    # 作用:初始化训练环境、智能体和各种训练所需的状态变量,是训练开始前的准备工作

    @rank0_only
    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        """保存检查点
        Args:
            file_name: 检查点文件名
            extra_state: 额外的状态信息,可选
        Returns:
            None
        """
        # 创建检查点字典
        checkpoint = {
            **self._agent.get_save_state(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        # 保存检查点
        save_file_path = os.path.join(
            self.config.habitat_baselines.checkpoint_folder, file_name
        )
        torch.save(checkpoint, save_file_path)
        # 同时保存为最新检查点
        torch.save(
            checkpoint,
            os.path.join(
                self.config.habitat_baselines.checkpoint_folder, "latest.pth"
            ),
        )
        # 如果配置了保存检查点的回调函数,则调用
        if self.config.habitat_baselines.on_save_ckpt_callback is not None:
            hydra.utils.call(
                self.config.habitat_baselines.on_save_ckpt_callback,
                save_file_path=save_file_path,
            )
    # 作用:保存模型检查点,包括模型状态和配置信息

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        """加载指定路径的检查点
        Args:
            checkpoint_path: 检查点文件路径
            *args: 额外的位置参数
            **kwargs: 额外的关键字参数
        Returns:
            包含检查点信息的字典
        """
        return torch.load(checkpoint_path, *args, **kwargs)
    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        """计算动作并执行环境步骤
        Args:
            buffer_index: 缓冲区索引,默认为0
        """
        # 获取环境数量
        num_envs = self.envs.num_envs
        # 计算环境切片范围
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        with g_timer.avg_time("trainer.sample_action"), inference_mode():
            # 获取当前步骤的批次数据
            step_batch = self._agent.rollouts.get_current_step(
                env_slice, buffer_index
            )

            profiling_wrapper.range_push("compute actions")

            # 获取批次长度信息
            step_batch_lens = {
                k: v
                for k, v in step_batch.items()
                if k.startswith("index_len")
            }
            # 使用actor_critic网络计算动作
            action_data = self._agent.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
                **step_batch_lens,
            )

        profiling_wrapper.range_pop()  # compute actions

        with g_timer.avg_time("trainer.obs_insert"):
            # 对每个环境执行动作
            for index_env, act in zip(
                range(env_slice.start, env_slice.stop),
                action_data.env_actions.cpu().unbind(0),
            ):
                # 根据动作空间类型处理动作
                if hasattr(self._agent, '_agents') and self._agent._agents[0]._actor_critic.action_distribution_type == 'categorical':
                    act = act.numpy()
                elif is_continuous_action_space(self._env_spec.action_space):
                    # 对连续动作进行裁剪
                    act = np.clip(
                        act.numpy(),
                        self._env_spec.action_space.low,
                        self._env_spec.action_space.high,
                    )
                else:
                    act = act.item()
                # 异步执行环境步骤
                self.envs.async_step_at(index_env, act)

        with g_timer.avg_time("trainer.obs_insert"):
            # 将动作数据插入到rollouts中
            self._agent.rollouts.insert(
                next_recurrent_hidden_states=action_data.rnn_hidden_states,
                actions=action_data.actions,
                action_log_probs=action_data.action_log_probs,
                value_preds=action_data.values,
                buffer_index=buffer_index,
                should_inserts=action_data.should_inserts,
                action_data=action_data,
            )
    # 作用:计算智能体的动作并在环境中执行,同时保存相关数据到rollouts中

    def _collect_environment_result(self, buffer_index: int = 0):
        """收集环境执行结果
        Args:
            buffer_index: 缓冲区索引,默认为0
        Returns:
            处理的环境数量
        """
        # 获取环境数量和切片范围
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        with g_timer.avg_time("trainer.step_env"):
            # 等待环境步骤完成并获取结果
            outputs = [
                self.envs.wait_step_at(index_env)
                for index_env in range(env_slice.start, env_slice.stop)
            ]

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

        with g_timer.avg_time("trainer.update_stats"):
            # 处理观测值
            observations = self.envs.post_step(observations)
            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
            
            # 检查RGB数据是否成功加载 - 已注释掉调试输出
            # rgb_keys = [key for key in batch.keys() if 'rgb' in key.lower()]
            # if rgb_keys:
            #     print(f"[RGB检查] 成功加载RGB数据，包含以下RGB观察键: {rgb_keys}")
            #     for rgb_key in rgb_keys:
            #         rgb_data = batch[rgb_key]
            #         print(f"[RGB检查] {rgb_key}: shape={rgb_data.shape}, dtype={rgb_data.dtype}, device={rgb_data.device}")
            #         # 转换为浮点数类型以计算统计信息
            #         rgb_float = rgb_data.float()
            #         print(f"[RGB检查] {rgb_key}: min={rgb_float.min().item():.3f}, max={rgb_float.max().item():.3f}, mean={rgb_float.mean().item():.3f}")
            # else:
            #     print("[RGB检查] 警告: 未检测到RGB数据，当前观察键包括:", list(batch.keys()))

            # 处理奖励和完成标志
            rewards = torch.tensor(
                rewards_l,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device=self.current_episode_reward.device,
            )
            done_masks = torch.logical_not(not_done_masks)

            # 更新统计信息
            self.current_episode_reward[env_slice] += rewards
            current_ep_reward = self.current_episode_reward[env_slice]
            self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))
            self.running_episode_stats["count"][env_slice] += done_masks.float()

            # 提取标量信息
            self._single_proc_infos = extract_scalars_from_infos(
                infos,
                ignore_keys=set(
                    k for k in infos[0].keys() if k not in self._rank0_keys
                ),
            )
            extracted_infos = extract_scalars_from_infos(
                infos, ignore_keys=self._rank0_keys
            )
            
            # 更新运行时统计信息
            for k, v_k in extracted_infos.items():
                v = torch.tensor(
                    v_k,
                    dtype=torch.float,
                    device=self.current_episode_reward.device,
                ).unsqueeze(1)
                if k not in self.running_episode_stats:
                    self.running_episode_stats[k] = torch.zeros_like(
                        self.running_episode_stats["count"]
                    )
                self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))

            # 重置已完成episode的奖励
            self.current_episode_reward[env_slice].masked_fill_(
                done_masks, 0.0
            )

        # 处理静态编码器
        if self._is_static_encoder:
            self._encoder = self._agent.actor_critic.visual_encoder
            if self._encoder is None:
                self._encoder = self._agent._agents[0].actor_critic.visual_encoder
                with inference_mode(), g_timer.avg_time("trainer.visual_features"):
                    batch_temp = {key.replace('agent_0_', ''): value for key, value in batch.items()}
                    batch[
                        'agent_0_' + PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch_temp)
            else:
                with inference_mode(), g_timer.avg_time("trainer.visual_features"):
                    batch[
                        PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch)

        # 将结果插入到rollouts中
        self._agent.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self._agent.rollouts.advance_rollout(buffer_index)

        return env_slice.stop - env_slice.start
    # 作用:收集环境执行的结果,更新统计信息,并将结果保存到rollouts中

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(self):
        """收集一个rollout步骤"""
        self._compute_actions_and_step_envs()
        return self._collect_environment_result()
    # 作用:执行一个完整的rollout步骤,包括计算动作和收集结果

    @profiling_wrapper.RangeContext("_update_agent")
    @g_timer.avg_time("trainer.update_agent")
    def _update_agent(self):
        """更新智能体"""
        print(f"[DEBUG] 进入_update_agent方法")
        
        with inference_mode():
            # 获取最后一步的数据
            step_batch = self._agent.rollouts.get_last_step()
            step_batch_lens = {
                k: v
                for k, v in step_batch.items()
                if k.startswith("index_len")
            }
            
            print(f"[DEBUG] 获取最后一步数据，batch keys: {list(step_batch.keys())}")

            # 计算下一个状态的价值
            next_value = self._agent.actor_critic.get_value(
                step_batch["observations"],
                step_batch.get("recurrent_hidden_states", None),
                step_batch["prev_actions"],
                step_batch["masks"],
                **step_batch_lens,
            )
            print(f"[DEBUG] 计算next_value完成，shape: {next_value.shape if hasattr(next_value, 'shape') else 'N/A'}")

        # 计算回报
        print(f"[DEBUG] 开始计算回报，GAE参数: use_gae={self._ppo_cfg.use_gae}, gamma={self._ppo_cfg.gamma}, tau={self._ppo_cfg.tau}")
        self._agent.rollouts.compute_returns(
            next_value,
            self._ppo_cfg.use_gae,
            self._ppo_cfg.gamma,
            self._ppo_cfg.tau,
        )
        print(f"[DEBUG] 回报计算完成")

        # 训练智能体
        print(f"[DEBUG] 开始PPO更新，设置为训练模式")
        self._agent.train()
        
        print(f"[DEBUG] 调用updater.update进行策略更新")
        losses = self._agent.updater.update(self._agent.rollouts)
        print(f"[DEBUG] PPO更新完成，损失类型: {type(losses)}, 损失键: {list(losses.keys()) if isinstance(losses, dict) else 'N/A'}")

        # 更新后的处理
        print(f"[DEBUG] 执行更新后清理")
        self._agent.rollouts.after_update()
        self._agent.after_update()
        
        print(f"[DEBUG] _update_agent方法完成")
        return losses
    # 作用:更新智能体的策略和价值网络

    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        """合并后处理步骤
        Args:
            losses: 损失字典
            count_steps_delta: 步数增量
        Returns:
            更新后的损失字典
        """
        # 整理统计信息
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        # 执行all_reduce操作
        stats = self._all_reduce(stats)

        # 更新窗口统计信息
        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        # 分布式处理
        if self._is_distributed:
            loss_name_ordering = sorted(losses.keys())
            stats = torch.tensor(
                [losses[k] for k in loss_name_ordering] + [count_steps_delta],
                device="cpu",
                dtype=torch.float32,
            )
            stats = self._all_reduce(stats)
            count_steps_delta = int(stats[-1].item())
            stats /= torch.distributed.get_world_size()

            losses = {
                k: stats[i].item() for i, k in enumerate(loss_name_ordering)
            }

        # 重置完成的rollout计数
        if self._is_distributed and rank0_only():
            self.num_rollouts_done_store.set("num_done", "0")

        # 更新总步数
        self.num_steps_done += count_steps_delta

        return losses
    # 作用:处理训练后的统计信息,在分布式环境中同步数据
    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        for k, v in losses.items():
            writer.add_scalar(f"learner/{k}", v, self.num_steps_done)

        for k, v in self._single_proc_infos.items():
            writer.add_scalar(k, np.mean(v), self.num_steps_done)

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)

        # Log perf metrics.
        writer.add_scalar("perf/fps", fps, self.num_steps_done)

        for timer_name, timer_val in g_timer.items():
            writer.add_scalar(
                f"perf/{timer_name}",
                timer_val.mean,
                self.num_steps_done,
            )

        # log stats
        if (
            self.num_updates_done % self.config.habitat_baselines.log_interval
            == 0
        ):
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                f"Num updates: {self.num_updates_done}\tNum frames {self.num_steps_done}"
            )

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )
            perf_stats_str = " ".join(
                [f"{k}: {v.mean:.3f}" for k, v in g_timer.items()]
            )
            logger.info(f"\tPerf Stats: {perf_stats_str}")
            if self.config.habitat_baselines.should_log_single_proc_infos:
                for k, v in self._single_proc_infos.items():
                    logger.info(f" - {k}: {np.mean(v):.3f}")

    def should_end_early(self, rollout_step) -> bool:
        if not self._is_distributed:
            return False
        # This is where the preemption of workers happens.  If a
        # worker detects it will be a straggler, it preempts itself!
        return (
            rollout_step
            >= self.config.habitat_baselines.rl.ppo.num_steps
            * self.SHORT_ROLLOUT_THRESHOLD
        ) and int(self.num_rollouts_done_store.get("num_done")) >= (
            self.config.habitat_baselines.rl.ddppo.sync_frac
            * torch.distributed.get_world_size()
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """
        print(f"[DEBUG] ========== 开始训练流程 ==========")
        print(f"[DEBUG] 训练配置: PPO步数={self.config.habitat_baselines.rl.ppo.num_steps}, 批次大小={self.config.habitat_baselines.rl.ppo.num_mini_batch}")
        print(f"[DEBUG] 学习率={self.config.habitat_baselines.rl.ppo.lr}, 熵系数={self.config.habitat_baselines.rl.ppo.entropy_coef}")
        
        resume_state = load_resume_state(self.config)
        print(f"[DEBUG] 加载恢复状态: {'成功' if resume_state is not None else '无恢复状态'}")
        self._init_train()

        count_checkpoints = 0
        prev_time = 0

        if self._is_distributed:
            torch.distributed.barrier()

        resume_run_id = None
        if resume_state is not None:
            self._agent.load_state_dict(resume_state)

            requeue_stats = resume_state["requeue_stats"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(
                requeue_stats["window_episode_stats"]
            )
            resume_run_id = requeue_stats.get("run_id", None)

        with (
            get_writer(
                self.config,
                resume_run_id=resume_run_id,
                flush_secs=self.flush_secs,
                purge_step=int(self.num_steps_done),
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            print(f"[DEBUG] 进入主训练循环")
            training_iteration = 0
            
            while not self.is_done():
                training_iteration += 1
                print(f"[DEBUG] ========== 训练迭代 {training_iteration} 开始 ==========")
                print(f"[DEBUG] 当前进度: {self.percent_done():.2%}, 已完成步数: {self.num_steps_done}")
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                self._agent.pre_rollout()

                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                        run_id=writer.get_run_id(),
                    )

                    save_resume_state(
                        dict(
                            **self._agent.get_resume_state(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                self._agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")
                
                # DEBUG: 开始rollout收集
                print(f"[DEBUG] 开始rollout收集，目标步数: {self._ppo_cfg.num_steps}")
                print(f"[DEBUG] 当前环境数量: {self.envs.num_envs}")

                profiling_wrapper.range_push("_collect_rollout_step")
                with g_timer.avg_time("trainer.rollout_collect"):
                    for buffer_index in range(self._agent.nbuffers):
                        print(f"[DEBUG] 初始化buffer {buffer_index}")
                        self._compute_actions_and_step_envs(buffer_index)

                    for step in range(self._ppo_cfg.num_steps):
                        is_last_step = (
                            self.should_end_early(step + 1)
                            or (step + 1) == self._ppo_cfg.num_steps
                        )
                        
                        # DEBUG: 每个rollout步骤的信息
                        if step % 10 == 0 or is_last_step:
                            print(f"[DEBUG] Rollout步骤 {step+1}/{self._ppo_cfg.num_steps}")

                        for buffer_index in range(self._agent.nbuffers):
                            count_steps_delta += (
                                self._collect_environment_result(buffer_index)
                            )
                            
                            # DEBUG: buffer收集信息 - 这行代码是一个调试注释,用于标记下面即将进行buffer数据收集的代码段
                            if step % 20 == 0:
                                print(f"[DEBUG] Buffer {buffer_index} 收集完成，累计步数增量: {count_steps_delta}")

                            if (buffer_index + 1) == self._agent.nbuffers:
                                profiling_wrapper.range_pop()  # _collect_rollout_step

                            if not is_last_step:
                                if (buffer_index + 1) == self._agent.nbuffers:
                                    profiling_wrapper.range_push(
                                        "_collect_rollout_step"
                                    )

                                self._compute_actions_and_step_envs(
                                    buffer_index
                                )

                        if is_last_step:
                            break

                profiling_wrapper.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                # DEBUG: 训练更新前的调试信息
                print(f"[DEBUG] 开始第 {self.num_updates_done + 1} 次更新，当前步数: {self.num_steps_done}")
                # MultiStorage doesn't have current_rollout_step_idx, skip this debug info
                # print(f"[DEBUG] 当前rollout步数: {self._agent.rollouts.current_rollout_step_idx}")
                
                # 保存rollout数据用于模仿学习（在更新前保存）
                self._save_rollout_data_for_imitation(self.num_updates_done + 1)
                
                losses = self._update_agent()
                
                # DEBUG: 训练更新后的调试信息
                print(f"[DEBUG] 更新完成，损失值: {losses}")
                print(f"[DEBUG] 梯度警告检查点 - 更新 {self.num_updates_done + 1}")

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    losses,
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    print(f'[DEBUG] PPO save to ckpt.{count_checkpoints}.pth - 更新次数: {self.num_updates_done}')
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update
                
                print(f"[DEBUG] ========== 训练迭代 {training_iteration} 结束 ==========")

            print(f"[DEBUG] 主训练循环结束，总迭代次数: {training_iteration}")
            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Some configurations require not to load the checkpoint, like when using
        # a hierarchial policy
        if self.config.habitat_baselines.eval.should_load_ckpt:
            # map_location="cpu" is almost always better than mapping to a CUDA device.
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )
            step_id = ckpt_dict["extra_state"]["step"]
            logger.info(f"Loaded checkpoint trained for {step_id} steps")
        else:
            ckpt_dict = {"config": None}

        if "config" not in ckpt_dict:
            ckpt_dict["config"] = None

        config = self._get_resume_state_config_or_new_config(
            ckpt_dict["config"]
        )
        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        if len(self.config.habitat_baselines.eval.video_option) > 0:
            n_agents = len(config.habitat.simulator.agents)
            for agent_i in range(n_agents):
                agent_name = config.habitat.simulator.agents_order[agent_i]
                agent_config = get_agent_config(
                    config.habitat.simulator, agent_i
                )

                agent_sensors = agent_config.sim_sensors
                extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
                with read_write(agent_sensors):
                    agent_sensors.update(extra_sensors)
                with read_write(config):
                    if config.habitat.gym.obs_keys is not None:
                        for render_view in extra_sensors.values():
                            if (
                                render_view.uuid
                                not in config.habitat.gym.obs_keys
                            ):
                                if n_agents > 1:
                                    config.habitat.gym.obs_keys.append(
                                        f"{agent_name}_{render_view.uuid}"
                                    )
                                else:
                                    config.habitat.gym.obs_keys.append(
                                        render_view.uuid
                                    )

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        self._init_envs(config, is_eval=True)

        self._agent = self._create_agent(None)
        if (
            self._agent.actor_critic.should_load_agent_state
            and self.config.habitat_baselines.eval.should_load_ckpt
        ):
            self._agent.load_state_dict(ckpt_dict)

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        evaluator = hydra.utils.instantiate(config.habitat_baselines.evaluator)
        assert isinstance(evaluator, Evaluator)
        evaluator.evaluate_agent(
            self._agent,
            self.envs,
            self.config,
            checkpoint_index,
            step_id,
            writer,
            self.device,
            self.obs_transforms,
            self._env_spec,
            self._rank0_keys,
        )

        self.envs.close()


def get_device(config: "DictConfig") -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda", config.habitat_baselines.torch_gpu_id)
        torch.cuda.set_device(device)
        return device
    else:
        return torch.device("cpu")

##################################################################################################################
# 模仿学习数据保存格式说明:
# imitation_data = {
#     "observations": {
#         # 来自机器人前视RGB相机，配置中的 obs_keys 包含 agent_0_articulated_agent_jaw_rgb 时才会生成。
#         # habitat 在每个 env.step()/reset() 返回的 observations 字典中提供，键名即 "rgb"。
#         "rgb": torch.Tensor,  # (num_steps, num_envs, H, W, C)

#         # 来自机器人前视深度相机，默认启用（agent_0_articulated_agent_jaw_depth）。
#         # 在 observations["depth"] 中返回，经过归一化后送入视觉编码器。
#         "depth": torch.Tensor,  # (num_steps, num_envs, H, W, 1)

#         # 来自 GPS 传感器，表示机器人在全局坐标系中的位置。
#         # 如果使用组合的 pointgoal 传感器，可从 observations["pointgoal"] 或 gps/compass 拆分得到。
#         "gps": torch.Tensor,  # (num_steps, num_envs, 3)
#         # 来自 Compass 传感器，表示机器人朝向。
#         # 同样可以与 gps 拼接形成 pointgoal 向量。
#         "compass": torch.Tensor,  # (num_steps, num_envs, 1)

#         # 其他传感器数据...
#         "pointgoal_with_gps_compass": torch.Tensor,  # (num_steps, num_envs, 3)
#     },
#     
#     # 每个时间步模型输出/记录的离散动作索引，来自策略的 sample() 或 mode()。
#     "actions": torch.Tensor,  # (num_steps, num_envs, 1)
#     
#     # 每个时间步的奖励值
#     "rewards": torch.Tensor,  # (num_steps, num_envs, 1)
#     
#     # 用于标识episode结束的mask
#     "masks": torch.Tensor,  # (num_steps, num_envs, 1)
#     
#     # 新增：完整的info信息，从_single_proc_infos中保存
#     # 包含所有环境返回的info数据，如碰撞、距离、奖励等指标
#     "info_data": {
#         "key1": torch.Tensor,  # 各种info指标
#         "key2": torch.Tensor,  # 用户可以根据需要提取任何指标
#         # ... 更多info数据
#     },
#     
#     # 新增：完整的运行时统计信息，从running_episode_stats中保存
#     # 包含训练过程中累积的所有统计数据
#     "running_episode_stats": {
#         "count": torch.Tensor,    # episode计数
#         "reward": torch.Tensor,   # 累积奖励
#         "key1": torch.Tensor,     # 其他统计指标
#         # ... 更多统计数据
#     },
#     
#     # 元数据
#     "update_count": int,  # 更新次数
#     "num_steps": int,     # rollout步数
#     "num_envs": int,      # 环境数量
# }
##################################################################################################################