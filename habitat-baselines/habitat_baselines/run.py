#!/usr/bin/env python3  # 指定Python解释器路径

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random  # 导入随机数模块
import sys  # 导入系统模块
from typing import TYPE_CHECKING  # 导入类型检查相关

import hydra  # 导入hydra配置管理库
import numpy as np  # 导入numpy科学计算库
import torch  # 导入PyTorch深度学习框架

from habitat.config.default import patch_config  # 导入配置补丁函数
from habitat.config.default_structured_configs import register_hydra_plugin  # 导入hydra插件注册函数
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,  # 导入Habitat基线配置插件
)

if TYPE_CHECKING:
    from omegaconf import DictConfig  # 类型检查时导入DictConfig类型

## for import functions related to falcon
import falcon  # 导入falcon相关功能

@hydra.main(  # hydra主函数装饰器
    version_base=None,
    config_path="config",  # 配置文件路径
    config_name="pointnav/ppo_pointnav_example",  # 配置文件名
)
def main(cfg: "DictConfig"):  # 主函数定义
    cfg = patch_config(cfg)  # 应用配置补丁
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")  # 根据配置执行评估或训练


def execute_exp(config: "DictConfig", run_type: str) -> None:  # 执行实验函数
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    random.seed(config.habitat.seed)  # 设置随机数种子
    np.random.seed(config.habitat.seed)  # 设置numpy随机数种子
    torch.manual_seed(config.habitat.seed)  # 设置PyTorch随机数种子
    if (
        config.habitat_baselines.force_torch_single_threaded  # 如果强制单线程
        and torch.cuda.is_available()  # 且CUDA可用
    ):
        torch.set_num_threads(1)  # 设置为单线程

    from habitat_baselines.common.baseline_registry import baseline_registry  # 导入基线注册表

    trainer_init = baseline_registry.get_trainer(  # 获取训练器初始化函数
        config.habitat_baselines.trainer_name
    )
    assert (  # 断言确保训练器支持
        trainer_init is not None
    ), f"{config.habitat_baselines.trainer_name} is not supported"
    trainer = trainer_init(config)  # 初始化训练器

    if run_type == "train":  # 如果是训练模式
        trainer.train()  # 执行训练
    elif run_type == "eval":  # 如果是评估模式
        trainer.eval()  # 执行评估


if __name__ == "__main__":  # 主程序入口
    register_hydra_plugin(HabitatBaselinesConfigPlugin)  # 注册Habitat基线配置插件
    if "--exp-config" in sys.argv or "--run-type" in sys.argv:  # 检查命令行参数
        raise ValueError(  # 如果使用旧API，抛出错误
            "The API of run.py has changed to be compatible with hydra.\n"
            "--exp-config is now --config-name and is a config path inside habitat-baselines/habitat_baselines/config/. \n"
            "--run-type train is replaced with habitat_baselines.evaluate=False (default) and --run-type eval is replaced with habitat_baselines.evaluate=True.\n"
            "instead of calling:\n\n"
            "python -u -m habitat_baselines.run --exp-config habitat-baselines/habitat_baselines/config/<path-to-config> --run-type train/eval\n\n"
            "You now need to do:\n\n"
            "python -u -m habitat_baselines.run --config-name=<path-to-config> habitat_baselines.evaluate=False/True\n"
        )
    main()  # 执行主函数
