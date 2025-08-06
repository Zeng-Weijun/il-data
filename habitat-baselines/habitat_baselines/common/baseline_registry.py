#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""BaselineRegistry is extended from habitat.Registry to provide
registration for trainer and policies, while keeping Registry
in habitat core intact.

Import the baseline registry object using

.. code:: py

    from habitat_baselines.common.baseline_registry import baseline_registry

Various decorators for registry different kind of classes with unique keys

-   Register a trainer: ``@baseline_registry.register_trainer``
-   Register a policy: ``@baseline_registry.register_policy``
"""

from typing import Optional

from habitat.core.registry import Registry


class BaselineRegistry(Registry):
    # 继承自Registry类,用于注册训练器和策略
    
    @classmethod
    def register_trainer(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a RL training algorithm to registry with key 'name'.

        Args:
            name: Key with which the trainer will be registered.
                If None will use the name of the class.

        """
        from habitat_baselines.common.base_trainer import BaseTrainer

        return cls._register_impl(
            "trainer", to_register, name, assert_type=BaseTrainer
        )
    # 注册训练器的装饰器方法

    @classmethod
    def get_trainer(cls, name):
        return cls._get_impl("trainer", name)
    # 获取已注册的训练器

    @classmethod
    def register_policy(cls, to_register=None, *, name: Optional[str] = None):
        r"""使用name注册一个强化学习策略。

        :param name: 用于注册策略的键值。
            如果为None，将使用类的名称

        .. code:: py

            from habitat_baselines.rl.ppo.policy import Policy
            from habitat_baselines.common.baseline_registry import (
                baseline_registry
            )

            @baseline_registry.register_policy
            class MyPolicy(Policy):
                pass


            # 或者

            @baseline_registry.register_policy(name="MyPolicyName")
            class MyPolicy(Policy):
                pass

        """
        from habitat_baselines.rl.ppo.policy import Policy

        return cls._register_impl(
            "policy", to_register, name, assert_type=Policy
        )
    # 注册策略的装饰器方法

    @classmethod
    def get_policy(cls, name: str):
        r"""获取名为name的强化学习策略。"""
        return cls._get_impl("policy", name)
    # 获取已注册的策略

    @classmethod
    def register_obs_transformer(
        cls, to_register=None, *, name: Optional[str] = None
    ):
        r"""使用name注册一个观察转换器。

        :param name: 用于注册观察转换器的键值。
            如果为None，将使用类的名称

        .. code:: py

            from habitat_baselines.common.obs_transformers import ObservationTransformer
            from habitat_baselines.common.baseline_registry import (
                baseline_registry
            )

            @baseline_registry.register_policy
            class MyObsTransformer(ObservationTransformer):
                pass


            # 或者

            @baseline_registry.register_policy(name="MyTransformer")
            class MyObsTransformer(ObservationTransformer):
                pass

        """
        from habitat_baselines.common.obs_transformers import (
            ObservationTransformer,
        )

        return cls._register_impl(
            "obs_transformer",
            to_register,
            name,
            assert_type=ObservationTransformer,
        )
    # 注册观察转换器的装饰器方法

    @classmethod
    def get_obs_transformer(cls, name: str):
        r"""获取名为name的观察转换器。"""
        return cls._get_impl("obs_transformer", name)
    # 获取已注册的观察转换器

    @classmethod
    def register_auxiliary_loss(
        cls, to_register=None, *, name: Optional[str] = None
    ):
        return cls._register_impl("aux_loss", to_register, name)
    # 注册辅助损失函数的装饰器方法

    @classmethod
    def get_auxiliary_loss(cls, name: str):
        return cls._get_impl("aux_loss", name)
    # 获取已注册的辅助损失函数

    @classmethod
    def register_storage(cls, to_register=None, *, name: Optional[str] = None):
        """
        注册数据存储器，用于在训练器中存储策略展开的数据，
        并为更新器获取数据批次。
        """

        return cls._register_impl("storage", to_register, name)
    # 注册数据存储器的装饰器方法

    @classmethod
    def get_storage(cls, name: str):
        return cls._get_impl("storage", name)
    # 获取已注册的数据存储器

    @classmethod
    def register_agent_access_mgr(
        cls, to_register=None, *, name: Optional[str] = None
    ):
        """
        注册一个智能体访问管理器，用于训练器的接口交互。用法：
        ```
        @baseline_registry.register_agent_access_mgr
        class ExampleAgentAccessMgr:
            pass
        ```
        或者使用name参数覆盖默认名称：
        ```
        @baseline_registry.register_agent_access_mgr(name="MyAgentAccessMgr")
        class ExampleAgentAccessMgr:
            pass
        ```
        """
        from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr

        return cls._register_impl(
            "agent", to_register, name, assert_type=AgentAccessMgr
        )
    # 注册智能体访问管理器的装饰器方法

    @classmethod
    def get_agent_access_mgr(cls, name: str):
        return cls._get_impl("agent", name)
    # 获取已注册的智能体访问管理器

    @classmethod
    def register_updater(cls, to_register=None, *, name: Optional[str] = None):
        """
        注册一个策略更新器。
        """

        return cls._register_impl("updater", to_register, name)
    # 注册策略更新器的装饰器方法

    @classmethod
    def get_updater(cls, name: str):
        return cls._get_impl("updater", name)
    # 获取已注册的策略更新器


baseline_registry = BaselineRegistry()
# 创建BaselineRegistry的实例
