# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner, HIMOnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import (
    get_args,
    update_cfg_from_args,
    class_to_dict,
    get_load_path,
    set_seed,
    parse_sim_params,
)
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(
        self,
        name: str,
        task_class: VecEnv,
        env_cfg: LeggedRobotCfg,
        train_cfg: LeggedRobotCfgPPO,
    ):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.一个字符串，表示已注册环境的名称。
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
                                    一个可选参数，表示 Isaac Gym 的命令行参数。如果为 None，则会调用 get_args() 获取参数。
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.
                                    一个可选参数，表示环境的配置文件，用于覆盖已注册的配置。如果为 None，则使用默认配置。
        Raises:
            ValueError: Error if no registered env corresponds to 'name'

        Returns:
            isaacgym.VecTaskPython: The created environment 创建的环境对象
            Dict: the corresponding config file 对应的配置文件
        """
        # 如果没有传递 args 参数，则获取命令行参数
        if args is None:
            args = get_args()

        # 检查是否有注册的环境名称
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")

        # 如果没有传递 env_cfg 参数，则加载配置文件
        if env_cfg is None:
            env_cfg, _ = self.get_cfgs(name)

        # 从 args 中覆盖配置（如果指定了的话）
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        
        # 解析仿真参数（首先转换为字典）
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        
        # 创建环境对象
        env = task_class(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            sim_device=args.sim_device,
            headless=args.headless,
        )
        return env, env_cfg

    def make_alg_runner(
        self, env, name=None, args=None, train_cfg=None, log_root="default"
    ) -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm) 一个环境对象，表示要训练的环境。
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None. 一个可选参数，表示已注册环境的名称。如果为 None，则使用配置文件。
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None. 一个可选参数，表示 Isaac Gym 的命令行参数。如果为 None，则会调用 get_args() 获取参数。
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None. 一个可选参数，表示训练的配置文件。如果为 None，则使用 name 获取配置文件。
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 一个可选参数，表示 Tensorboard 的日志目录。默认值为 "default"，表示使用默认路径。
                                    Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm 创建的训练算法对象
            Dict: the corresponding config file 对应的配置文件
        """
        # 如果没有传递 args 参数，则获取命令行参数
        if args is None:
            args = get_args()
        # 如果没有传递 train_cfg 参数，则使用 name 获取配置文件
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # 加载配置文件
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
                
        # 从 args 中覆盖配置（如果指定了的话）
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        # 设置日志目录
        if log_root == "default":
            log_root = os.path.join(
                LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name
            )
            log_dir = os.path.join(
                log_root,
                datetime.now().strftime("%b%d_%H-%M-%S")
                + "_"
                + train_cfg.runner.run_name,
            )
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(
                log_root,
                datetime.now().strftime("%b%d_%H-%M-%S")
                + "_"
                + train_cfg.runner.run_name,
            )
            
        # 将配置转换为字典
        train_cfg_dict = class_to_dict(train_cfg)
        
        # 创建训练算法对象
        runner = HIMOnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
        
        # 保存恢复路径并在创建新日志目录之前加载模型
        resume = train_cfg.runner.resume
        if resume:
            # 加载之前训练的模型
            resume_path = get_load_path(
                log_root,
                load_run=train_cfg.runner.load_run,
                checkpoint=train_cfg.runner.checkpoint,
            )
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
            
        # 返回训练算法对象和配置文件
        return runner, train_cfg


# make global task registry
task_registry = TaskRegistry()
