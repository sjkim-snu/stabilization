# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


import gymnasium as gym

gym.register(
    id="Stabilization-Quadrotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "stabilization.tasks.manager_based.stabilization.runners.Runner:StabilizationEnvCfg",
        "rl_games_cfg_entry_point": "stabilization.tasks.manager_based.stabilization.agent:ppo_cfg.yaml",
    },
)