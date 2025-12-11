# /home/ksj/stabilization/source/stabilization/stabilization/__init__.py
import gymnasium as gym
# from . import agents

# gym.register(
#     id="Isaac-Quad-Stabilization-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": "stabilization.tasks.manager_based.stabilization.runners.TrainRunner:StabilizationEnvCfg",
#         "rl_games_cfg_entry_point": "stabilization.tasks.manager_based.stabilization.agents:rl_games_ppo_cfg.yaml",
#     },
# )

gym.register(
    id="Isaac-Quad-Stabilization-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "stabilization.tasks.manager_based.stabilization.runners.ManipulatorTrainRunner:ManipulatorEnvCfg",
        "rl_games_cfg_entry_point": "stabilization.tasks.manager_based.stabilization.agents:rl_games_ppo_cfg.yaml",
    },
)