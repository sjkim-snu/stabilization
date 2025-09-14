import gymnasium as gym

gym.register(
    id="Isaac-Quad-Stabilization-v0", 
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            "stabilization.tasks.manager_based.stabilization.runners.Runner:StabilizationEnvCfg",  # ← 존재 확인됨
        "rl_games_cfg_entry_point":
            "stabilization.tasks.manager_based.stabilization.agents:rl_games_ppo_cfg.yaml",        # ← 실제 파일명/모듈 경로에 맞춤
    },
)
