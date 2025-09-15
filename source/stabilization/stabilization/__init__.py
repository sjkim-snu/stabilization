# /home/ksj/stabilization/source/stabilization/stabilization/__init__.py
import gymnasium as gym

gym.register(
    id="Isaac-Quad-Stabilization-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # 매니저 기반 RL 환경
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "stabilization.tasks.manager_based.stabilization."
            "runners.Runner:StabilizationEnvCfg"   
        ),
        "rl_games_cfg_entry_point": (
            "stabilization.tasks.manager_based.stabilization.agents:"
            "rl_games_ppo_cfg.yaml"
        ),
    },
)
