# /home/ksj/stabilization/source/run_stab.py
import sys, runpy
import gymnasium as gym


sys.path.insert(0, "/home/ksj/stabilization/source")

ENV_ID = "Isaac-Quad-Stabilization-v0"

try:
    gym.spec(ENV_ID)
except Exception:
    gym.register(
        id=ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "stabilization.tasks.manager_based.stabilization.runners.TrainRunner:StabilizationEnvCfg",
            "rl_games_cfg_entry_point": "stabilization.tasks.manager_based.stabilization.agents:rl_games_ppo_cfg.yaml",
        },
    )


TRAIN = "/home/ksj/IsaacLab/scripts/reinforcement_learning/rl_games/train.py"
sys.argv = [TRAIN] + sys.argv[1:]
runpy.run_path(TRAIN, run_name="__main__")
