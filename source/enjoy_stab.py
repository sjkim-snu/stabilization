
import argparse, sys, runpy
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--headless", action="store_true")
args_cli, unknown = parser.parse_known_args()

app = AppLauncher(headless=args_cli.headless).app

import gymnasium as gym

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

import stabilization.tasks.manager_based.stabilization.runners.TrainRunner  # noqa: F401

from isaaclab_rl.rl_games import RlGamesGpuEnv
from rl_games.common import vecenv
vecenv.register("rlgpu", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))

from rl_games.torch_runner import Runner
import yaml

r = Runner()
cfg_path = "/home/ksj/stabilization/source/stabilization/stabilization/tasks/manager_based/stabilization/agents/rl_games_ppo_cfg.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["params"]["config"]["env_name"] = "rlgpu"
cfg["params"]["config"]["num_actors"] = int(args_cli.num_envs)
cfg["params"]["config"].setdefault("env_config", {})
cfg["params"]["config"]["env_config"].update({
    "task_name": ENV_ID,         
    "task": ENV_ID,              
    "headless": args_cli.headless
})
cfg['params'].setdefault('player', {})                   
cfg['params']['player'].update({'deterministic': True})  

r.load(cfg)
r.run({"train": False, "play": True,
       "checkpoint": args_cli.checkpoint, "num_envs": args_cli.num_envs})
app.close()
