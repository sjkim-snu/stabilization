# /home/ksj/stabilization/source/run_stab.py
import sys, runpy
import gymnasium as gym

# 0) 소스 경로를 확실히 추가 (패키지 미설치여도 import 가능)
sys.path.insert(0, "/home/ksj/stabilization/source")

# 1) 등록 보장: 이미 있으면 통과, 없으면 지금 등록
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

# 2) (선택) 패키지 임포트 — 여기서는 부가 효과 없음. 등록만 보장되면 불필요.
# import stabilization

# 3) 학습 스크립트에 사용자가 넘긴 CLI 인자를 그대로 전달
TRAIN = "/home/ksj/IsaacLab/scripts/reinforcement_learning/rl_games/train.py"
sys.argv = [TRAIN] + sys.argv[1:]
runpy.run_path(TRAIN, run_name="__main__")
