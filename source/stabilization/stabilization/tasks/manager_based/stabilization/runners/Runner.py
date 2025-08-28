"""
Run quadrotor stabilization task with ManagerBasedRLEnv.

Usage (example):
  ./isaaclab.sh -p stabilization/tasks/manager_based/stabilization/runners/Runner.py \
      --num_envs 16 --renderer RayTracedLighting
"""

# -----------------
# Manual action (always applied)
# -----------------
MANUAL_ACTION = [-0.939, -0.939, -0.939, -0.939]  # each in [-1, 1]

import argparse
from isaaclab.app import AppLauncher

# -----------------
# CLI
# -----------------
parser = argparse.ArgumentParser(description="Run quadrotor stabilization (manager-based RL env).")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------
# Imports AFTER app launch
# -----------------
import torch
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from stabilization.tasks.manager_based.stabilization.config import load_parameters

# Project modules
import stabilization.tasks.manager_based.stabilization.envs as envs      # Scenes.py (StabilizationSceneCfg)
import stabilization.tasks.manager_based.stabilization.mdp as mdp       # Actions/Observations/Rewards/Events/Terminations

# -----------------
# Bridge Configs
# -----------------
@configclass
class ActionsCfg:
    """
    Action manager wrapper.
    - BaseController(ActionTerm) from mdp/Actions.py
    """
    # NOTE: class_type 지정이 필수입니다.
    base_controller: mdp.BaseControllerCfg = mdp.BaseControllerCfg(
        class_type=mdp.BaseController,
        asset_name="Robot",
    )


@configclass
class ObservationsCfg(mdp.ObservationsCfg):
    """
    Observation manager wrapper.
    - mdp.ObservationsCfg.PolicyCfg(ObsGroup) from mdp/Observations.py
    """
    policy: mdp.ObservationsCfg.PolicyCfg = mdp.ObservationsCfg.PolicyCfg()

    def __post_init__(self):
        # 하나의 벡터로 결합하고, 부가적 센서 노이즈/왜곡은 비활성화(필요 시 프로젝트에서 조정)
        self.policy.concatenate_terms = True
        self.policy.enable_corruption = False


@configclass
class StabilizationEnvCfg(ManagerBasedRLEnvCfg):
    """
    Full RL env config tying together: scene + observations + actions + events + rewards + terminations
    """
    # scene
    scene: envs.StabilizationSceneCfg = envs.StabilizationSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=0.5,
        clone_in_fabric=True,
    )

    # managers
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: mdp.EventCfg = mdp.EventCfg()
    rewards: mdp.RewardCfg = mdp.RewardCfg()
    terminations: mdp.TerminationsCfg = mdp.TerminationsCfg()

    # common sim/view settings
    def __post_init__(self) -> None:
        # Physics / control rate
        self.decimation = 2                 # actions applied every 2 physics steps
        self.episode_length_s = 20.0        # hard cap; 내부 termination에도 의존
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation

        # Viewer
        self.viewer.eye = (3.0, 3.0, 2.0)


# -----------------
# Main loop
# -----------------
def main():
    # Build env
    env_cfg = StabilizationEnvCfg()
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Manual action tensor (broadcast to all envs each step)
    manual_action_tensor = torch.tensor(
        MANUAL_ACTION, device=env.action_manager.action.device, dtype=env.action_manager.action.dtype
    ).clamp(-1.0, 1.0)  # 안전 범위 고정

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # 주기적 전체 리셋(개별 env는 termination/truncation으로도 리셋됨)
            if step % 600 == 0:
                env.reset()

            # 항상 수동 액션 적용: (4,) -> (num_envs, 4)
            actions = manual_action_tensor.unsqueeze(0).expand_as(env.action_manager.action)
            obs, rew, terminated, truncated, info = env.step(actions)

            # -----------------
            # Monitoring (env0)
            # -----------------
            if step % 60 == 0:
                robot = env.scene["Robot"]
                root_state = robot.data.root_state_w  # (N, 13): pos(3), quat(4), linvel(3), angvel(3)
                p0 = root_state[0, :3]

                # 자세: 공용 관측 헬퍼 사용
                quat_w = robot.data.root_quat_w      # (N, 4) world-frame quaternion (xyzw)
                r, p, y = mdp.ObservationFns.quaternion_to_orientation(quat_w)
                rpy0 = (r[0].item(), p[0].item(), y[0].item())

                print(
                    f"[env0] pos=({p0[0]:+.3f}, {p0[1]:+.3f}, {p0[2]:+.3f})  "
                    f"rpy=({rpy0[0]:+.2f}, {rpy0[1]:+.2f}, {rpy0[2]:+.2f})  "
                    f"rew={rew[0].item():+.4f}"
                )

            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
