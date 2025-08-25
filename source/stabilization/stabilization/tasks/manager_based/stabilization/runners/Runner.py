"""
Run quadrotor stabilization task with ManagerBasedRLEnv.

Usage (example):
  ./isaaclab.sh -p stabilization/tasks/manager_based/stabilization/runners/Runner.py \
      --num_envs 16 --renderer RayTracedLighting
"""

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
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg

# Project modules
import stabilization.tasks.manager_based.stabilization.envs as envs      # Scenes.py (StabilizationSceneCfg)
import stabilization.tasks.manager_based.stabilization.mdp as mdp        # Actions/Observations/Rewards/Events/Terminations


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
    # device는 AppLauncher 인자에 따라 설정됨
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 엔티티 핸들 (로그 출력용)
    robot_entity = SceneEntityCfg(name="Robot")

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # 주기적 전체 리셋(개별 env는 termination/truncation으로도 리셋됨)
            if step % 600 == 0:
                env.reset()

            # 아직 학습 전이므로 0-action으로 step (정책 연결 시 교체)
            actions = torch.zeros_like(env.action_manager.action)
            obs, rew, terminated, truncated, info = env.step(actions)

            # -----------------
            # Monitoring (env0)
            # -----------------
            if step % 60 == 0:
                # 위치
                robot = env.scene["Robot"]
                root_state = robot.data.root_state_w  # (N, 13): pos(3), quat(4), linvel(3), angvel(3)
                p0 = root_state[0, :3]

                # 자세: 프로젝트의 공용 관측 헬퍼를 사용(중복/오류 방지)
                quat_w = robot.data.root_quat_w      # (N, 4) world-frame quaternion (xyzw)
                # ObservationFns.quaternion_to_orientation -> (roll, pitch, yaw)
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
