import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run quadrotor stabilization (manager-based RL env).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.terminations import time_out as il_time_out
from stabilization.tasks.manager_based.stabilization.config import load_parameters

# Project modules
import stabilization.tasks.manager_based.stabilization.envs as envs
import stabilization.tasks.manager_based.stabilization.mdp as mdp

# Load configuration from YAML file
CONFIG = load_parameters()


"""
Helper functions
"""

def _steps_and_limit(env):
    """Get current steps and step-limit for predicting time_out at this step."""
    if hasattr(env, "episode_length_buf"):
        steps_elapsed = env.episode_length_buf.view(-1)
    elif hasattr(env, "progress_buf"):
        steps_elapsed = env.progress_buf.view(-1)
    else:
        steps_elapsed = env.episode_step_count.view(-1)

    try:
        steps_limit = int(env.max_episode_length)
    except Exception:
        steps_limit = int(float(getattr(env.cfg, "episode_length_s", 0.0)) / env.step_dt + 0.5)
    return steps_elapsed, steps_limit


def _collect_done_reasons_now(env, env_idx: int = 0):
    """Collect possible done reasons from current state."""
    f = mdp.TerminationFns
    reasons = []
    try:
        if f.is_flipped(env)[env_idx]:
            reasons.append("flipped")
    except Exception:
        pass
    try:
        if f.is_far_from_spawn(env)[env_idx]:
            reasons.append("far_from_spawn")
    except Exception:
        pass
    try:
        if f.is_crashed(env)[env_idx]:
            reasons.append("crashed")
    except Exception:
        pass
    try:
        if f.is_nan_or_inf(env)[env_idx]:
            reasons.append("nan_or_inf")
    except Exception:
        pass
    try:
        if f.is_stabilized(env)[env_idx]:
            reasons.append("stabilized")
    except Exception:
        pass
    try:
        if il_time_out(env)[env_idx]:
            reasons.append("time_out")
    except Exception:
        pass
    return reasons


def _collect_done_reasons_with_predict(env, env_idx: int = 0):
    """Pre-capture reasons and also predict step-edge time_out (steps+1)."""
    reasons = _collect_done_reasons_now(env, env_idx)
    if not reasons:
        try:
            steps_elapsed, steps_limit = _steps_and_limit(env)
            # 스텝 호출 직후에 time_out이 뜨는 경계: 지금 스텝+1로 도달하는지 예측
            if int(steps_elapsed[env_idx].item()) + 1 >= steps_limit:
                reasons.append("time_out")
        except Exception:
            pass
    return reasons


@configclass
class ActionsCfg:
    base_controller: mdp.BaseControllerCfg = mdp.BaseControllerCfg(
        class_type=mdp.BaseController,
        asset_name="Robot",
    )

@configclass
class ObservationsCfg(mdp.ObservationsCfg):
    policy: mdp.ObservationsCfg.PolicyCfg = mdp.ObservationsCfg.PolicyCfg()
    def __post_init__(self):
        self.policy.concatenate_terms = CONFIG["OBSERVATION"]["CONCATENATE_TERMS"]
        self.policy.enable_corruption = CONFIG["OBSERVATION"]["ENABLE_CORRUPTION"]

@configclass
class StabilizationEnvCfg(ManagerBasedRLEnvCfg):
    scene: envs.StabilizationSceneCfg = envs.StabilizationSceneCfg(
        num_envs=CONFIG["SCENE"]["NUM_ENVS"],
        env_spacing=CONFIG["SCENE"]["ENV_SPACING"],
        clone_in_fabric=CONFIG["SCENE"]["CLONE_IN_FABRIC"],
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: mdp.EventCfg = mdp.EventCfg()
    rewards: mdp.RewardCfg = mdp.RewardCfg()
    terminations: mdp.TerminationsCfg = mdp.TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = CONFIG["ENV"]["DECIMATION"]
        self.episode_length_s = CONFIG["ENV"]["EPISODE_LENGTH_S"]
        self.sim.dt = CONFIG["ENV"]["PHYSICS_DT"]
        self.viewer.eye = CONFIG["ENV"]["VIEWER"]
        self.sim.render_interval = self.decimation


def main():
    env_cfg = StabilizationEnvCfg()
    env_cfg.sim.device = CONFIG["LAUNCHER"]["DEVICE"]
    env_cfg.sim.headless = CONFIG["LAUNCHER"]["HEADLESS"]
    env = ManagerBasedRLEnv(cfg=env_cfg)
    use_manual = CONFIG["LAUNCHER"]["USE_MANUAL_ACTION"]

    if use_manual:
        manual_action_tensor = torch.tensor(
            CONFIG["LAUNCHER"]["MANUAL_ACTION"],
            device=CONFIG["LAUNCHER"]["DEVICE"],
            dtype=env.action_manager.action.dtype,
        ).clamp(-1.0, 1.0)

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if step % 600 == 0:
                env.reset()

            # pre-capture reasons with step-edge time_out prediction
            pre_reasons_env0 = _collect_done_reasons_with_predict(env, env_idx=0)

            if use_manual:
                actions = manual_action_tensor.unsqueeze(0).expand_as(env.action_manager.action)
            else:
                actions = torch.zeros_like(env.action_manager.action)

            obs, rew, terminated, truncated, info = env.step(actions)

            if step % 60 == 0:
                robot = env.scene["Robot"]
                root_state = robot.data.root_state_w
                p0 = root_state[0, :3]
                quat_w = robot.data.root_quat_w
                r, p, y = mdp.ObservationFns.quaternion_to_orientation(quat_w)
                rpy0 = (r[0].item(), p[0].item(), y[0].item())
                mode = "MANUAL" if use_manual else "ZERO"

                done_str = ""
                term0 = bool(terminated[0].item())
                trunc0 = bool(truncated[0].item())
                if term0 or trunc0:
                    label = "TERM" if term0 else "TRUNC"
                    reason_txt = ",".join(pre_reasons_env0) if pre_reasons_env0 else "unknown"
                    done_str = f"  done={label}({reason_txt})"

                print(
                    f"[{mode}] [env0] pos=({p0[0]:+.3f}, {p0[1]:+.3f}, {p0[2]:+.3f})  "
                    f"rpy=({rpy0[0]:+.2f}, {rpy0[1]:+.2f}, {rpy0[2]:+.2f})  "
                    f"rew={rew[0].item():+.4f}{done_str}"
                )

            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
