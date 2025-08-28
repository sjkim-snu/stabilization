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
                print(f"[INFO] Environment reset at step {step}")

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

                term0 = bool(terminated[0].item())
                trunc0 = bool(truncated[0].item())

                tm = env.termination_manager
                names = list(tm.active_terms)
                fired = [n for n in names if bool(tm.get_term(n)[0].item())]
                timeout0 = bool(tm.time_outs[0].item()) if hasattr(tm, "time_outs") else False
                reasons = fired + (["time_out"] if timeout0 else [])
                reason_txt = ",".join(reasons) if reasons else "unknown"

                done_str = ""
                if term0 or trunc0:
                    label = "TERM" if term0 else "TRUNC"
                    done_str = f"  done={label}({reason_txt})"

                print(
                    f"[{mode}] [env0] pos=({p0[0]:+.3f}, {p0[1]:+.3f}, {p0[2]:+.3f})  "
                    f"rpy=({rpy0[0]:+.2f}, {rpy0[1]:+.2f}, {rpy0[2]:+.2f})  "
                    f"rew={rew[0].item():+.4f}{done_str}"
                )

                if term0 or trunc0:
                    print(f"[REASON] fired={','.join(fired) if fired else 'none'} timeout={timeout0}")

            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
