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

    asset = SceneEntityCfg(name="Robot")
    step = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            if step == 0:
                env.reset()

            if use_manual:
                actions = manual_action_tensor.unsqueeze(0).expand_as(env.action_manager.action)
            else:
                actions = torch.zeros_like(env.action_manager.action)

            obs, rew, terminated, truncated, info = env.step(actions)

            # Logging
            if step % 60 == 0:
                pos_err = mdp.ObservationFns.position_error_w(env, asset)[0]        # (3,)
                lin_b   = mdp.ObservationFns.lin_vel_body(env, asset)[0]            # (3,)
                ang_b   = mdp.ObservationFns.ang_vel_body(env, asset)[0]            # (3,)
                roll    = mdp.ObservationFns.roll_current(env, asset)[0].item()     # scalar
                pitch   = mdp.ObservationFns.pitch_current(env, asset)[0].item()    # scalar
                yaw     = mdp.ObservationFns.yaw_current(env, asset)[0].item()      # scalar

                print(
                    f"[env0] pos_err_w=({pos_err[0]:+.3f},{pos_err[1]:+.3f},{pos_err[2]:+.3f})  "
                    f"lin_vel_b=({lin_b[0]:+.3f},{lin_b[1]:+.3f},{lin_b[2]:+.3f})  "
                    f"ang_vel_b=({ang_b[0]:+.3f},{ang_b[1]:+.3f},{ang_b[2]:+.3f})  "
                    f"orientation=({roll:+.2f},{pitch:+.2f},{yaw:+.2f})  "
                    f"reward={rew[0].item():+.4f}"
                )

            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
