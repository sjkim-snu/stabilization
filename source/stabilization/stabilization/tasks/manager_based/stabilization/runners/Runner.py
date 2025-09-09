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

import stabilization.tasks.manager_based.stabilization.envs as envs
import stabilization.tasks.manager_based.stabilization.mdp as mdp
import Logger
import math

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
    
    # define environment
    env_cfg = StabilizationEnvCfg()
    env_cfg.sim.device = CONFIG["LAUNCHER"]["DEVICE"]
    env_cfg.sim.headless = CONFIG["LAUNCHER"]["HEADLESS"]
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # activate logger
    csv_logger = Logger.EpisodeCSVLogger(
    num_envs=env.num_envs,
    cfg=Logger.CSVLoggerCfg(
        device=CONFIG["LAUNCHER"]["DEVICE"],
        policy_dt_s=CONFIG["ENV"]["PHYSICS_DT"] * CONFIG["ENV"]["DECIMATION"],
        ),
    )

    # decide whether to use manual action input
    use_manual = CONFIG["LAUNCHER"]["USE_MANUAL_ACTION"]
    
    if use_manual:
        manual_action_tensor = torch.tensor(
            CONFIG["LAUNCHER"]["MANUAL_ACTION"],
            device=CONFIG["LAUNCHER"]["DEVICE"],
            dtype=env.action_manager.action.dtype,
        ).clamp(-1.0, 1.0)

    # main loop
    asset = SceneEntityCfg(name="Robot")
    step = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            
            # initialize simulation
            if step == 0:
                env.reset()

            # if manual action is enabled, use it; otherwise, use zero action (TODO)
            if use_manual:
                actions = manual_action_tensor.unsqueeze(0).expand_as(env.action_manager.action)
            else:
                actions = torch.zeros_like(env.action_manager.action)

            # step simulation
            obs, rew, terminated, truncated, info = env.step(actions)

            # define done signal and log as csv files
            dones = (terminated | truncated)
            rew_terms_step = info.get("rew_terms", None) if isinstance(info, dict) else None
            
            # for logging, convert actions to tensor on the correct device
            csv_logger.log_step(
                rewards=rew,
                dones=dones,
                rew_terms_step=info.get("rew_terms", None),
                term_mgr=getattr(env, "termination_manager", None),
                actions=actions, 
            )

            if step % 25 == 0: # 5Hz
                # 관측값
                lin_w = mdp.ObservationFns.get_lin_vel_w(env, asset)[0]
                ang_b = mdp.ObservationFns.get_ang_vel_b(env, asset)[0]

                # 액션/추력 디버깅
                bc = env.action_manager.get_term("base_controller")  # name은 ActionsCfg의 필드명과 동일
                # raw(-1~1), processed(ω: rad/s)
                raw_act = bc.raw_actions[0]                 # (4,)
                omega   = bc.processed_actions[0]           # (4,) rad/s

                # 총추력 합 Fz = Σ(k_f * ω^2)  [N]
                # (BaseController에서 k_f는 rad/s^2 계수로 변환되어 있음)
                kf = bc._k_f
                Fz_total = float((omega**2 * kf).sum().item())

                # 보조 출력: 각 모터 rpm
                rpm = (omega * (60.0 / (2.0 * math.pi))).tolist()

                print(
                    f"lin_vel_w=({lin_w[0]:+.3f},{lin_w[1]:+.3f},{lin_w[2]:+.3f})  "
                    f"ang_vel_b=({ang_b[0]:+.3f},{ang_b[1]:+.3f},{ang_b[2]:+.3f})  "
                    f"raw_act={tuple(float(x) for x in raw_act.tolist())}  "
                    f"omega(rad/s)={tuple(float(x) for x in omega.tolist())}  "
                    f"rpm={tuple(float(x) for x in rpm)}  "
                    f"Fz_sum_N={Fz_total:+.3f}  "
                    f"reward={rew[0].item():+.4f}"
                )

            step += 1

    csv_logger.close()
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
