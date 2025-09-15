import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from stabilization.tasks.manager_based.stabilization.config import load_parameters
import stabilization.tasks.manager_based.stabilization.mdp as mdp
import stabilization.tasks.manager_based.stabilization.envs as envs
from stabilization.tasks.manager_based.stabilization.runners import Logger
from isaaclab.envs import ManagerBasedRLEnv  # 추가 (+)

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

def _enable_csv_logging_for_rl_games():  # 추가 (+)
    try:  # 추가 (+)
        _orig_init = ManagerBasedRLEnv.__init__  # 추가 (+)
        _orig_step = ManagerBasedRLEnv.step  # 추가 (+)
        _orig_close = ManagerBasedRLEnv.close  # 추가 (+)
    except Exception:  # 추가 (+)
        return  # 추가 (+)

    def _new_init(self, *args, **kwargs):  # 추가 (+)
        _orig_init(self, *args, **kwargs)  # 추가 (+)
        try:  # 추가 (+)
            self._csv_logger = Logger.EpisodeCSVLogger(  # 추가 (+)
                num_envs=self.num_envs,  # 추가 (+)
                cfg=Logger.CSVLoggerCfg(  # 추가 (+)
                    device=CONFIG["LAUNCHER"]["DEVICE"],  # 추가 (+)
                    policy_dt_s=float(CONFIG["ENV"]["PHYSICS_DT"]) * float(CONFIG["ENV"]["DECIMATION"]),  # 추가 (+)
                    flush_every_rows=1,  # 추가 (+)
                ),  # 추가 (+)
            )  # 추가 (+)
        except Exception:  # 추가 (+)
            self._csv_logger = None  # 추가 (+)

    def _new_step(self, actions):  # 추가 (+)
        out = _orig_step(self, actions)  # 추가 (+)
        try:  # 추가 (+)
            logger = getattr(self, "_csv_logger", None)  # 추가 (+)
            if logger is not None:  # 추가 (+)
                obs, rew, terminated, truncated, info = out  # 추가 (+)
                dones = (terminated | truncated)  # 추가 (+)
                logger.log_step(  # 추가 (+)
                    rewards=rew,  # 추가 (+)
                    dones=dones,  # 추가 (+)
                    rew_terms_step=self.extras.get("rew_terms", None),  # 추가 (+)
                    term_mgr=getattr(self, "termination_manager", None),  # 추가 (+)
                    actions=actions,  # 추가 (+)
                )  # 추가 (+)
        except Exception:  # 추가 (+)
            pass  # 추가 (+)
        return out  # 추가 (+)

    def _new_close(self):  # 추가 (+)
        try:  # 추가 (+)
            logger = getattr(self, "_csv_logger", None)  # 추가 (+)
            if logger is not None:  # 추가 (+)
                logger.close()  # 추가 (+)
        except Exception:  # 추가 (+)
            pass  # 추가 (+)
        return _orig_close(self)  # 추가 (+)

    ManagerBasedRLEnv.__init__ = _new_init  # 추가 (+)
    ManagerBasedRLEnv.step = _new_step  # 추가 (+)
    ManagerBasedRLEnv.close = _new_close  # 추가 (+)

_enable_csv_logging_for_rl_games()  # 추가 (+)
