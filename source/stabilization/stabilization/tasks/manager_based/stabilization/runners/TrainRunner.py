import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from stabilization.tasks.manager_based.stabilization.config import load_parameters
import stabilization.tasks.manager_based.stabilization.mdp as mdp
import stabilization.tasks.manager_based.stabilization.envs as envs
from stabilization.tasks.manager_based.stabilization.runners import Logger
from isaaclab.envs import ManagerBasedRLEnv  

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

def _enable_csv_logging_for_rl_games():  
    try:  
        _orig_init = ManagerBasedRLEnv.__init__  
        _orig_step = ManagerBasedRLEnv.step  
        _orig_close = ManagerBasedRLEnv.close  
    except Exception:  
        return  

    def _new_init(self, *args, **kwargs):  
        _orig_init(self, *args, **kwargs)  
        try:  
            self._csv_logger = Logger.EpisodeCSVLogger(  
                num_envs=self.num_envs,  
                cfg=Logger.CSVLoggerCfg(  
                    device=CONFIG["LAUNCHER"]["DEVICE"],  
                    policy_dt_s=float(CONFIG["ENV"]["PHYSICS_DT"]) * float(CONFIG["ENV"]["DECIMATION"]),  
                    flush_every_rows=1,  
                ),  
            )  
        except Exception:  
            self._csv_logger = None  

    def _new_step(self, actions):  
        out = _orig_step(self, actions)  
        try:  
            logger = getattr(self, "_csv_logger", None)  
            if logger is not None:  
                obs, rew, terminated, truncated, info = out  
                dones = (terminated | truncated)  
                logger.log_step(  
                    rewards=rew,  
                    dones=dones,  
                    rew_terms_step=self.extras.get("rew_terms", None),  
                    term_mgr=getattr(self, "termination_manager", None),  
                    actions=actions,  
                )  
        except Exception:  
            pass  
        return out  

    def _new_close(self):  
        try:  
            logger = getattr(self, "_csv_logger", None)  
            if logger is not None:  
                logger.close()  
        except Exception:  
            pass  
        return _orig_close(self)  

    ManagerBasedRLEnv.__init__ = _new_init  
    ManagerBasedRLEnv.step = _new_step  
    ManagerBasedRLEnv.close = _new_close  

_enable_csv_logging_for_rl_games()  
