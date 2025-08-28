import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import stabilization.tasks.manager_based.stabilization.mdp as mdp
from stabilization.tasks.manager_based.stabilization.config import load_parameters

CONFIG = load_parameters()

def _push_rew_term(env, name: str, value: torch.Tensor):
    try:
        d = env.extras.get("rew_terms", {})
        d[name] = value
        env.extras["rew_terms"] = d
    except Exception:
        pass

def l2_norm(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((tensor ** 2).sum(dim=1) + 1e-8)

def sigmoid(norm: torch.Tensor, k: float) -> torch.Tensor:
    k_t = torch.tensor(k, device=norm.device, dtype=norm.dtype)
    return 2 / (1 + torch.exp(k_t * norm))

def k_from_half(norm_half: float) -> float:
    return float(torch.log(torch.tensor(3.0)) / norm_half)

class RewardFns:
    @staticmethod
    def pos_err_w_sigmoid(
        env: ManagerBasedEnv,
        norm_half: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")
    ) -> torch.Tensor:
        error_w = mdp.ObservationFns.position_error_w(env, asset_cfg)
        error_norm = l2_norm(error_w)
        k = k_from_half(norm_half)
        pos_reward = sigmoid(error_norm, k)
        _push_rew_term(env, "pos_err", pos_reward)
        return pos_reward

    @staticmethod
    def lin_vel_b_sigmoid(
        env: ManagerBasedEnv,
        norm_half: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")
    ) -> torch.Tensor:
        lin_vel_b = mdp.ObservationFns.lin_vel_body(env, asset_cfg)
        lin_vel_norm = l2_norm(lin_vel_b)
        k = k_from_half(norm_half)
        vel_reward = sigmoid(lin_vel_norm, k)
        _push_rew_term(env, "lin_vel", vel_reward)
        return vel_reward

    @staticmethod
    def ang_vel_b_sigmoid(
        env: ManagerBasedEnv, 
        norm_half: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")
    ) -> torch.Tensor:
        ang_vel_b = mdp.ObservationFns.ang_vel_body(env, asset_cfg)
        ang_vel_norm = l2_norm(ang_vel_b)
        k = k_from_half(norm_half)
        ang_vel_reward = sigmoid(ang_vel_norm, k)
        _push_rew_term(env, "ang_vel", ang_vel_reward)
        return ang_vel_reward

    @staticmethod
    def orientation_sigmoid(
        env: ManagerBasedEnv, 
        norm_half: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")
    ) -> torch.Tensor:
        roll  = mdp.ObservationFns.roll_current(env, asset_cfg).reshape(-1, 1)
        pitch = mdp.ObservationFns.pitch_current(env, asset_cfg).reshape(-1, 1)
        yaw   = mdp.ObservationFns.yaw_current(env, asset_cfg).reshape(-1, 1)
        orientation = torch.cat([roll, pitch, yaw], dim=1)
        orientation_norm = l2_norm(orientation)
        k = k_from_half(norm_half)
        orientation_reward = sigmoid(orientation_norm, k)
        _push_rew_term(env, "ori_err", orientation_reward)
        return orientation_reward

@configclass
class RewardCfg:
    pos_err_w = RewTerm(
        func=RewardFns.pos_err_w_sigmoid,
        params={
            "asset_cfg": SceneEntityCfg(name="Robot"), 
            "norm_half": CONFIG["REWARD"]["POS_ERR_HALF"]},
        weight=CONFIG["REWARD"]["POS_ERR_WEIGHT"],
    )
    lin_vel_b = RewTerm(
        func=RewardFns.lin_vel_b_sigmoid,
        params={
            "asset_cfg": SceneEntityCfg(name="Robot"), 
            "norm_half": CONFIG["REWARD"]["LIN_VEL_HALF"]},
        weight=CONFIG["REWARD"]["LIN_VEL_WEIGHT"],
    )
    ang_vel_b = RewTerm(
        func=RewardFns.ang_vel_b_sigmoid,
        params={
            "asset_cfg": SceneEntityCfg(name="Robot"), 
            "norm_half": CONFIG["REWARD"]["ANG_VEL_HALF"]},
        weight=CONFIG["REWARD"]["ANG_VEL_WEIGHT"],
    )
    orientation = RewTerm(
        func=RewardFns.orientation_sigmoid,
        params={
            "asset_cfg": SceneEntityCfg(name="Robot"), 
            "norm_half": CONFIG["REWARD"]["ORI_ERR_HALF"]},
        weight=CONFIG["REWARD"]["ORI_ERR_WEIGHT"],
    )
