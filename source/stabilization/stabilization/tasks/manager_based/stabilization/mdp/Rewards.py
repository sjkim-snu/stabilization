import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import stabilization.tasks.manager_based.stabilization.mdp as mdp
from stabilization.tasks.manager_based.stabilization.config import load_parameters
import math
from isaaclab.envs.mdp.terminations import time_out as _time_out

# Load configuration from YAML file
CONFIG = load_parameters()

"""
Helper functions
"""

def _push_rew_term(env, name: str, value: torch.Tensor):
    try:
        d = env.extras.get("rew_terms", {})
        d[name] = value
        env.extras["rew_terms"] = d
    except Exception:
        pass

def l2_norm(tensor: torch.Tensor) -> torch.Tensor:
    
    """
    Compute the L2 norm of a tensor along the last dimension.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape
    Returns:
        torch.Tensor: L2 norm of shape 
    """
    
    return torch.sqrt((tensor ** 2).sum(dim=1) + 1e-8)

def sigmoid(norm: torch.Tensor, k: float) -> torch.Tensor:
    """
    Sigmoid function for reward shaping.
    
    Args:
        norm (torch.Tensor): Input tensor.
        k (float): Steepness parameter.
    Returns:
        torch.Tensor: Output tensor after applying the sigmoid function.
    Note: 
        Sigmoid returns the value of (0, 1] for norm in [0, inf).
    """
    
    k_t = torch.tensor(k, device=norm.device, dtype=norm.dtype)
    return 2 / (1 + torch.exp(k_t * norm))

def k_from_half(norm_half: float) -> float:
    
    """
    Compute the steepness parameter k from the half-maximum point norm_half.
    
    Args:
        norm_half (float): The point at which the sigmoid function reaches half its maximum value.
    Returns:
        float: The steepness parameter k.
    Note:
        This is derived from the equation 0.5 = 2 / (1 + exp(k * norm_half)).
    """

    return float(torch.log(torch.tensor(3.0)) / norm_half)

class RewardFns:
    
    @staticmethod
    def pos_err_w_sigmoid(
        env: ManagerBasedEnv,
        norm_half: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")
    ) -> torch.Tensor:
        
        """
        Reward based on the position error in world frame using a sigmoid function.
        
        Args:
            env (ManagerBasedEnv): The environment instance.
            asset_cfg (SceneEntityCfg): Name of the quadrotor entity.
            norm_half (float): Half point for the sigmoid function.
        Returns:
            reward: Tensor of shape (N,) representing the position error reward.
        Note:
            The reward is 1 when the position error is zero,
            and 0.5 when the position error equals norm_half.
        """

        spawn_pos_w = mdp.ObservationFns.get_spawn_pos_w(env, asset_cfg) # (N, 3)
        current_pos_w = mdp.ObservationFns.get_current_pos_w(env, asset_cfg) # (N, 3)
        error_w = spawn_pos_w - current_pos_w # (N, 3)
        error_norm = l2_norm(error_w) # (N,)
        k = k_from_half(norm_half)
        pos_reward = sigmoid(error_norm, k) # (N,)
        _push_rew_term(env, "pos_err", pos_reward)
        return pos_reward
    
    @staticmethod
    def lin_vel_w_sigmoid(
        env: ManagerBasedEnv,
        norm_half: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")
    ) -> torch.Tensor:
        
        """
        Reward based on the linear velocity in body frame using a sigmoid function.
        
        Args:
            env (ManagerBasedEnv): The environment instance.
            asset_cfg (SceneEntityCfg): Name of the quadrotor entity.
            norm_half (float): Half point for the sigmoid function.
        Returns:
            reward: Tensor of shape (N,) representing the linear velocity reward.
        Note:
            The reward is 1 when the linear velocity is zero,
            and 0.5 when the linear velocity equals norm_half.
        """
        
        lin_vel_w = mdp.ObservationFns.get_lin_vel_w(env, asset_cfg)
        lin_vel_norm = l2_norm(lin_vel_w)
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
        
        """
        Reward based on the angular velocity in body frame using a sigmoid function.
        
        Args:
            env (ManagerBasedEnv): The environment instance.
            asset_cfg (SceneEntityCfg): Name of the quadrotor entity.
            norm_half (float): Half point for the sigmoid function.
        Returns:
            reward: Tensor of shape (N,) representing the angular velocity reward.
        Note:
            The reward is 1 when the angular velocity is zero,
            and 0.5 when the angular velocity equals norm_half.
        """
        
        ang_vel_b = mdp.ObservationFns.get_ang_vel_b(env, asset_cfg)
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
        
        """
        Reward based on the orientation (roll, pitch) using a sigmoid function.
        Returns (N,) float tensor.
        """
        
        quat = mdp.ObservationFns.get_quaternion_w(env, asset_cfg)  # (N, 4)
        roll, pitch, _ = math_utils.euler_xyz_from_quat(quat)   # each (N,)
        orientation = torch.stack([roll, pitch], dim=1)         # (N,2)
        orientation_norm = l2_norm(orientation)                 # (N,)
        k = k_from_half(norm_half)
        orientation_reward = sigmoid(orientation_norm, k)   # (N,)
        _push_rew_term(env, "ori_err", orientation_reward)
        return orientation_reward

    @staticmethod
    def time_penalty(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:
        quat = mdp.ObservationFns.get_quaternion_w(env, asset_cfg)  # (N, 4)
        N = quat.shape[0]
        device, dtype = quat.device, quat.dtype
        per_sec = 0.05
        val = torch.full((N,), per_sec, device=device, dtype=dtype)
        _push_rew_term(env, "time_penalty", val)
        return val

    @staticmethod
    def stabilized_bonus(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:
        ok = mdp.TerminationFns.is_stabilized(
            env=env,
            pos_tol_m=float(CONFIG["STABILIZATION"]["POS_ERR_TOL"]),
            lin_vel_tol_mps=float(CONFIG["STABILIZATION"]["LIN_VEL_TOL"]),
            ang_vel_tol_radps=float(CONFIG["STABILIZATION"]["ANG_VEL_TOL"]),
            tilt_tol_rad=math.radians(float(CONFIG["STABILIZATION"]["TILT_DEGREE_TOL"])),
            asset_cfg=asset_cfg,
        ).to(torch.float32)
        _push_rew_term(env, "stabilized", ok)
        return ok

    @staticmethod
    def abnormal_penalty(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:
        # 1) far from spawn
        far = mdp.TerminationFns.is_far_from_spawn(
            env=env,
            dist_threshold_m=float(CONFIG["TERMINATION"]["DIST_THRESHOLD"]),
            asset_cfg=asset_cfg,
        )
        # 2) flipped
        flipped = mdp.TerminationFns.is_flipped(
            env=env,
            tilt_threshold_rad=math.radians(float(CONFIG["TERMINATION"]["FLIP_TILT_DEGREE"])),
            asset_cfg=asset_cfg,
        )
        # 3) crashed
        crashed = mdp.TerminationFns.is_crashed(
            env=env,
            z_min_m=float(CONFIG["TERMINATION"]["ALTITUDE_THRESHOLD"]),
            asset_cfg=asset_cfg,
        )
        # 4) NaN or Inf
        naninf = mdp.TerminationFns.is_nan_or_inf(
            env=env,
            asset_cfg=asset_cfg,
        )
        # 종합 abnormal
        bad = (far | flipped | crashed | naninf).to(torch.float32)
        out = torch.where(bad > 0.5, -torch.ones_like(bad), torch.zeros_like(bad))
        _push_rew_term(env, "abnormal", out)
        return out

@configclass
class RewardCfg:
    
    pos_err_w = RewTerm(
        func=RewardFns.pos_err_w_sigmoid,
        params={
            "asset_cfg": SceneEntityCfg(name="Robot"), 
            "norm_half": CONFIG["REWARD"]["POS_ERR_HALF"]},
        weight=CONFIG["REWARD"]["POS_ERR_WEIGHT"],
    )
    
    lin_vel_w = RewTerm(
        func=RewardFns.lin_vel_w_sigmoid,
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

    time_penalty = RewTerm(
        func=RewardFns.time_penalty,
        params={"asset_cfg": SceneEntityCfg(name="Robot")},
        weight=CONFIG["REWARD"]["TIME_PENALTY_WEIGHT"],
    )

    stabilized_bonus = RewTerm(
        func=RewardFns.stabilized_bonus,
        params={"asset_cfg": SceneEntityCfg(name="Robot")},
        weight=CONFIG["REWARD"]["STABILIZED_BONUS_WEIGHT"],
    )

    abnormal_penalty = RewTerm(
        func=RewardFns.abnormal_penalty,
        params={"asset_cfg": SceneEntityCfg(name="Robot")},
        weight=CONFIG["REWARD"]["ABNORMAL_PENALTY_WEIGHT"],
    )

# Aliases for easier access
pos_err_w_sigmoid = RewardFns.pos_err_w_sigmoid
lin_vel_w_sigmoid = RewardFns.lin_vel_w_sigmoid
ang_vel_b_sigmoid = RewardFns.ang_vel_b_sigmoid
orientation_sigmoid = RewardFns.orientation_sigmoid
time_penalty = RewardFns.time_penalty
stabilized_bonus = RewardFns.stabilized_bonus
abnormal_penalty = RewardFns.abnormal_penalty