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
    
    """Compute the L2 norm of a tensor along the last dimension."""
    
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

def linear_interpolation(x: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    lower_t = torch.as_tensor(lower, device=x.device, dtype=x.dtype)  # 추가 (+)
    upper_t = torch.as_tensor(upper, device=x.device, dtype=x.dtype)  # 추가 (+)
    t = ((upper_t - x) / (upper_t - lower_t + 1e-8)).clamp(0.0, 1.0)  # 추가 (+)
    return t 

def gate_near(dist: torch.Tensor, d_near: float, d_far: float) -> torch.Tensor:  # 추가 (+)
    return ((d_far - dist) / (d_far - d_near + 1e-8)).clamp(0.0, 1.0)  # 추가 (+)

def get_pos_err_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    spawn_pos_w = mdp.ObservationFns.get_spawn_pos_w(env, asset_cfg) 
    current_pos_w = mdp.ObservationFns.get_current_pos_w(env, asset_cfg)  
    pos_err_w = spawn_pos_w - current_pos_w  
    return pos_err_w

def get_tilt_angle(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    quat = mdp.ObservationFns.get_quaternion_w(env, asset_cfg)  # 추가 (+)
    roll, pitch, _ = math_utils.euler_xyz_from_quat(quat)  # 추가 (+)
    return roll, pitch

def get_allowed_tilt_rad(  
    dist: torch.Tensor, deg_near: float, deg_far: float, d_near: float, d_far: float
) -> torch.Tensor:
    t = gate_near(dist, d_near, d_far)  # 추가 (+)
    deg = deg_near * t + deg_far * (1.0 - t)  # 추가 (+)
    return torch.deg2rad(torch.as_tensor(deg, device=dist.device, dtype=dist.dtype))  # 추가 (+)


class RewardFns:
    
    @staticmethod
    def pos_err_w_sigmoid(env: ManagerBasedEnv, norm_half: float, asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:  # 추가 (+)
        pos_err_w = get_pos_err_w(env, asset_cfg)  # 추가 (+)
        error_norm = l2_norm(pos_err_w)  # 추가 (+)
        k = k_from_half(norm_half)  # 추가 (+)
        pos_reward = sigmoid(error_norm, k)  # 추가 (+)

        dist = error_norm  # 추가 (+)
        d_near = float(CONFIG["STABILIZATION"]["POS_ERR_TOL"])  # 추가 (+)
        d_far  = float(CONFIG["REWARD"]["POS_ERR_HALF"])  # 추가 (+)
        deg_near = float(CONFIG["STABILIZATION"]["TILT_DEGREE_TOL"])  # 추가 (+)
        deg_far  = float(CONFIG["CASCADE_CONTROLLER"]["TILT_MAX"][0])  # 추가 (+)

        roll, pitch = get_tilt_angle(env, asset_cfg)  # 추가 (+)
        tilt = torch.sqrt(roll * roll + pitch * pitch + 1e-8)  # 추가 (+)
        tilt_allow = get_allowed_tilt_rad(dist, deg_near, deg_far, d_near, d_far)  # 추가 (+)
        tilt_ex = (tilt - tilt_allow).clamp(min=0.0)  # 추가 (+)
        g_theta = linear_interpolation(tilt_ex, 0.0, math.radians(deg_near))  # 추가 (+)

        ang_b = mdp.ObservationFns.get_ang_vel_b(env, asset_cfg)  # 추가 (+)
        ang_norm = l2_norm(ang_b)  # 추가 (+)
        rate_hard = float(CONFIG["STABILIZATION"]["ANG_VEL_TOL"])  # 추가 (+)
        rate_soft = float(CONFIG["REWARD"]["ANG_VEL_HALF"])  # 추가 (+)
        g_rate = linear_interpolation(ang_norm, rate_hard, rate_soft)  # 추가 (+)

        pos_reward = pos_reward * g_theta * g_rate  # 추가 (+)
        _push_rew_term(env, "pos_err", pos_reward)  # 추가 (+)
        return pos_reward  # 추가 (+)

    
    @staticmethod
    def lin_vel_w_sigmoid(env: ManagerBasedEnv, norm_half: float, asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:  # 추가 (+)
        to_vec = get_pos_err_w(env, asset_cfg)  # 추가 (+)
        dist = l2_norm(to_vec)  # 추가 (+)
        e_to = to_vec / (dist.unsqueeze(-1) + 1e-8)  # 추가 (+)

        v_r = (vel_w * e_to).sum(dim=-1)  # 추가 (+)
        v_t = vel_w - v_r.unsqueeze(-1) * e_to  # 추가 (+)
        v_t_norm = l2_norm(v_t)  # 추가 (+)

        d_near = float(CONFIG["STABILIZATION"]["POS_ERR_TOL"])  # 추가 (+)
        d_far  = float(CONFIG["REWARD"]["POS_ERR_HALF"])  # 추가 (+)
        gnear  = gate_near(dist, d_near, d_far)  # 추가 (+)

        kv   = float(CONFIG["REWARD"]["LIN_VEL_HALF"]) / float(CONFIG["REWARD"]["POS_ERR_HALF"])  # 추가 (+)
        vmin = float(CONFIG["STABILIZATION"]["LIN_VEL_TOL"])  # 추가 (+)
        vlim_xy = 0.5 * (float(CONFIG["CASCADE_CONTROLLER"]["VEL_LIMIT"][0]) + float(CONFIG["CASCADE_CONTROLLER"]["VEL_LIMIT"][1]))  # 추가 (+)
        vmax = float(vlim_xy)  # 추가 (+)
        v_star = torch.clip(kv * dist, min=vmin, max=vmax)  # 추가 (+)

        sig_near = float(CONFIG["STABILIZATION"]["LIN_VEL_TOL"])  # 추가 (+)
        sig_far  = float(CONFIG["REWARD"]["LIN_VEL_HALF"])  # 추가 (+)
        sigma = sig_near * gnear + sig_far * (1.0 - gnear)  # 추가 (+)

        track = torch.exp(-0.5 * ((v_r - v_star) / (sigma + 1e-8)) ** 2)  # 추가 (+)

        w_tan = gnear / (sig_near + 1e-8) + (1.0 - gnear) / (sig_far + 1e-8)  # 추가 (+)
        side_pen = -w_tan * v_t_norm  # 추가 (+)

        roll, pitch = get_tilt_angle(env, asset_cfg)  # 추가 (+)
        tilt = torch.sqrt(roll * roll + pitch * pitch + 1e-8)  # 추가 (+)
        deg_near = float(CONFIG["STABILIZATION"]["TILT_DEGREE_TOL"])  # 추가 (+)
        deg_far  = float(CONFIG["CASCADE_CONTROLLER"]["TILT_MAX"][0])  # 추가 (+)
        tilt_allow = get_allowed_tilt_rad(dist, deg_near, deg_far, d_near, d_far)  # 추가 (+)
        tilt_ex = (tilt - tilt_allow).clamp(min=0.0)  # 추가 (+)
        g_theta = linear_interpolation(tilt_ex, 0.0, math.radians(deg_near))  # 추가 (+)

        ang_b = mdp.ObservationFns.get_ang_vel_b(env, asset_cfg)  # 추가 (+)
        ang_norm = l2_norm(ang_b)  # 추가 (+)
        rate_hard = float(CONFIG["STABILIZATION"]["ANG_VEL_TOL"])  # 추가 (+)
        rate_soft = float(CONFIG["REWARD"]["ANG_VEL_HALF"])  # 추가 (+)
        g_rate = linear_interpolation(ang_norm, rate_hard, rate_soft)  # 추가 (+)

        vel_reward = g_theta * g_rate * (track + side_pen)  # 추가 (+)
        _push_rew_term(env, "lin_vel", vel_reward)  # 추가 (+)
        return vel_reward  # 추가 (+)

    
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
    def orientation_sigmoid(env: ManagerBasedEnv, norm_half: float, asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:  # 추가 (+)
        roll, pitch = get_tilt_angle(env, asset_cfg)  # 추가 (+)
        tilt = torch.sqrt(roll * roll + pitch * pitch + 1e-8)  # 추가 (+)
        k = k_from_half(norm_half)  # 추가 (+)
        base = sigmoid(tilt, k)  # 추가 (+)

        pos_err_w = get_pos_err_w(env, asset_cfg)  # 추가 (+)
        dist = l2_norm(pos_err_w)  # 추가 (+)
        d_near = float(CONFIG["STABILIZATION"]["POS_ERR_TOL"])  # 추가 (+)
        d_far  = float(CONFIG["REWARD"]["POS_ERR_HALF"])  # 추가 (+)
        deg_near = float(CONFIG["STABILIZATION"]["TILT_DEGREE_TOL"])  # 추가 (+)
        deg_far  = float(CONFIG["CASCADE_CONTROLLER"]["TILT_MAX"][0])  # 추가 (+)
        tilt_allow = get_allowed_tilt_rad(dist, deg_near, deg_far, d_near, d_far)  # 추가 (+)
        tilt_ex = (tilt - tilt_allow).clamp(min=0.0)  # 추가 (+)
        g_theta = linear_interpolation(tilt_ex, 0.0, math.radians(deg_near))  # 추가 (+)

        ori_reward = base * g_theta  # 추가 (+)
        _push_rew_term(env, "orientation", ori_reward)  # 추가 (+)
        return ori_reward  # 추가 (+)


    @staticmethod
    def time_penalty(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:  # 추가 (+)
        quat = mdp.ObservationFns.get_quaternion_w(env, asset_cfg)  # 추가 (+)
        N = quat.shape[0]  # 추가 (+)
        device, dtype = quat.device, quat.dtype  # 추가 (+)

        per_sec = 1.0 / float(CONFIG["ENV"]["EPISODE_LENGTH_S"])  # 추가 (+)
        val = torch.full((N,), -per_sec, device=device, dtype=dtype)  # 추가 (+)

        dist = l2_norm(get_pos_err_w(env, asset_cfg))  # 추가 (+)
        prev = env.extras.get("_prev_dist", None)  # 추가 (+)
        if (prev is None) or (not isinstance(prev, torch.Tensor)) or (prev.shape != dist.shape):  # 추가 (+)
            prev = dist.detach()  # 추가 (+)
        k_prog = 1.0 / float(CONFIG["REWARD"]["POS_ERR_HALF"])  # 추가 (+)
        val = val + k_prog * (prev - dist)  # 추가 (+)
        env.extras["_prev_dist"] = dist.detach()  # 추가 (+)

        _push_rew_term(env, "time_penalty", val)  # 추가 (+)
        return val  # 추가 (+)


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