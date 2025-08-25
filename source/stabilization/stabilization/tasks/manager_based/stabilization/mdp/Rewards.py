import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import stabilization.tasks.manager_based.stabilization.mdp as mdp

"""
Helper functions
"""

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
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot"), 
        norm_half: float = 1.0) -> torch.Tensor:
        
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
        
        error_w = mdp.ObservationFns.position_error_w(env, asset_cfg) # (N, 3)
        error_norm = l2_norm(error_w) # (N,)
        k = k_from_half(norm_half)
        pos_reward = sigmoid(error_norm, k) # (N,)
        return pos_reward
    
    @staticmethod
    def lin_vel_b_sigmoid(
        env: ManagerBasedEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot"), 
        norm_half: float = 1.0) -> torch.Tensor:
        
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
        
        lin_vel_b = mdp.ObservationFns.lin_vel_body(env, asset_cfg)
        lin_vel_norm = l2_norm(lin_vel_b)
        k = k_from_half(norm_half)
        vel_reward = sigmoid(lin_vel_norm, k)
        return vel_reward
    
    @staticmethod
    def ang_vel_b_sigmoid(
        env: ManagerBasedEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot"), 
        norm_half: float = 1.0) -> torch.Tensor:
        
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
        
        ang_vel_b = mdp.ObservationFns.ang_vel_body(env, asset_cfg)
        ang_vel_norm = l2_norm(ang_vel_b)
        k = k_from_half(norm_half)
        ang_vel_reward = sigmoid(ang_vel_norm, k)
        return ang_vel_reward
    
    @staticmethod
    def orientation_sigmoid(
        env: ManagerBasedEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot"), 
        norm_half: float = 0.5) -> torch.Tensor:
        
        """
        Reward based on the orientation (roll, pitch, yaw) using a sigmoid function.
        
        Args:
            env (ManagerBasedEnv): The environment instance.
            asset_cfg (SceneEntityCfg): Name of the quadrotor entity.
            norm_half (float): Half point for the sigmoid function in radians.
        
        Returns:
            reward: Tensor of shape (N,) representing the orientation reward.
            
        Note:
            The reward is 1 when the orientation is zero,
            and 0.5 when the orientation equals norm_half.
        """
        
        roll = mdp.ObservationFns.roll_current(env, asset_cfg)
        pitch = mdp.ObservationFns.pitch_current(env, asset_cfg)
        yaw = mdp.ObservationFns.yaw_current(env, asset_cfg)
        orientation = torch.stack([roll, pitch, yaw], dim=1) # (N, 3)
        orientation_norm = l2_norm(orientation) # (N,)
        k = k_from_half(norm_half)
        orientation_reward = sigmoid(orientation_norm, k) # (N,)
        return orientation_reward


@configclass
class RewardCfg:
    
    pos_err_w = RewTerm(
        func=RewardFns.pos_err_w_sigmoid,
        params={"asset_cfg": SceneEntityCfg(name="Robot"), "norm_half": 1.0},
        weight=1.0,
    )
    
    lin_vel_b = RewTerm(
        func=RewardFns.lin_vel_b_sigmoid,
        params={"asset_cfg": SceneEntityCfg(name="Robot"), "norm_half": 1.0},
        weight=1.0,
    )
    
    ang_vel_b = RewTerm(
        func=RewardFns.ang_vel_b_sigmoid,
        params={"asset_cfg": SceneEntityCfg(name="Robot"), "norm_half": 1.0},
        weight=1.0,
    )
    
    orientation = RewTerm(
        func=RewardFns.orientation_sigmoid,
        params={"asset_cfg": SceneEntityCfg(name="Robot"), "norm_half": 0.5},
        weight=1.0,
    )
        