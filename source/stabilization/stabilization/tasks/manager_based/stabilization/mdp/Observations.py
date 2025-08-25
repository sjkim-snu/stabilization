import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from typing import Optional, Tuple

# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.utils.html#module-isaaclab.utils.math

class ObservationFns:
    
    """
    Helper functions for computing position rewards.
    """
        
    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationData.default_root_state
    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.scene.html#isaaclab.scene.InteractiveScene.env_origins
    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.ManagerTermBaseCfg
    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.SceneEntityCfg
    
    @staticmethod
    def spawn_position_w(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
        
        """
        Get the initial spawn position of the quadrotor in the environment.
        Args:
            env (ManagerBasedEnv): The environment instance.
            asset_name (str): Name of the quadrotor entity.
        
        Returns:
            spawn_position: Tensor of shape (N, 3) representing the initial spawn positions in world frame.
        
        Note:
            default_root_state represents the default pose of each entity in its local frame.
            env_origins represents the origin positions of each entity in the global frame.
        """
        
        asset = env.scene[asset_name]
        return asset.data.default_root_state[:, 0:3] + env.scene.env_origins # (N, 3)

    
    """
    SceneEntityCfg input is required to access the entity in the scene.
    1. ObservationTermCfg inherits ManagerTermBaseCfg.
    2. ManagerTermBaseCfg has attribute 'params' which is a dictionary [str, Any | SceneEntityCfg].
    """
    
    @staticmethod
    def lin_vel_body(
        env: ManagerBasedEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:
        
        """
        Get the linear velocity of the quadrotor in the body frame.
        
        Args:
            env (ManagerBasedEnv): The environment instance.
            asset_cfg (SceneEntityCfg): Name of the quadrotor entity.
        
        Returns:
            lin_vel_body: Tensor of shape (N, 3) representing the linear velocity in body frame.
        """
        
        asset = env.scene[asset_cfg.name]
        lin_vel_body = asset.data.root_lin_vel_b
        return lin_vel_body # (N, 3)
    
    
    @staticmethod
    def position_error_w(
        env: ManagerBasedEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:
        
        """
        Compute the position error of the quadrotor relative to a target position in world frame.
        
        Args:
            env (ManagerBasedEnv): The environment instance.
            asset_cfg (SceneEntityCfg): Name of the quadrotor entity.
        
        Returns:
            position_error: Tensor of shape (N, 3) representing the position error in world frame.
        """
        
        asset = env.scene[asset_cfg.name]
        position_w = asset.data.root_pos_w
        target_w = ObservationFns.spawn_position_w(env, asset_cfg.name) # (N, 3)
        position_error_w = target_w - position_w
        return position_error_w # (N, 3)


    """
    Helper functions for computing orientation rewards.
    """
    
    @staticmethod
    def quaternion_to_orientation(quaternion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        """
        Convert a quaternion to orientation angles.
        Args:
            quaternion (torch.Tensor): Tensor of shape (N, 4)
        
        Returns:
            A tuple containing roll, pitch, and yaw angles in radians.
            Each angle tensor has shape (N,).
        """
        
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(quaternion, wrap_to_2pi=False)
        return roll, pitch, yaw # Each of shape (N,)
    
    @staticmethod
    def roll_current(
        env: ManagerBasedEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:

        """
        Get the current roll angle of the quadrotor in radians.
        
        Args:
            env (ManagerBasedEnv): The environment instance.
            asset_cfg (SceneEntityCfg): Name of the quadrotor entity.
        
        Returns:
            roll: Tensor of shape (N,) representing the roll angles in radians.
        """
        
        asset = env.scene[asset_cfg.name]
        quaternion = asset.data.root_quat_w
        roll, _, _ = ObservationFns.quaternion_to_orientation(quaternion)
        return roll # (N,)
    
    @staticmethod
    def pitch_current(
        env: ManagerBasedEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:
        
        """
        Get the current pitch angle of the quadrotor in radians.
        
        Args:
            env (ManagerBasedEnv): The environment instance.
            asset_cfg (SceneEntityCfg): Name of the quadrotor entity.
        
        Returns:
            pitch: Tensor of shape (N,) representing the pitch angles in radians.
        """
        
        asset = env.scene[asset_cfg.name]
        quaternion = asset.data.root_quat_w
        _, pitch, _ = ObservationFns.quaternion_to_orientation(quaternion)
        return pitch # (N,)
    
    @staticmethod
    def yaw_current(
        env: ManagerBasedEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:
        
        """
        Get the current yaw angle of the quadrotor in radians.
        
        Args:
            env (ManagerBasedEnv): The environment instance.
            asset_cfg (SceneEntityCfg): Name of the quadrotor entity.
        
        Returns:
            yaw: Tensor of shape (N,) representing the yaw angles in radians.
        """
        
        asset = env.scene[asset_cfg.name]
        quaternion = asset.data.root_quat_w
        _, _, yaw = ObservationFns.quaternion_to_orientation(quaternion)
        return yaw # (N,)
    
    @staticmethod
    def ang_vel_body(
        env: ManagerBasedEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:
        
        """
        Get the angular velocity of the quadrotor in the body frame.
        
        Args:
            env (ManagerBasedEnv): The environment instance.
            asset_cfg (SceneEntityCfg): Name of the quadrotor entity.
        
        Returns:
            ang_vel_body: Tensor of shape (N, 3) representing the angular velocity in body frame.
        """
        
        asset = env.scene[asset_cfg.name]
        ang_vel_body = asset.data.root_ang_vel_b
        return ang_vel_body # (N, 3)
    
    
@configclass
class ObservationsCfg:
        
    @configclass
    class PolicyCfg(ObsGroup):
        """Configuration for policy observations."""
        
        pos_err_w = ObsTerm(
            func=ObservationFns.position_error_w,
            params={"asset_cfg": SceneEntityCfg(name="Robot"), "target_pos": None},
        )
        
        lin_vel_b = ObsTerm(
            func=ObservationFns.lin_vel_body,
            params={"asset_cfg": SceneEntityCfg(name="Robot")},
        )
        
        ang_vel_b = ObsTerm(
            func=ObservationFns.ang_vel_body,
            params={"asset_cfg": SceneEntityCfg(name="Robot")},
        )
        
        roll = ObsTerm(
            func=ObservationFns.roll_current,
            params={"asset_cfg": SceneEntityCfg(name="Robot")},
        )
        
        pitch = ObsTerm(
            func=ObservationFns.pitch_current,
            params={"asset_cfg": SceneEntityCfg(name="Robot")},
        )
        
        yaw = ObsTerm(
            func=ObservationFns.yaw_current,
            params={"asset_cfg": SceneEntityCfg(name="Robot")},
        )

