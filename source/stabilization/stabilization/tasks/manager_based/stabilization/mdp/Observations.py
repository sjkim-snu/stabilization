import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.utils.html#module-isaaclab.utils.math    
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationData.default_root_state
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.scene.html#isaaclab.scene.InteractiveScene.env_origins
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.ManagerTermBaseCfg
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.SceneEntityCfg

"""
Observation Terms
1. Current position in world frame
2. Spawn position in world frame
3. Linear velocity in world frame
4. Orientation quaternion in world frame
5. Body rate in body frame
"""

class ObservationFns:
    
    """Observation functions get data from the environment."""
    
    @staticmethod
    def get_current_pos_w(
        env: ManagerBasedEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:
        
        """Get the current position of the asset in world frame."""
        
        asset = env.scene[asset_cfg.name]
        current_position_w = asset.data.root_pos_w
        return current_position_w # (N, 3)
    
    @staticmethod
    def get_spawn_pos_w(
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:
        
        """Get the initial spawn position of the quadrotor in the environment."""
        
        asset = env.scene[asset_cfg.name]
        spawn_position = asset.data.default_root_state[:, 0:3] + env.scene.env_origins # (N, 3)
        return spawn_position

    @staticmethod
    def get_lin_vel_w(
        env: ManagerBasedEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:
        
        """Get the linear velocity of the quadrotor in the body frame."""
        
        asset = env.scene[asset_cfg.name]
        lin_vel_w = asset.data.root_lin_vel_w
        return lin_vel_w # (N, 3)

    @staticmethod
    def get_quaternion_w(
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:
        
        """Get the orientation quaternion of the quadrotor in world frame."""
        
        asset = env.scene[asset_cfg.name]
        quaternion_w = asset.data.root_quat_w
        return quaternion_w # (N, 4)
    
    @staticmethod
    def get_ang_vel_b(
        env: ManagerBasedEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")) -> torch.Tensor:
        
        """Get the angular velocity of the quadrotor in the body frame."""
        
        asset = env.scene[asset_cfg.name]
        ang_vel_b = asset.data.root_ang_vel_b
        return ang_vel_b # (N, 3)
    
    
    
@configclass
class ObservationsCfg:
        
    @configclass
    class PolicyCfg(ObsGroup):
        """Configuration for policy observations."""
        
        current_pos_w = ObsTerm(
            func=ObservationFns.get_current_pos_w,
            params={"asset_cfg": SceneEntityCfg(name="Robot")},
        )
        
        spawn_pos_w = ObsTerm(
            func=ObservationFns.get_spawn_pos_w,
            params={"asset_cfg": SceneEntityCfg(name="Robot")},
        )
        
        lin_vel_w = ObsTerm(
            func=ObservationFns.get_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg(name="Robot")},
        )
        
        quaternion_w = ObsTerm(
            func=ObservationFns.get_quaternion_w,
            params={"asset_cfg": SceneEntityCfg(name="Robot")},
        )
        
        ang_vel_b = ObsTerm(
            func=ObservationFns.get_ang_vel_b,
            params={"asset_cfg": SceneEntityCfg(name="Robot")},
        )