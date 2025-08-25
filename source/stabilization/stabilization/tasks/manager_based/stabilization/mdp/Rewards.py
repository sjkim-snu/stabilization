from isaaclab.managers import RewardTermCfg as RewTerm 
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils 

class RewardFunctions:
    
    @staticmethod
    def quaternion_to_rotation(quaternion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        """
        Convert a quaternion to a rotation matrix.
        
        Args:
            quaternion (torch.Tensor): Tensor of shape (N, 4)
        
        Returns:
            A tuple containing roll, pitch, and yaw angles in radians.
            Each angle tensor has shape (N,).
        """
        
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(quaternion, wrap_to_2pi=False)
        return roll, pitch, yaw
    
    @staticmethod
    def position_reward