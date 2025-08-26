from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.assets import AssetBase, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from dataclasses import dataclass
from isaaclab.utils import configclass
from abc import abstractmethod

import stabilization.tasks.manager_based.stabilization.envs as envs
import isaaclab.envs as env
import torch
import math


"""
Action configuration
BaseControllerCfg includes parameters for quadrotor dynamics and action processing.
BaseController implements action, force/torque computation, and application to the quadrotor.
In this script, body frame coordinate system is defined as:
- +x: front, +y: left, +z: up
- Roll: rotation around +x (right-hand rule)
- Pitch: rotation around +y (right-hand rule)
- Yaw: rotation around +z (right-hand rule)
Note that NED (North-East-Down) is not used here.
"""

# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.utils.html#module-isaaclab.utils.configclass
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.ActionTermCfg

@configclass
class BaseControllerCfg(ActionTermCfg):
    
    """
    Parameters for a quadrotor controller and action processing.
    Coefficients are based on : https://github.com/isaac-sim/IsaacLab/discussions/1701
    Quadrotor config based on : https://www.bitcraze.io/products/old-products/crazyflie-2-1/
    """
    
    asset_name: str = "Robot"               # Name of the quadrotor entity
    
    arm_length: float = 0.046               # Length of the arm (for quadrotor)
    k_f_rpm2: float = 6.11e-8               # Thrust coefficient (N/rpm^2)
    k_m_rpm2: float = 1.5e-9                # Motor torque coefficient (Nm/rpm^2)
    coeff_is_rpm2: bool = True              # Whether k_f and k_m are in rpm^2 units
    
    w_min_rpm: float = 0.0                  # Minimum motor rpm
    w_max_rpm: float = 35000.0              # Maximum motor rpm
    
    rotor_dirs: list[float] = [+1.0, -1.0, +1.0, -1.0]     # Rotor spin directions (+1:CCW, -1:CW)
    rotor_xy: list[list[float]] = [                        # Rotor positions (+x: front, +y: left)
        [+1.0, +1.0],   # front left
        [+1.0, -1.0],   # front right
        [-1.0, -1.0],   # rear right
        [-1.0, +1.0],   # rear left
    ]
    
    clamp_actions: bool = True              # Whether to clamp actions to the valid range



# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.ActionTerm

class BaseController(ActionTerm):
    
    """
    The action term for a quadrotor controller, which defines action processing (motor speeds), 
    thrust/torque computation, and application to the quadrotor.
    
    Processing of actions: This operation is performed once per environment step
    and is responsible for pre-processing the raw actions sent to the environment.
    
    Applying actions: This operation is performed once per simulation step 
    and is responsible for applying the processed actions to the asset managed by the term.
    """

    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.ActionTermCfg
    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.html#isaaclab.envs.ManagerBasedEnv
    # https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/managers/manager_base.html#ManagerTermBase
    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.AssetBase
    
    def __init__(self, cfg: BaseControllerCfg, env: ManagerBasedEnv):
        
        """
        Initialize the action term.
        Args:
            cfg (ActionTermCfg): Configuration for the action term.
            env (ManagerBasedEnv): The environment to which the action term is applied.
        """
        
        super().__init__(cfg, env)
        
        """
        How below script works:
        1. BaseControllerCfg inherits ActionTermCfg, which defines asset_name as "Robot".
        2. BaseController inherits ActionTerm, which takes cfg and env as input.
        3. ActionTerm inherits ManagerTermBase, which stores cfg and env as self.cfg and self._env.
        4. Note that env stands for ManagerBasedEnv, and cfg stands for BaseControllerCfg.
        5. ManagerBasedEnv defines self.scene as InteractiveScene(self.cfg.scene).
        6. Finally, self._env.scene[self.cfg.asset_name] stands for the quadrotor asset.
        7. The quadrotor asset is of type AssetBase, which has attribute 'device'.
        """
        
        self._asset: AssetBase = self._env.scene[self.cfg.asset_name]
        device = self._asset.device

        # Define conversion factors
        self._rpm_to_rad = torch.tensor(2.0 * torch.pi / 60.0, device=device)
        self._rad_to_rpm = torch.tensor(60.0 / (2.0 * torch.pi), device=device)
        
        # Convert coefficients to rad/s
        if self.cfg.coeff_is_rpm2:
            self._k_f = torch.tensor(self.cfg.k_f_rpm2, device=device) * (self._rad_to_rpm ** 2)
            self._k_m = torch.tensor(self.cfg.k_m_rpm2, device=device) * (self._rad_to_rpm ** 2)
        else:
            self._k_f = torch.tensor(self.cfg.k_f_rpm2, device=device)
            self._k_m = torch.tensor(self.cfg.k_m_rpm2, device=device)
        
        # Convert min/max rpm to rad/s
        self._w_min = torch.tensor(self.cfg.w_min_rpm, device=device) * self._rpm_to_rad
        self._w_max = torch.tensor(self.cfg.w_max_rpm, device=device) * self._rpm_to_rad
        
        # Set rotor positions and spin directions
        l = self.cfg.arm_length
        rotor_xy = torch.tensor(self.cfg.rotor_xy, dtype=torch.float32, device=device) * l   # (4,2)
        rotor_dirs = torch.tensor(self.cfg.rotor_dirs, dtype=torch.float32, device=device)   # (4,)
        self._rotor_pos_xy = rotor_xy
        self._rotor_dirs   = rotor_dirs
        
        # Initialize action tensors
        N = self._env.num_envs
        self._raw = torch.zeros(N, 4, device=device)          # (N,4) raw actions
        self._thrust = torch.zeros(N, 1, 3, device=device)    # (N,1,3) total thrust force
        self._moment = torch.zeros(N, 1, 3, device=device)    # (N,1,3) total moment/torque
        self._omega = torch.zeros(N, 4, device=device)        # (N,4) processed motor speeds (rad/s)
        
        # Get body ids and names to apply forces/torques
        # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.Articulation.find_bodies
        
        ids, names = self._asset.find_bodies(".*", preserve_order=True)
        self._body_ids = [int(ids[0])]
        

    """
    Properties
    """
    
    @property
    def action_dim(self) -> int:
        return 4  # Dimension of the action space which is 4 for quadrotor
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw  # raw actions before processing
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._omega  # processed actions (motor speeds in rad/s)
        
    
    """
    Operations
    """
    
    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        
        """
        Process raw actions to compute motor speeds.
        This function is called once per environment step.
        
        Args:
            actions (torch.Tensor): Raw actions in the range [-1, 1].
        Returns:
            torch.Tensor: Processed motor speeds in rad/s.
        """
        
        # Use same device
        actions = actions.to(self._omega.device)
        
        # Store raw actions (for debugging & logging)
        self._raw = actions
        
        # Clamp actions if needed
        if self.cfg.clamp_actions:
            actions = actions.clamp(-1.0, 1.0)
        
        # Convert actions [-1, 1] to motor speeds omega [w_min, w_max]
        u = (actions + 1.0) / 2.0                                       # scale to [0, 1]
        self._omega = u * (self._w_max - self._w_min) + self._w_min     # scale to [w_min, w_max]
        return self._omega
    
    
    def apply_actions(self):
        
        """
        Process rotor speeds to compute and apply forces/torques to the quadrotor.
        This function is called once per simulation step.
        """
        
        # Compute thrust from rotor speeds
        w2 = self._omega ** 2                                   # element-wise square of omega : (N,4)
        f  = self._k_f * w2                                     # thrust force from each rotor : (N,4)
        Fz = f.sum(dim=1)                                       # total thrust of each quadrotor : (N,)
        
        # Apply forces to the quadrotor
        self._thrust.zero_()                                    # Initialize to zero
        self._thrust[:, 0, 2] = Fz                              # thrust along +z : (N,)

        # Compute moments from rotor pos and dirs
        x = self._rotor_pos_xy[:,0]                             # Rotor x positions : (4,)    
        y = self._rotor_pos_xy[:,1]                             # Rotor y positions : (4,)   
        tx = (f * y).sum(dim=1)                                 # Roll : (N,)
        ty = (f * -x).sum(dim=1)                                # Pitch : (N,)
        tz = (self._rotor_dirs * (self._k_m * w2)).sum(dim=1)   # Yaw : (N,)
        
        # Apply moments to the quadrotor
        self._moment.zero_()                                    # Initialize to zero 
        self._moment[:, 0, 0] = tx                              # torque around +x : (N,) 
        self._moment[:, 0, 1] = ty                              # torque around +y : (N,)
        self._moment[:, 0, 2] = tz                              # torque around +z : (N,)
        
        # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.RigidObject.set_external_force_and_torque
        
        self._asset.set_external_force_and_torque(
            forces=self._thrust,
            torques=self._moment,
            body_ids=self._body_ids,
            is_global=False,
            )