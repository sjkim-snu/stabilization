"""
Action configuration
This module defines the action space for the stabilization task
For more details, refer to the Isaac Lab documentation :
https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#module-isaaclab.envs.mdp.actions
"""

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
import stabilization.tasks.manager_based.stabilization.envs as envs
from typing import List, Optional
from dataclasses import dataclass, field
from isaaclab.utils import configclass
import isaaclab.envs as env

# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.ActionTermCfg

@configclass
class BaseControllerCfg(ActionTermCfg):
    """Base configuration for controllers."""

    robot_entity_name: str = "Robot"        # Name of the quadrotor entity
    body_name: Optional[str] = None         # If None, apply to entire quadrotor
    
    arm_length: float = 0.046               # Length of the arm (for quadrotor)
    k_f_rpm2: float = 6.11e-8               # Thrust coefficient (N/rpm^2)
    k_m_rpm2: float = 1.5e-9                # Motor torque coefficient (Nm/rpm^2)
    coeff_is_rpm2: bool = True       
    
    w_min_rpm: float = 0.0                  # Minimum motor rpm
    w_max_rpm: float = 35000.0              # Maximum motor rpm
    
    rotor_dirs: List[int] = field(default_factory=lambda: [1, -1, 1, -1])  # Rotor directions
    rotor_xy: List[List[float]] = field(
        default_factory=lambda: [
            [+1, +1], # Rotor 1
            [-1, +1], # Rotor 2
            [-1, -1], # Rotor 3
            [+1, -1], # Rotor 4
        ]
    )  # Rotor positions in the quadrotor frame
    
    clamp_actions: bool = True  # Whether to clamp actions to the valid range




# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.ActionTerm

class BaseController(ActionTerm):
    """Base class for controllers"""

    def __init__(self, cfg: BaseControllerCfg, env) -> None:
        super().__init__(cfg, env)
        self.cfg: BaseControllerCfg = cfg
        
        # Get the robot entity from the Scene
        self._robot = self._env.scene[self.cfg.robot_entity_name]
            
        # Convert rpm to rad/s
        self._rpm_to_rad = 2.0 * torch.pi / 60.0 
        self._rad_to_rpm = 60.0 / (2.0 * torch.pi)
        
        # Convert coefficients to rad/s
        factor = (self._rad_to_rpm ** 2)
        if self.cfg.coeff_is_rpm2:
            self._k_f = self.cfg.k_f_rpm2 * factor
            self._k_m = self.cfg.k_m_rpm2 * factor  
        else:
            self._k_f = self.cfg.k_f_rpm2
            self._k_m = self.cfg.k_m_rpm2
        
        # Convert min/max rpm to rad/s
        self._w_min = self.cfg.w_min_rpm * self._rpm_to_rad
        self._w_max = self.cfg.w_max_rpm * self._rpm_to_rad
        
        # Set rotor positions and spin directions
        l = self.cfg.arm_length
        rotor_xy = torch.tensor(self.cfg.rotor_xy, dtype=torch.float32) * l   # (4,2)
        rotor_dirs = torch.tensor(self.cfg.rotor_dirs, dtype=torch.float32)   # (4,)
        self._rotor_pos_xy = rotor_xy
        self._rotor_dirs   = rotor_dirs
        
        # Set action tensors
        N = self._env.num_envs
        self._thrust = torch.zeros(N, 1, 3)
        self._moment = torch.zeros(N, 1, 3) 
        self._omega  = torch.zeros(N, 4)     
        
    
    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.cfg.clamp_actions:
            actions = actions.clamp(-1.0, 1.0)
            
        u = (actions + 1.0) / 2.0  # Scale to [0, 1]
        self._omega = u * (self._w_max - self._w_min) + self._w_min  # Scale to [w_min, w_max]
        return self._omega
    
    def apply_actions(self) -> None:
        w2 = self._omega ** 2                      # (N,4)
        f  = self._k_f * w2                        # (N,4)

        Fz = f.sum(dim=1)                          # (N,)
        self._thrust[:, 0, 0] = 0.0
        self._thrust[:, 0, 1] = 0.0
        self._thrust[:, 0, 2] = Fz                 # (N,)

        x = self._rotor_pos_xy[:, 0]               # (4,)
        y = self._rotor_pos_xy[:, 1]               # (4,)

        tx = (y * f).sum(dim=1)                    # (N,)
        ty = (-x * f).sum(dim=1)                   # (N,)

        tz = (self._rotor_dirs.unsqueeze(0) * (self._k_m * w2)).sum(dim=1)  # (N,)

        self._moment[:, 0, 0] = tx
        self._moment[:, 0, 1] = ty
        self._moment[:, 0, 2] = tz

        self._robot.set_external_force_and_torque(
            self._thrust, self._moment, body_ids=self._robot.root_body_indices
        )
        
    @property
    def action_dim(self) -> int:
        return 4
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._omega