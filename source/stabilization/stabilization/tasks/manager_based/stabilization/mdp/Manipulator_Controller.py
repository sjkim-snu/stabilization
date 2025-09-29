from __future__ import annotations
from dataclasses import dataclass
from isaaclab.utils import configclass
import torch
from typing import Sequence, Tuple
from stabilization.tasks.manager_based.stabilization.config import load_parameters

CONFIG = load_parameters()

"""
Helper functions for cascade controller
"""

class ManipulatorControllerFns:
    
    @staticmethod
    def to_tensor_1x(array: Sequence[float],
                     dtype: torch.dtype,
                     device: torch.device) -> torch.Tensor:
        t = torch.as_tensor(array, dtype=dtype, device=device)
        return t.reshape(1, t.numel())                         # (1, L)

    @staticmethod
    def to_tensor_Nx(array: Sequence[float],
                     num_envs: int,
                     dtype: torch.dtype,
                     device: torch.device) -> torch.Tensor:
        t1 = ManipulatorControllerFns.to_tensor_1x(array, dtype, device)  # (1, L)
        return t1.expand(num_envs, t1.shape[1])                # (N, L)
    
    @staticmethod
    def matrix_to_quaternion(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

        t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] 
        w = torch.empty_like(t)
        x = torch.empty_like(t)
        y = torch.empty_like(t)
        z = torch.empty_like(t)

        mask = t > 0
        s = torch.sqrt(t[mask] + 1.0) * 2 
        w[mask] = 0.25 * s
        x[mask] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
        y[mask] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
        z[mask] = (R[mask, 1, 0] - R[mask, 0, 1]) / s

        mask_x = (~mask) & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
        s = torch.sqrt(1.0 + R[mask_x, 0, 0] - R[mask_x, 1, 1] - R[mask_x, 2, 2] + eps) * 2
        w[mask_x] = (R[mask_x, 2, 1] - R[mask_x, 1, 2]) / s
        x[mask_x] = 0.25 * s
        y[mask_x] = (R[mask_x, 0, 1] + R[mask_x, 1, 0]) / s
        z[mask_x] = (R[mask_x, 0, 2] + R[mask_x, 2, 0]) / s

        mask_y = (~mask) & (~mask_x) & (R[..., 1, 1] > R[..., 2, 2])
        s = torch.sqrt(1.0 + R[mask_y, 1, 1] - R[mask_y, 0, 0] - R[mask_y, 2, 2] + eps) * 2
        w[mask_y] = (R[mask_y, 0, 2] - R[mask_y, 2, 0]) / s
        x[mask_y] = (R[mask_y, 0, 1] + R[mask_y, 1, 0]) / s
        y[mask_y] = 0.25 * s
        z[mask_y] = (R[mask_y, 1, 2] + R[mask_y, 2, 1]) / s

        mask_z = (~mask) & (~mask_x) & (~mask_y)
        s = torch.sqrt(1.0 + R[mask_z, 2, 2] - R[mask_z, 0, 0] - R[mask_z, 1, 1] + eps) * 2
        w[mask_z] = (R[mask_z, 1, 0] - R[mask_z, 0, 1]) / s
        x[mask_z] = (R[mask_z, 0, 2] + R[mask_z, 2, 0]) / s
        y[mask_z] = (R[mask_z, 1, 2] + R[mask_z, 2, 1]) / s
        z[mask_z] = 0.25 * s

        quat = torch.stack([w, x, y, z], dim=-1)
        quat = quat / (quat.norm(dim=-1, keepdim=True) + eps)

        return quat  # (N, 4)
    
    @staticmethod
    def hamilton_product(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:

        w1, x1, y1, z1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        w2, x2, y2, z2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return torch.stack([w, x, y, z], dim=1)


@configclass
class ManipulatorCascadeControllerCfg:
    
    # Position controller parameters
    pos_P: Tuple[float, float, float] = CONFIG["MANIPULATOR_CONTROLLER"]["POS_P"]
    pos_slew_max: float = CONFIG["MANIPULATOR_CONTROLLER"]["POS_SLEW_MAX"]

    # Velocity controller parameters
    vel_P: Tuple[float, float, float] = CONFIG["MANIPULATOR_CONTROLLER"]["VEL_P"]
    vel_I: Tuple[float, float, float] = CONFIG["MANIPULATOR_CONTROLLER"]["VEL_I"]
    
    # Attitude controller parameters
    att_P: Tuple[float, float, float] = CONFIG["MANIPULATOR_CONTROLLER"]["ATT_P"]
    tilt_max: Tuple[float] = CONFIG["MANIPULATOR_CONTROLLER"]["TILT_MAX"]

    # Body rate controller parameters
    rate_P: Tuple[float, float, float] = CONFIG["MANIPULATOR_CONTROLLER"]["RATE_P"]
    rate_I: Tuple[float, float, float] = CONFIG["MANIPULATOR_CONTROLLER"]["RATE_I"]

    # Saturation limits
    vel_limit: Tuple[float, float, float] = CONFIG["MANIPULATOR_CONTROLLER"]["VEL_LIMIT"]
    ang_vel_limit: Tuple[float, float, float] = CONFIG["MANIPULATOR_CONTROLLER"]["ANG_VEL_LIMIT"]

    # Saturation limits for anti windup
    acc_limit: Tuple[float, float, float] = CONFIG["MANIPULATOR_CONTROLLER"]["ACC_LIMIT"]
    torque_limit: Tuple[float, float, float] = CONFIG["MANIPULATOR_CONTROLLER"]["TORQUE_LIMIT"]

    # Integral limits for anti windup
    vel_I_clamp: Tuple[float, float, float] = CONFIG["MANIPULATOR_CONTROLLER"]["VEL_I_CLAMP"]
    rate_I_clamp: Tuple[float, float, float] = CONFIG["MANIPULATOR_CONTROLLER"]["RATE_I_CLAMP"]

    # Numerics
    eps: float = 1e-6
    

class ManipulatorCascadeController:
    
    def __init__(self,
                 num_envs: int,
                 dt: float,
                 cfg: ManipulatorCascadeControllerCfg,
                 dtype: torch.dtype = torch.float32):
        
        self.num_envs = num_envs
        self.dt = dt
        self.cfg = cfg
        self.dtype = dtype
        self.device = torch.device(CONFIG["LAUNCHER"]["DEVICE"])

        # Tensors for gains and limits (1,3)
        self._pos_P          = ManipulatorControllerFns.to_tensor_1x(self.cfg.pos_P,         self.dtype, self.device)
        self._vel_P          = ManipulatorControllerFns.to_tensor_1x(self.cfg.vel_P,         self.dtype, self.device)
        self._vel_I          = ManipulatorControllerFns.to_tensor_1x(self.cfg.vel_I,         self.dtype, self.device)
        self._att_P          = ManipulatorControllerFns.to_tensor_1x(self.cfg.att_P,         self.dtype, self.device)
        self._rate_P         = ManipulatorControllerFns.to_tensor_1x(self.cfg.rate_P,        self.dtype, self.device)
        self._rate_I         = ManipulatorControllerFns.to_tensor_1x(self.cfg.rate_I,        self.dtype, self.device)
        self._tilt_max       = ManipulatorControllerFns.to_tensor_1x(self.cfg.tilt_max,      self.dtype, self.device)
        self._vel_limit      = ManipulatorControllerFns.to_tensor_1x(self.cfg.vel_limit,     self.dtype, self.device)
        self._ang_vel_limit  = ManipulatorControllerFns.to_tensor_1x(self.cfg.ang_vel_limit, self.dtype, self.device)
        self._pos_slew_max   = ManipulatorControllerFns.to_tensor_1x(self.cfg.pos_slew_max,  self.dtype, self.device)
        
        # Integral term limits for anti windup (N,3)
        self._vel_I_clamp    = ManipulatorControllerFns.to_tensor_1x(self.cfg.vel_I_clamp,   self.dtype, self.device)
        self._rate_I_clamp   = ManipulatorControllerFns.to_tensor_1x(self.cfg.rate_I_clamp,  self.dtype, self.device)

        # Acceleration, rate limits are used for anti windup (N,3)
        self._acc_limit      = ManipulatorControllerFns.to_tensor_1x(self.cfg.acc_limit,     self.dtype, self.device)
        self._torque_limit   = ManipulatorControllerFns.to_tensor_1x(self.cfg.torque_limit,    self.dtype, self.device)
        
        # Initialize integral terms (N,3)
        self._vel_int        = torch.zeros((num_envs, 3), dtype=self.dtype, device=self.device)
        self._ang_vel_int    = torch.zeros((num_envs, 3), dtype=self.dtype, device=self.device)
        
        # Store eps for float
        self._eps: float = float(self.cfg.eps)
        
    def position_control(self,
                         pos_w: torch.Tensor,    # (N, 3)
                         pos_sp_w: torch.Tensor  # (N, 3)
                         ) -> torch.Tensor:
        
        pos_err = pos_sp_w - pos_w  # (N, 3)
        
        # Slew rate limit
        max_norm = torch.as_tensor(self._pos_slew_max, dtype=self.dtype, device=self.device)
        err_norm = torch.linalg.norm(pos_err, dim=-1, keepdim=True)      # (N, 1)
        scale = torch.clamp(max_norm / (err_norm + self._eps), max=1.0)  # (N, 1)
        pos_err = pos_err * scale

        # Proportional term
        vel_sp_w = self._pos_P * pos_err                      
        vel_sp_w = torch.clamp(vel_sp_w, -self._vel_limit, self._vel_limit)
        
        # Store for logging
        self._pos_w = pos_w
        self._pos_sp_w = pos_sp_w
        
        return vel_sp_w  # (N, 3)
    
    def velocity_control(self,
                         vel_w: torch.Tensor,    # (N, 3)
                         vel_sp_w: torch.Tensor  # (N, 3)
                         ) -> torch.Tensor:
        
        vel_err = vel_sp_w - vel_w  # (N, 3)
        vel_int_new = self._vel_int + vel_err * self.dt
        
        # Proportional and clamped Integral terms
        P_term = self._vel_P * vel_err
        I_term = self._vel_I * vel_int_new
        I_term = torch.clamp(I_term, -self._vel_I_clamp, self._vel_I_clamp)
        acc_sp_w = P_term + I_term
        
        # Anti windup: only integrate if not saturated
        acc_sp_lim_w = torch.clamp(acc_sp_w, -self._acc_limit, self._acc_limit)
        saturation = (acc_sp_w > self._acc_limit) | (acc_sp_w < -self._acc_limit)
        same_dir = torch.sign(vel_err) == torch.sign(acc_sp_w)
        reject = saturation & same_dir
        self._vel_int = torch.where(reject, self._vel_int, vel_int_new)
        
        # Compute final output with updated integral
        I_term = self._vel_I * self._vel_int
        I_term = torch.clamp(I_term, -self._vel_I_clamp, self._vel_I_clamp)
        acc_sp_w = P_term + I_term
        
        # Saturate final output
        acc_sp_w = torch.clamp(acc_sp_w, -self._acc_limit, self._acc_limit)
        
        # Store for logging
        self._acc_sp_w = acc_sp_w
        
        return acc_sp_w   # (N,3)
        
    def acc_yaw_to_quaternion_thrust(
        self,
        acc_sp_w: torch.Tensor,   # (N,3)
        yaw_sp: torch.Tensor,     # (N,1)
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Initial settings
        N = self.num_envs
        tilt_max = torch.deg2rad(self._tilt_max).expand(N,1)  # (N,1)

        # Compute desired thrust direction (b3)
        g = ManipulatorControllerFns.to_tensor_Nx([0.0, 0.0, -9.81], N, self.dtype, self.device)
        thrust_w_raw = acc_sp_w - g                                          # (N,3)
        t_norm_raw = torch.norm(thrust_w_raw, dim=1, keepdim=True)
        t_norm_raw = torch.clamp_min(t_norm_raw, float(self._eps))           # (N,1)
        b3_raw = thrust_w_raw / t_norm_raw                                   # (N,3)

        # Compute tilt for limitation
        cos = b3_raw[:, 2:3].clamp(-1.0, 1.0)                                # (N,1)
        sin = torch.sqrt((1.0 - cos * cos).clamp_min(self._eps))             # (N,1)
        s_max = torch.sin(tilt_max).clamp_min(0.0).clamp_max(1.0)            # (N,1)
        scale_xy = torch.minimum(torch.ones_like(sin), s_max / (sin + self._eps))
        over = sin > (s_max + 1e-6)

        # Apply tilt limitation
        b3_xy_lim = b3_raw[:, 0:2] * torch.where(over, scale_xy, torch.ones_like(scale_xy))
        b3_z_lim  = torch.sqrt((1.0 - (b3_xy_lim ** 2).sum(dim=1, keepdim=True)).clamp_min(self._eps))
        b3 = torch.cat([b3_xy_lim, b3_z_lim], dim=1)      

        # Compensate thrust to maintain altitude
        Tz_raw = thrust_w_raw[:, 2:3]                                        # (N,1)
        t_norm = (Tz_raw / b3[:, 2:3]).clamp_min(float(self._eps))           # (N,1)
        thrust_w = b3 * t_norm                                               # (N,3)

        # Set tensor dimensions
        if yaw_sp.ndim == 2 and yaw_sp.shape[1] == 1:
            yaw_sp = yaw_sp.squeeze(-1)                                      # (N,)
        
        # Compute desired heading direction (b1)
        c_yaw = torch.cos(yaw_sp)                                            # (N,)
        s_yaw = torch.sin(yaw_sp)                                            # (N,)
        b1 = torch.stack([c_yaw, s_yaw, torch.zeros_like(c_yaw)], dim=1)     # (N,3)

        # Compute orthonormal basis (b1, b2, b3)
        b2 = torch.cross(b3, b1, dim=1)                                      # (N,3)
        b2 = b2 / (torch.norm(b2, dim=1, keepdim=True) + 1e-12)      
        b1 = torch.cross(b2, b3, dim=1)                               
        b1 = b1 / (torch.norm(b1, dim=1, keepdim=True) + 1e-12)      

        # Rotation matrix to quaternion
        R = torch.stack([b1, b2, b3], dim=2)                                 # (N,3,3)
        quat_sp_w = ManipulatorControllerFns.matrix_to_quaternion(R, self._eps)         # (N,4)

        # Store for logging
        self._thrust_w   = thrust_w
        self._thrust_norm= t_norm
        self._quat_sp_w  = quat_sp_w

        return quat_sp_w, t_norm


    def attitude_control(self,
                         quat_w: torch.Tensor,
                         quat_sp_w: torch.Tensor):
        
        # Compute Quaternion error
        quat_w_conj = quat_w * torch.tensor([1, -1, -1, -1], device=self.device, dtype=self.dtype)
        quat_err = ManipulatorControllerFns.hamilton_product(quat_w_conj, quat_sp_w)  # (N, 4)

        # Extract scalar and vector parts
        q0 = quat_err[:, 0:1]    # (N, 1)
        qv = quat_err[:, 1:4]    # (N, 3)
        
        # Ensure shortest path
        qv = torch.where(q0 < 0, -qv, qv)
        att_err = 2.0 * qv
        
        # Proportional term
        ang_vel_sp_b = self._att_P * att_err    # (N, 3)
        
        # Saturate
        ang_vel_sp_b = torch.clamp(ang_vel_sp_b, -self._ang_vel_limit, self._ang_vel_limit)

        # Store for logging
        self._ang_vel_sp_b = ang_vel_sp_b

        return ang_vel_sp_b  # (N, 3)

    def body_rate_control(self, 
                          inertia_diag: torch.Tensor,  # (N, 3)
                          ang_vel_b: torch.Tensor,     # (N, 3)
                          ang_vel_sp_b: torch.Tensor,  # (N, 3)
                          tau_g_b: torch.Tensor = None # (N, 3)
                          ) -> torch.Tensor:
        
        # Compute error and new integral
        ang_vel_err   = ang_vel_sp_b - ang_vel_b
        i_new = self._ang_vel_int + ang_vel_err * self.dt

        # Proportional and clamped Integral terms
        P = self._rate_P * ang_vel_err
        I = self._rate_I * i_new
        I = torch.clamp(I, -self._rate_I_clamp, self._rate_I_clamp)

        # Initial torque command and saturation
        tau_cmd = inertia_diag * (P + I)  # [N·m]
        tau_lim = torch.clamp(tau_cmd, -self._torque_limit, self._torque_limit)

        # Anti windup: only integrate if not saturated and error in same direction
        saturation = (tau_cmd.abs() > self._torque_limit)
        same_dir = torch.sign(ang_vel_err) == torch.sign(tau_cmd)
        reject = saturation & same_dir
        self._ang_vel_int = torch.where(reject, self._ang_vel_int, i_new)

        # Recompute final torque command with updated integral
        I = self._rate_I * self._ang_vel_int
        I = torch.clamp(I, -self._rate_I_clamp, self._rate_I_clamp)
        tau_cmd = inertia_diag * (P + I)
        tau_cmd = torch.clamp(tau_cmd, -self._torque_limit, self._torque_limit)

        # Include gyroscopic effects
        gyro = torch.cross(ang_vel_b, inertia_diag * ang_vel_b, dim=1)
        if tau_g_b is None:                                                          
            tau_g_b = torch.zeros_like(ang_vel_b)                                    

        u_unsat = tau_cmd + gyro                                                     
        torque_sp_b = torch.clamp(u_unsat - tau_g_b, -self._torque_limit, self._torque_limit) 

        # Store for debugging
        self._tau_cmd_b   = tau_cmd         # Before gyroscopic effects
        self._torque_sp_b = torque_sp_b     # Final output

        return torque_sp_b
