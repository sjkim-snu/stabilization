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

class ControllerFns:
    
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
        t1 = ControllerFns.to_tensor_1x(array, dtype, device)  # (1, L)
        return t1.expand(num_envs, t1.shape[1])                # (N, L)
    
    @staticmethod
    def matrix_to_quaternion(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

        t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]  # trace
        w = torch.empty_like(t)
        x = torch.empty_like(t)
        y = torch.empty_like(t)
        z = torch.empty_like(t)

        mask = t > 0
        s = torch.sqrt(t[mask] + 1.0) * 2  # s = 4*w
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
class CascadeControllerCfg:
    
    # Position controller gains
    pos_P: Tuple[float, float, float] = CONFIG["CASCADE_CONTROLLER"]["POS_P"]

    # Velocity controller gains
    vel_P: Tuple[float, float, float] = CONFIG["CASCADE_CONTROLLER"]["VEL_P"]
    vel_I: Tuple[float, float, float] = CONFIG["CASCADE_CONTROLLER"]["VEL_I"]

    # Attitude controller gains
    att_P: Tuple[float, float, float] = CONFIG["CASCADE_CONTROLLER"]["ATT_P"]

    # Body rate controller gains
    rate_P: Tuple[float, float, float] = CONFIG["CASCADE_CONTROLLER"]["RATE_P"]
    rate_I: Tuple[float, float, float] = CONFIG["CASCADE_CONTROLLER"]["RATE_I"]

    # Saturation limits
    vel_limit: Tuple[float, float, float] = CONFIG["CASCADE_CONTROLLER"]["VEL_LIMIT"]
    ang_vel_limit: Tuple[float, float, float] = CONFIG["CASCADE_CONTROLLER"]["ANG_VEL_LIMIT"]

    # Saturation limits for anti windup
    acc_limit: Tuple[float, float, float] = CONFIG["CASCADE_CONTROLLER"]["ACC_LIMIT"]
    torque_limit: Tuple[float, float, float] = CONFIG["CASCADE_CONTROLLER"]["TORQUE_LIMIT"]

    # Integral limits for anti windup
    vel_I_clamp: Tuple[float, float, float] = CONFIG["CASCADE_CONTROLLER"]["VEL_I_CLAMP"]
    rate_I_clamp: Tuple[float, float, float] = CONFIG["CASCADE_CONTROLLER"]["RATE_I_CLAMP"]

    # Numerics
    eps: float = 1e-6
    

class CascadeController:
    
    def __init__(self,
                 num_envs: int,
                 dt: float,
                 cfg: CascadeControllerCfg,
                 dtype: torch.dtype = torch.float32):
        
        self.num_envs = num_envs
        self.dt = dt
        self.cfg = cfg
        self.dtype = dtype
        self.device = torch.device(CONFIG["LAUNCHER"]["DEVICE"])

        # Tensors for gains and limits (1,3)
        self._pos_P          = ControllerFns.to_tensor_1x(self.cfg.pos_P,         self.dtype, self.device)
        self._vel_P          = ControllerFns.to_tensor_1x(self.cfg.vel_P,         self.dtype, self.device)
        self._vel_I          = ControllerFns.to_tensor_1x(self.cfg.vel_I,         self.dtype, self.device)
        self._att_P          = ControllerFns.to_tensor_1x(self.cfg.att_P,         self.dtype, self.device)
        self._rate_P         = ControllerFns.to_tensor_1x(self.cfg.rate_P,        self.dtype, self.device)
        self._rate_I         = ControllerFns.to_tensor_1x(self.cfg.rate_I,        self.dtype, self.device)

        self._vel_limit      = ControllerFns.to_tensor_1x(self.cfg.vel_limit,     self.dtype, self.device)
        self._ang_vel_limit      = ControllerFns.to_tensor_1x(self.cfg.ang_vel_limit,     self.dtype, self.device)

        # Integral term limits for anti windup (N,3)
        self._vel_I_clamp    = ControllerFns.to_tensor_1x(self.cfg.vel_I_clamp,   self.dtype, self.device)
        self._rate_I_clamp   = ControllerFns.to_tensor_1x(self.cfg.rate_I_clamp,  self.dtype, self.device)

        # Acceleration, rate limits are used for anti windup (N,3)
        self._acc_limit      = ControllerFns.to_tensor_1x(self.cfg.acc_limit,     self.dtype, self.device)
        self._torque_limit     = ControllerFns.to_tensor_1x(self.cfg.torque_limit,    self.dtype, self.device)
        
        # Initialize integral terms (N,3)
        self._vel_int        = torch.zeros((num_envs, 3), dtype=self.dtype, device=self.device)
        self._ang_vel_int    = torch.zeros((num_envs, 3), dtype=self.dtype, device=self.device)
        
        # Store eps for float
        self._eps: float = float(self.cfg.eps)
        
    def position_control(self,
                         pos_w: torch.Tensor,
                         pos_sp_w: torch.Tensor):
        
        pos_err = pos_sp_w - pos_w      # (N, 3)
        vel_sp_w = self._pos_P * pos_err                      
        vel_sp_w = torch.clamp(vel_sp_w, -self._vel_limit, self._vel_limit)
        return vel_sp_w                 # (N, 3)
    
    def velocity_control(self,
                         vel_w: torch.Tensor,
                         vel_sp_w: torch.Tensor):
        
        vel_err = vel_sp_w - vel_w      # (N, 3)
        vel_int_new = self._vel_int + vel_err * self.dt
        
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
        
        # Compute final output
        I_term = self._vel_I * self._vel_int
        I_term = torch.clamp(I_term, -self._vel_I_clamp, self._vel_I_clamp)
        acc_sp_w = P_term + I_term
        acc_sp_w = torch.clamp(acc_sp_w, -self._acc_limit, self._acc_limit)
        
        # Store for logging
        self._acc_sp_w = acc_sp_w
        
        return acc_sp_w      # (N, 3)
        
    def acc_yaw_to_quaternion_thrust(self,
                                   acc_sp_w: torch.Tensor,
                                   yaw_sp: torch.Tensor):
        
        # Get gravity vector of N entities, default is [0,0,-9.81]
        g = ControllerFns.to_tensor_Nx([0, 0, -9.81], self.num_envs ,self.dtype, self.device) # (N, 3)
        
        # Compute thrust vector and b3 axis
        thrust_w = acc_sp_w - g                                                # (N, 3)
        t_norm = torch.norm(thrust_w, dim=1, keepdim=True).clamp_min(float(self._eps))
        b3 = thrust_w / t_norm                                                 # (N, 3)

        # Compute desired heading from yaw setpoint
        if yaw_sp.ndim == 2 and yaw_sp.shape[1] == 1:
            yaw_sp = yaw_sp.squeeze(-1)  # (N,)
        c_yaw = torch.cos(yaw_sp)                                              # (N,)
        s_yaw = torch.sin(yaw_sp)                                              # (N,)
        b1_ref = torch.stack([c_yaw, s_yaw, torch.zeros_like(c_yaw)], dim=1)   # (N, 3)
        
        # Compute b2, b1 axes
        b2 = torch.cross(b3, b1_ref, dim=1)                                    # (N, 3)
        b2 = torch.nn.functional.normalize(b2, dim=1, eps=self.cfg.eps)        # (N, 3)
        b1 = torch.cross(b2, b3, dim=1)                                        # (N, 3)
        
        # Compute rotation matrix and quaternion
        R = torch.stack([b1, b2, b3], dim=2)                                   # (N, 3, 3)
        quat_sp_w = ControllerFns.matrix_to_quaternion(R, self.cfg.eps)        # (N, 4)
        
        # Store for logging 
        self._thrust_w = thrust_w
        self._thrust_norm = t_norm
        self._quat_sp_w = quat_sp_w

        return quat_sp_w, t_norm  # (N, 4), (N, 1)


    def attitude_control(self,
                         quat_w: torch.Tensor,
                         quat_sp_w: torch.Tensor):
        
        # Compute Quaternion error
        quat_w_conj = quat_w * torch.tensor([1, -1, -1, -1], device=self.device, dtype=self.dtype)
        quat_err = ControllerFns.hamilton_product(quat_sp_w, quat_w_conj)  # (N, 4)

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
                          inertia_matrix: torch.Tensor, # (N, 3)
                          ang_vel_b: torch.Tensor,      # (N, 3) [rad/s]
                          ang_vel_sp_b: torch.Tensor):
        
        # Compute attitude error
        ang_vel_err = ang_vel_sp_b - ang_vel_b      
        ang_vel_int_new = self._ang_vel_int + ang_vel_err * self.dt

        # Compute P, I terms
        P_term = self._rate_P * ang_vel_err                  
        I_term = self._rate_I * ang_vel_int_new
        tau_sp_b = P_term + I_term

        # Compute gyro torque
        gyro = torch.cross(ang_vel_b, inertia_matrix * ang_vel_b, dim=1)  # (N, 3)
        torque_sp_b = inertia_matrix * tau_sp_b + gyro  # (N, 3)
        
        # Anti windup: only integrate if not saturated
        torque_sp_lim_b = torch.clamp(torque_sp_b, -self._torque_limit, self._torque_limit)
        saturation = (torque_sp_b > self._torque_limit) | (torque_sp_b < -self._torque_limit)
        same_dir = torch.sign(ang_vel_err) == torch.sign(torque_sp_b)
        reject = saturation & same_dir
        self._ang_vel_int = torch.where(reject, self._ang_vel_int, ang_vel_int_new)

        # Compute final output
        I_term = self._rate_I * self._ang_vel_int
        I_term = torch.clamp(I_term, -self._rate_I_clamp, self._rate_I_clamp)
        tau_sp_b = P_term + I_term
        tau_sp_b = torch.clamp(tau_sp_b, -self._torque_limit, self._torque_limit)
        torque_sp_b = tau_sp_b + gyro
        torque_sp_b = torch.clamp(torque_sp_b, -self._torque_limit, self._torque_limit)

        # Store for logging
        self._torque_sp_b = torque_sp_b
        
        return torque_sp_b # (N, 3)
