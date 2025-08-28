from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs.mdp.terminations import time_out
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass
from typing import Tuple
from stabilization.tasks.manager_based.stabilization.config import load_parameters

import math
import torch
import stabilization.tasks.manager_based.stabilization.mdp as mdp

# Load configuration from YAML file
CONFIG = load_parameters()

"""
Helper functions
"""

def _is_stable_from_obs(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    pos_tol_m: float,
    lin_vel_tol_mps: float,
    ang_vel_tol_radps: float,
    tilt_tol_rad: float,
) -> torch.Tensor:
    
    """
    Determine if the quadrotor is stable based on observation functions.
    
        Args:
            env (ManagerBasedEnv): The environment instance.
            asset_cfg (SceneEntityCfg): Name of the quadrotor entity.
            pos_tol_m (float): Position error tolerance in meters.
            lin_vel_tol_mps (float): Linear velocity tolerance in m/s.
            ang_vel_tol_radps (float): Angular velocity tolerance in rad/s.
            tilt_tol_rad (float): Tilt tolerance in radians.
        Returns:
            stable: Tensor of shape (N,) representing stability (True if stable).
    """
    
    # determine whether position error is within tolerance
    pos_err_w = mdp.ObservationFns.position_error_w(env, asset_cfg)  # (N, 3)
    within_pos = torch.linalg.norm(pos_err_w, dim=-1) <= pos_tol_m   # (N,)

    # determine whether linear velocity is within tolerance
    lin_vel_b = mdp.ObservationFns.lin_vel_body(env, asset_cfg)      # (N, 3)
    within_lin = torch.linalg.norm(lin_vel_b, dim=-1) <= lin_vel_tol_mps # (N,)

    # determine whether angular velocity is within tolerance
    ang_vel_b = mdp.ObservationFns.ang_vel_body(env, asset_cfg)      # (N, 3)
    within_ang = torch.linalg.norm(ang_vel_b, dim=-1) <= ang_vel_tol_radps # (N,)

    # determine whether roll is within tolerance
    roll  = mdp.ObservationFns.roll_current(env, asset_cfg).squeeze(-1)    # (N,)
    within_roll = torch.abs(roll) <= tilt_tol_rad                          # (N,)
    
    # determine whether pitch is within tolerance
    pitch = mdp.ObservationFns.pitch_current(env, asset_cfg).squeeze(-1)   # (N,)
    within_pitch = torch.abs(pitch) <= tilt_tol_rad                        # (N,)

    return (within_pos & within_lin & within_ang & within_roll & within_pitch)   # (N,)



"""
Termination functions based on observations
"""

class TerminationFns:

    @staticmethod
    def is_flipped(
        env: ManagerBasedEnv,
        tilt_threshold_rad: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot"),
    ) -> torch.Tensor:
        
        """
        determine if the quadrotor has flipped over based on tilt angle.
        Args:
            env (ManagerBasedEnv): The environment instance.
            asset_cfg (SceneEntityCfg): Name of the quadrotor entity.
            tilt_threshold_rad (float): Tilt angle threshold in radians.
        Returns:
            Tensor (N,) representing whether the quadrotor has flipped (True if flipped)
        """
        roll  = mdp.ObservationFns.roll_current(env, asset_cfg).squeeze(-1)   # (N,)
        pitch = mdp.ObservationFns.pitch_current(env, asset_cfg).squeeze(-1)  # (N,)
        roll_within = torch.abs(roll)   <= tilt_threshold_rad  # (N,)
        pitch_within = torch.abs(pitch) <= tilt_threshold_rad  # (N,)
        return (roll_within & pitch_within)                    # (N,)


    @staticmethod
    def is_far_from_spawn(
        env: ManagerBasedEnv,
        dist_threshold_m: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")
    ) -> torch.Tensor:
        
        """
        Determine if the quadrotor is far from its spawn position.
        
        Args:
            env (ManagerBasedEnv): The environment instance.
            asset_cfg (SceneEntityCfg): Name of the quadrotor entity.
            dist_threshold_m (float): Distance threshold in meters.
        Returns:
            Tensor (N,) representing whether the quadrotor is far from spawn (True if far)
        """
        
        pos_err_w = mdp.ObservationFns.position_error_w(env, asset_cfg)  # (N, 3)
        dist = torch.linalg.norm(pos_err_w, dim=-1)                      # (N,)
        return (dist > dist_threshold_m)                                 # (N,)


    @staticmethod
    def is_crashed(
        env: ManagerBasedEnv,
        z_min_m: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot"),
    ) -> torch.Tensor:
        
        """
        Determine if the quadrotor has crashed (i.e., is too close to the ground).
        """
        
        asset = env.scene[asset_cfg.name]
        position_w = asset.data.root_pos_w            # (N, 3)
        z_current = torch.abs(position_w[:, 2])       # (N,)
        z_within = z_current <= z_min_m               # (N,)
        return (z_within)                             # (N,)


    @staticmethod
    def is_stabilized(
        env: ManagerBasedEnv,
        pos_tol_m: float,
        lin_vel_tol_mps: float,
        ang_vel_tol_radps: float,
        tilt_tol_rad: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot"),
    ) -> torch.Tensor:
        
        """
        Determine if the quadrotor is stabilized based on observation functions.
        """
        
        return _is_stable_from_obs(
            env, asset_cfg, pos_tol_m, lin_vel_tol_mps, ang_vel_tol_radps, tilt_tol_rad
        )  # (N,)


    @staticmethod
    def is_nan_or_inf(
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot"),
    ) -> torch.Tensor:
        
        """
        Determine if any of the key state observations contain NaN or Inf values.
        1) roll, pitch, yaw
        2) position error (x, y, z)
        3) linear velocity (vx, vy, vz)
        4) angular velocity (wx, wy, wz)
        """
        
        # If inf or nan in any of the following observations, return True
        roll  = mdp.ObservationFns.roll_current(env, asset_cfg).squeeze(-1)    # (N,)
        pitch = mdp.ObservationFns.pitch_current(env, asset_cfg).squeeze(-1)   # (N,)
        yaw   = mdp.ObservationFns.yaw_current(env, asset_cfg).squeeze(-1)     # (N,)
        pos_err_w = mdp.ObservationFns.position_error_w(env, asset_cfg)        # (N, 3)
        lin_vel_b = mdp.ObservationFns.lin_vel_body(env, asset_cfg)            # (N, 3)
        ang_vel_b = mdp.ObservationFns.ang_vel_body(env, asset_cfg)            # (N, 3)

        scalars = torch.stack([roll, pitch, yaw], dim=-1)                       # (N, 3)
        mats = torch.cat([pos_err_w, lin_vel_b, ang_vel_b], dim=-1)             # (N, 9)

        bad = (
            torch.isnan(scalars).any(dim=-1)
            | torch.isinf(scalars).any(dim=-1)
            | torch.isnan(mats).any(dim=-1)
            | torch.isinf(mats).any(dim=-1)
        )  # (N,)
        return bad


@configclass
class TerminationsCfg:
    
    """
    Manager 등록용 종료 조건 모음.
    모든 판정은 ObservationFns를 통해 산출된 값(=관측)을 기준으로 수행됩니다.
    """

    flipped = DoneTerm(
        func=TerminationFns.is_flipped,
        params={
            "asset_cfg": SceneEntityCfg(name="Robot"), 
            "tilt_threshold_rad": math.radians(CONFIG["TERMINATION"]["FLIP_TILT_DEGREE"])},
        time_out=False,
    )

    far_from_spawn = DoneTerm(
        func=TerminationFns.is_far_from_spawn,
        params={
            "asset_cfg": SceneEntityCfg(name="Robot"), 
            "dist_threshold_m": CONFIG["TERMINATION"]["DIST_THRESHOLD"],
        },
        time_out=False,
    )

    crashed = DoneTerm(
        func=TerminationFns.is_crashed,
        params={
            "asset_cfg": SceneEntityCfg(name="Robot"), 
            "z_min_m": CONFIG["TERMINATION"]["ALTITUDE_THRESHOLD"],
        },
        time_out=False,
    )

    nan_or_inf = DoneTerm(
        func=TerminationFns.is_nan_or_inf,
        params={"asset_cfg": SceneEntityCfg(name="Robot")},
        time_out=False,
    )

    stabilized = DoneTerm(
        func=TerminationFns.is_stabilized,
        params={
            "asset_cfg": SceneEntityCfg(name="Robot"),
            "pos_tol_m": CONFIG["STABILIZATION"]["POS_ERR_TOL"],
            "lin_vel_tol_mps": CONFIG["STABILIZATION"]["LIN_VEL_TOL"],
            "ang_vel_tol_radps": CONFIG["STABILIZATION"]["ANG_VEL_TOL"],
            "tilt_tol_rad": math.radians(CONFIG["STABILIZATION"]["TILT_DEGREE_TOL"]),
        },
        time_out=False,
    )

    time_out = DoneTerm(
        func=time_out,
        params={},
        time_out=True,
    )
