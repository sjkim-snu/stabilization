from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass
from typing import Tuple

import math
import torch
import isaaclab.utils.math as math_utils
import stabilization.tasks.manager_based.stabilization.mdp as mdp

class EventFns:
    
    @staticmethod
    def throw_reset(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot"),
        lin_vel_range: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
            (-2.0, -2.0, 1.0),   # (vx_min, vy_min, vz_min) 
            (2.0, 2.0, 4.0)      # (vx_max, vy_max, vz_max)
        ),                       # m/s
        ang_vel_range: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
            (-4.0, -4.0, -4.0),  # (wx_min, wy_min, wz_min)
            (4.0, 4.0, 4.0)      # (wx_max, wy_max, wz_max)
        ),                       # rad/s     
        max_tilt_rad: float = math.pi / 3.0,   # 60 degrees
        yaw_range: Tuple[float, float] = (-math.pi, math.pi),  # radians
        max_omega_norm: float = 7.0,  # rad/s
    ) -> None:
        
        """
        Event function to reset the quadrotor by throwing it into the air with random
        linear and angular velocities, and random orientation within specified limits.
        
        Args:
            env (ManagerBasedEnv): The environment instance.
            asset_cfg (SceneEntityCfg): Name of the quadrotor entity.
            lin_vel_range: Range for linear velocity (vx, vy, vz) in m/s.
            ang_vel_range: Range for angular velocity (wx, wy, wz) in rad/s.
            max_tilt_rad: Maximum tilt angle from vertical in radians.
            yaw_range: Range for yaw angle in radians.
            max_omega_norm: Maximum norm for angular velocity in rad/s.
        """
        
        asset = env.scene[asset_cfg.name]
        N = env.num_envs
        device = asset.device
        
        # Fied initial position
        pos_w = mdp.ObservationFns.spawn_position_w(env, asset_cfg.name).to(device)  # (N, 3)
        
        # Randomize orientation
        roll = math_utils.sample_uniform(-max_tilt_rad, max_tilt_rad, N, device=device) # (N,)
        pitch = math_utils.sample_uniform(-max_tilt_rad, max_tilt_rad, N, device=device) # (N,)
        yaw = math_utils.sample_uniform(yaw_range[0], yaw_range[1], N, device=device) # (N,)
        quat_w = math_utils.quat_from_euler_xyz(roll, pitch, yaw) # (N, 4)
        
        # Randomize linear velocity
        lin_vel_min = torch.tensor(lin_vel_range[0], dtype=torch.float32, device=device)  # (3,)
        lin_vel_max = torch.tensor(lin_vel_range[1], dtype=torch.float32, device=device)  # (3,)
        lin_vel_w = torch.rand((N, 3), dtype=torch.float32, device=device) * (lin_vel_max - lin_vel_min) + lin_vel_min  # (N, 3)

        # Randomize angular velocity
        ang_vel_min = torch.tensor(ang_vel_range[0], dtype=torch.float32, device=device)  # (3,)
        ang_vel_max = torch.tensor(ang_vel_range[1], dtype=torch.float32, device=device)  # (3,)
        ang_vel_w = torch.rand((N, 3), dtype=torch.float32, device=device) * (ang_vel_max - ang_vel_min) + ang_vel_min  # (N, 3)

        # Construct root state
        root_state = torch.zeros((N, 13), dtype=torch.float32, device=device)  # (N, 13)
        root_state[:, 0:3] = pos_w
        root_state[:, 3:7] = quat_w
        root_state[:, 7:10] = lin_vel_w
        root_state[:, 10:13] = ang_vel_w
        
        # Apply to asset
        # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.RigidObject.write_root_link_state_to_sim
        asset.write_root_link_state_to_sim(root_state)
        
@configclass
class EventCfg:

    # https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/managers/event_manager.html#EventManager
    
    throw_on_reset = EventTerm(
        func=EventFns.throw_reset,
        mode="reset",
        is_global_time=False,
        params={
            "asset_cfg": SceneEntityCfg(name="Robot"),
            "lin_vel_range": ((-2.0, -2.0, 1.0), (2.0, 2.0, 4.0)),
            "ang_vel_range": ((-4.0, -4.0, -4.0), (4.0, 4.0, 4.0)),
            "max_tilt_rad": math.pi / 3.0,
            "yaw_range": (-math.pi, math.pi),
            "max_omega_norm": 7.0,},
    )