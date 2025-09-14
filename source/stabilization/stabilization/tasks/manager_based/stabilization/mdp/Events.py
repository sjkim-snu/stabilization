from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass
from typing import Tuple

import math
import torch
import isaaclab.utils.math as math_utils
import stabilization.tasks.manager_based.stabilization.mdp as mdp
from stabilization.tasks.manager_based.stabilization.config import load_parameters

# Load configuration from YAML file
CONFIG = load_parameters()

class EventFns:
    
    @staticmethod
    def throw_reset(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        lin_vel_range: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        ang_vel_range: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        tilt_range_deg: Tuple[float, float],
        yaw_range: Tuple[float, float],
        max_omega_norm: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot")
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
        device = asset.device
        
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=device)
        else:
            env_ids = env_ids.to(device)
        
        M = env_ids.shape[0]
        pos_all = mdp.ObservationFns.get_spawn_pos_w(env, asset_cfg).to(device)  # (N, 3)
        pos_w = pos_all[env_ids]  # (M, 3)
        
        # Randomize orientation
        roll = math_utils.sample_uniform(tilt_range_deg[0], tilt_range_deg[1], M, device=device) # (M,)
        pitch = math_utils.sample_uniform(tilt_range_deg[0], tilt_range_deg[1], M, device=device) # (M,)
        yaw = math_utils.sample_uniform(yaw_range[0], yaw_range[1], M, device=device) # (M,)
        quat_w = math_utils.quat_from_euler_xyz(roll, pitch, yaw) # (M, 4)

        # Randomize linear velocity
        lin_vel_min = torch.tensor(lin_vel_range[0], dtype=torch.float32, device=device)  # (3,)
        lin_vel_max = torch.tensor(lin_vel_range[1], dtype=torch.float32, device=device)  # (3,)
        
        # To avoid very small velocities, sample magnitude and sign separately
        mag  = torch.rand((M, 3), dtype=torch.float32, device=device) * (lin_vel_max - lin_vel_min) + lin_vel_min  # (M,3)
        sign = torch.where(
            torch.rand((M, 3), device=device) < 0.5,
            -torch.ones((M, 3), dtype=torch.float32, device=device),
            torch.ones((M, 3), dtype=torch.float32, device=device),
        )  # (M,3)

        lin_vel_w = sign * mag  # (M,3) in (-max, -min) ∪ (min, max)
        lin_vel_w[:, 2] = lin_vel_w[:, 2].abs()  # ensure positive z velocity
        
        # Randomize angular velocity
        ang_vel_min = torch.tensor(ang_vel_range[0], dtype=torch.float32, device=device)  # (3,)
        ang_vel_max = torch.tensor(ang_vel_range[1], dtype=torch.float32, device=device)  # (3,)
        ang_vel_b = torch.rand((M, 3), dtype=torch.float32, device=device) * (ang_vel_max - ang_vel_min) + ang_vel_min  # (M, 3)  # 추가 (+)
        omega_norm = torch.linalg.norm(ang_vel_b, dim=1, keepdim=True)  # 추가 (+)
        scale = torch.clamp(max_omega_norm / (omega_norm + 1e-8), max=1.0)  # 추가 (+)
        ang_vel_b = ang_vel_b * scale  # 추가 (+)

        # Rotate body-rate to world-rate using quaternion (v' = v + qw*(2*qv×v) + qv×(2*qv×v))  # 추가 (+)
        qv = quat_w[:, 1:4]  # (M,3)  # 추가 (+)
        qw = quat_w[:, 0:1]  # (M,1)  # 추가 (+)
        t = 2.0 * torch.cross(qv, ang_vel_b, dim=1)  # (M,3)  # 추가 (+)
        ang_vel_w = ang_vel_b + qw * t + torch.cross(qv, t, dim=1)  # (M,3)  # 추가 (+)

        # Construct root state
        root_state = torch.zeros((M, 13), dtype=torch.float32, device=device)  # (M, 13)
        root_state[:, 0:3] = pos_w
        root_state[:, 3:7] = quat_w
        root_state[:, 7:10] = lin_vel_w
        root_state[:, 10:13] = ang_vel_w
        
        # Apply to asset
        # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.RigidObject.write_root_link_state_to_sim
        
        asset.write_root_link_state_to_sim(root_state, env_ids=env_ids)
        
@configclass
class EventCfg:

    # https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/managers/event_manager.html#EventManager
    
    throw_on_reset = EventTerm(
        func=EventFns.throw_reset,
        mode="reset",
        is_global_time=False,
        params={
            "asset_cfg": SceneEntityCfg(name="Robot"),
            "lin_vel_range": (
                CONFIG["EVENT"]["LIN_VEL_MIN"], 
                CONFIG["EVENT"]["LIN_VEL_MAX"]
                ),
            "ang_vel_range": (
                CONFIG["EVENT"]["ANG_VEL_MIN"], 
                CONFIG["EVENT"]["ANG_VEL_MAX"]
                ),
            "tilt_range_deg": (math.radians(CONFIG["EVENT"]["TILT_RANGE_DEG"][0]), 
                               math.radians(CONFIG["EVENT"]["TILT_RANGE_DEG"][1])),
            "yaw_range": (math.radians(CONFIG["EVENT"]["YAW_DEGREE_RANGE"][0]), 
                          math.radians(CONFIG["EVENT"]["YAW_DEGREE_RANGE"][1])),
            "max_omega_norm": CONFIG["EVENT"]["OMEGA_NORM_MAX"],
        },
    )
