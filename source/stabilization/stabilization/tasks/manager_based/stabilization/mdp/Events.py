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

import isaacsim.core.utils.bounds as bounds_utils

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
        ang_vel_b = torch.rand((M, 3), dtype=torch.float32, device=device) * (ang_vel_max - ang_vel_min) + ang_vel_min  # (M, 3)  
        omega_norm = torch.linalg.norm(ang_vel_b, dim=1, keepdim=True)  
        scale = torch.clamp(max_omega_norm / (omega_norm + 1e-8), max=1.0)  
        ang_vel_b = ang_vel_b * scale  

        # Rotate body-rate to world-rate using quaternion (v' = v + qw*(2*qv×v) + qv×(2*qv×v))  
        qv = quat_w[:, 1:4]  # (M,3)  
        qw = quat_w[:, 0:1]  # (M,1)  
        t = 2.0 * torch.cross(qv, ang_vel_b, dim=1)  # (M,3)  
        ang_vel_w = ang_vel_b + qw * t + torch.cross(qv, t, dim=1)  # (M,3)  

        # Construct root state
        root_state = torch.zeros((M, 13), dtype=torch.float32, device=device)  # (M, 13)
        root_state[:, 0:3] = pos_w
        root_state[:, 3:7] = quat_w
        root_state[:, 7:10] = lin_vel_w
        root_state[:, 10:13] = ang_vel_w
        
        # Apply to asset
        # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.RigidObject.write_root_link_state_to_sim
        
        asset.write_root_link_state_to_sim(root_state, env_ids=env_ids)
    
    @staticmethod        
    def wind_generator(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot"),
        wind_enabled: bool = True,
        wind_mean_speed: float = 3.0,
    ) -> None:
        
        scene = env.scene
        asset = env.scene[asset_cfg.name]
        device = asset.device
        dtype = torch.float32
        N = env.num_envs
        
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=device)
        else:
            env_ids = env_ids.to(device)
        
        # Wind Parameters
        rho = 1.225       # air density [kg/m^3]        
        sigma = 1         # standard deviation (If mean is 3m/s, About 99.73% is between 0~6m/s)
        alpha_dir = 0.05  # direction change rate
        Cd_quad = 1.0
        Cd_arm = 1.0
        Cd_grip = 1.1
        
        # Get body IDs
        ids, names = asset.find_bodies(".*", preserve_order=True)
        quad_index = names.index("quadrotor_visual")
        arm_index = names.index("arm_visual")
        gripper_index = names.index("gripper_visual") 
        
        quad_ids = [int(ids[quad_index])]       
        arm_ids = [int(ids[arm_index])]
        gripper_ids = [int(ids[gripper_index])]
        body_ids = [quad_ids[0], arm_ids[0], gripper_ids[0]]
        
        # Initialize buffer (for continuous direction) — scene-wide single direction  
        if not hasattr(scene, "_wind_dir_w"):  
            d0 = torch.randn(1, 3, device=device, dtype=dtype)  
            d0 = torch.nn.functional.normalize(d0, dim=1)  
            scene._wind_dir_w = d0.expand(N, 3).clone()  
        
        # Update direction (continuous, same for whole scene)  
        rand = torch.randn(1, 3, device=device, dtype=dtype)  
        rand = torch.nn.functional.normalize(rand, dim=1)  
        u_prev = scene._wind_dir_w[0:1, :]  
        u_new = torch.nn.functional.normalize((1.0 - alpha_dir) * u_prev + alpha_dir * rand, dim=1)  
        scene._wind_dir_w = u_new.expand(N, 3).clone()  
        dir_vec = scene._wind_dir_w  # (N,3)
        
        # Get wind speed based on gaussian distribution (scene-wide single scalar)  
        mu = float(wind_mean_speed)  
        speed = mu + sigma * torch.randn(1, 1, device=device, dtype=dtype)  
        retry = 0  
        while (speed <= 0.0).any() and retry < 5:  
            speed = mu + sigma * torch.randn(1, 1, device=device, dtype=dtype)  
            retry += 1  
        speed = torch.maximum(speed, torch.zeros_like(speed))  
        scene._wind_speed = speed.expand(N, 1).clone()  
        
        wind_vec = dir_vec * scene._wind_speed  
        if not wind_enabled:
            wind_vec.zero_()  
                
        scene._wind_vec = wind_vec 
        
        # Get AABB Boundary
        # https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.core.utils/docs/index.html?utm_source=chatgpt.com#isaacsim.core.utils.bounds.compute_aabb
        
        cache = bounds_utils.create_bbox_cache()
        root_prim_path = getattr(asset, "prim_path", None)
        quad_path = f"{root_prim_path}/{names/[quad_index]}"
        arm_path = f"{root_prim_path/{names/[arm_index]}}"
        gripper_path = f"{root_prim_path/{names/[gripper_index]}}"
        
        def get_aabb_boundary(prim_path: str) -> torch.Tensor:
            aabb = bounds_utils.compute_aabb(cache, prim_path=prim_path)
            Lx = float(aabb[3] - aabb[0])
            Ly = float(aabb[4] - aabb[1])
            Lz = float(aabb[5] - aabb[2])
            Cx = float(0.5 * (aabb[3] + aabb[0]))
            Cy = float(0.5 * (aabb[4] + aabb[1]))
            Cz = float(0.5 * (aabb[5] + aabb[2]))
            L  = torch.tensor([Lx, Ly, Lz], device=device, dtype=dtype)
            C  = torch.tensor([Cx, Cy, Cz], device=device, dtype=dtype)
            return L, C
        
        quad_boundary, qaud_center_w = get_aabb_boundary(quad_path)
        arm_boundary, arm_center_w = get_aabb_boundary(arm_path)
        gripper_boundary, gripper_center_w = get_aabb_boundary(gripper_path)
        
        # Get wind direction vector (Normalized)
        u_global = torch.nn.functional.normalize(wind_vec, dim=1)
        u_abs = u_global.abs()  # (N,3)
        
        # Get reference area
        def get_ref_area(aabb_boundary: torch.Tensor, u_abs: torch.Tensor) -> torch.Tensor:
            Lx, Ly, Lz = aabb_boundary[0], aabb_boundary[1], aabb_boundary[2]
            Axy = Lx * Ly
            Azx = Lx * Lz
            Ayz = Ly * Lz
            Ax = Ayz * u_abs[:, 0:1]
            Ay = Azx * u_abs[:, 1:2]
            Az = Axy * u_abs[:, 2:3]
            return torch.cat([Ax, Ay, Az], dim=1) # (N,3)
        
        quad_ref_area = get_ref_area(quad_boundary, u_abs) # (N,3)
        arm_ref_area = get_ref_area(arm_boundary, u_abs) # (N,3)
        gripper_ref_area = get_ref_area(gripper_boundary, u_abs) # (N,3)
        
        # Get quaternion and linear velocity data of each elements
        body_quat_w    = asset.data.body_quat_w       # (N, B, 4)  
        body_lin_w     = asset.data.body_lin_w        # (N, B, 4)
        
        # Get quaternion of each elements
        quad_quat_w    = body_quat_w[:, quad_index, :]      # (N,4)
        arm_quat_w     = body_quat_w[:, arm_index, :]       # (N,4)
        gripper_quat_w = body_quat_w[:, gripper_index,:]    # (N,4)
        
        # Compute Rotation matrix
        R_wq = math_utils.matrix_from_quat(quad_quat_w)     # (N,3,3)
        R_wa = math_utils.matrix_from_quat(arm_quat_w)      # (N,3,3)
        R_wg = math_utils.matrix_from_quat(gripper_quat_w)  # (N,3,3)

        R_qw = R_wq.transpose(1, 2)                         
        R_aw = R_wa.transpose(1, 2)                         

        R_qa = torch.bmm(R_qw, R_wa)            # arm from quad
        R_ag = torch.bmm(R_aw, R_wg)            # gripper from arm
        R_qg = torch.bmm(R_qw, R_wg)            # gripper from quad
        
        def get_force_and_torque(
            aabb_boundary: torch.Tensor, 
            center_w: torch.Tensor,
            ref_area: torch.Tensor, 
            Cd: float, 
            R_wb: torch.Tensor, 
            idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            
            L = aabb_boundary.view(1,3) # (1,3)
            h = 0.5 * L
            hxN = h[:,]

        
    
    
        
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
    
    wind_generation = EventTerm(
        func=EventFns.wind_generator,
        mode="interval",
        is_global_time=True,
        interval_range_s=(  
            1.0 / CONFIG["EVENT"]["WIND_UPDATE_HZ"],  
            1.0 / CONFIG["EVENT"]["WIND_UPDATE_HZ"],  
        ),  
        params={  
            "asset_cfg": SceneEntityCfg(name="Robot"),  
            "wind_enabled": CONFIG["EVENT"]["WIND_ENABLED"],  
            "wind_mean_speed": CONFIG["EVENT"]["WIND_MEAN_SPEED"],  
        },  
    )  


# Alias for easier access
throw_reset = EventFns.throw_reset
wind_generator = EventFns.wind_generator
