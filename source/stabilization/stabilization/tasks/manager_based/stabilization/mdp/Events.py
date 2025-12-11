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
            
        if not wind_enabled:
            scene._wind_vec = torch.zeros((N, 3), device=device, dtype=dtype)
            return
    
        # Wind parameters 
        alpha_dir = 0.05   # Rate of wind direction change
        sigma = 1.0        # standard deviation (if mean is 3, about 99.73% is between 0~6)
        
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
        scene._wind_vec = wind_vec 
        
        
    def apply_wind_effect(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot"),
        wind_enabled: bool = True,
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
        Cd = 1.0          # drag coefficient
        
        # Get boundary and center in quadrotor frame
        aabb_boundary = {  
            "quadrotor_visual": (0.205544, 0.208000, 0.063800),  
            "arm_visual":       (0.227800, 0.028400, 0.028400),  
            "gripper_visual":   (0.060000, 0.040000, 0.022000),  
        }  
        center_pos_q = {  
            "quadrotor_visual": (0.000000, 0.000000, -0.004100),  
            "arm_visual":       (0.091575, 0.000000, -0.050000),  
            "gripper_visual":   (0.214762, 0.000000, -0.050000),  
        }  
        
        # Get body IDs
        ids, names = asset.find_bodies(".*", preserve_order=True)
        quad_index = names.index("quadrotor_visual")
        arm_index = names.index("arm_visual")
        gripper_index = names.index("gripper_visual") 
        
        quad_ids = [int(ids[quad_index])]       
        arm_ids = [int(ids[arm_index])]
        gripper_ids = [int(ids[gripper_index])]
        body_ids = [quad_ids[0], arm_ids[0], gripper_ids[0]]
        
        # Get quaternion, position and linear velocity data of each elements
        body_quat_w    = asset.data.body_quat_w       # (N, B, 4)  
        body_lin_vel_w = asset.data.body_lin_vel_w    # (N, B, 3)
        body_pos_w     = asset.data.body_pos_w        # (N, B, 3)
        
        # Get quaternion of each elements
        quad_quat_w    = body_quat_w[:, quad_index, :]      # (N,4)
        arm_quat_w     = body_quat_w[:, arm_index, :]       # (N,4)
        gripper_quat_w = body_quat_w[:, gripper_index,:]    # (N,4)
        
        # Compute Rotation matrix tensor
        R_wq = math_utils.matrix_from_quat(quad_quat_w)     # (N,3,3)
        R_wa = math_utils.matrix_from_quat(arm_quat_w)      # (N,3,3)
        R_wg = math_utils.matrix_from_quat(gripper_quat_w)  # (N,3,3)
        R_wb = torch.stack([R_wq, R_wa, R_wg], dim=1)       # (N,3,3,3)
        R_bw = R_wb.transpose(-1, -2)                       # (N,3,3,3)
        
        # Get linear velocity of each bodies
        quad_lin_vel_w = body_lin_vel_w[:, quad_index, :]   # (N,3)
        arm_lin_vel_w = body_lin_vel_w[:, arm_index, :]     # (N,3)
        gripper_lin_vel_w = body_lin_vel_w[:, gripper_index, :] # (N,3)
        vb = body_lin_vel_w[:, body_ids, :]  # (N,3,3)

        # wind disabled → zero-out forces/torques and return
        if not wind_enabled:
            forces = torch.zeros((N, 3, 3), device=device, dtype=dtype)
            torques = torch.zeros_like(forces)
            asset.set_external_force_and_torque(
                forces=forces, torques=torques, body_ids=body_ids, is_global=False
            )
            return
        
        # Get relative wind vector
        if not hasattr(scene, "_wind_vec"):  
            scene._wind_vec = torch.zeros((N, 3), device=device, dtype=dtype)  
        wind_w = scene._wind_vec  # (N,3)
        v_rel = wind_w.unsqueeze(1) - vb     # (N,3,3)
        
        # Get relative velocity norm and qdrag
        v_mag = torch.linalg.norm(v_rel, dim=-1).clamp(min=1e-9)   # (N,3)
        v_hat_w = v_rel / v_mag.unsqueeze(-1)                      # (N,3,3)
        q_dyn = 0.5 * rho * (v_mag ** 2)                           # (N,3)
        
        # Get inital aabb boundary tensor
        local_boundary = torch.tensor([
            aabb_boundary["quadrotor_visual"],
            aabb_boundary["arm_visual"],
            aabb_boundary["gripper_visual"]
            ], 
            device=device, dtype=dtype)
        local_boundary = local_boundary.unsqueeze(0).expand(N,-1,-1).contiguous() # (N,3,3)
        half_boundary = local_boundary * 0.5 # (N,3,3)
        
        # Get center position tensor of bodies
        Cq = torch.tensor([
            center_pos_q["quadrotor_visual"],
            center_pos_q["arm_visual"],
            center_pos_q["gripper_visual"]
            ],
            device=device, dtype=dtype)
        Cq = Cq.unsqueeze(0).expand(N,-1,-1).contiguous() # (N,3,3)
        
        # Get center of surfaces of each bodies
        zeros = torch.zeros_like(half_boundary[..., 0])  # (N,3)  
        offset_x_max = torch.stack([half_boundary[..., 0], zeros, zeros], dim=-1)  # (N,3,3)  
        offset_x_min = -offset_x_max                                               # (N,3,3)
        offset_y_max = torch.stack([zeros, half_boundary[..., 1], zeros], dim=-1)  # (N,3,3)
        offset_y_min = -offset_y_max                                               # (N,3,3)
        offset_z_max = torch.stack([zeros, zeros, half_boundary[..., 2]], dim=-1)  # (N,3,3)
        offset_z_min = -offset_z_max                                               # (N,3,3)
        
        # Gather center of surfaces in one tensor
        offset_tot = torch.stack([offset_x_max, offset_x_min, offset_y_max, offset_y_min, offset_z_max, offset_z_min], dim=2)  # (N,3,6,3)  
        
        # Translation
        offset_tot_q = Cq.unsqueeze(2) + offset_tot  # (N,3,6,3)
        
        # Indices of windward faces
        # u_b: relative wind vector in body frame
        # face number: 0:+x, 1:-x,2:+y,3:-y,4:+z,5:-z 
        u_b = torch.einsum("nbij,nbj->nbi", R_bw, v_hat_w)                         # (N,3,3)
        idx_x = torch.where(u_b[..., 0] >= 0,
                            torch.tensor(0, device=device),
                            torch.tensor(1, device=device))                        # (N,3)
        idx_y = torch.where(u_b[..., 1] >= 0,
                            torch.tensor(2, device=device),
                            torch.tensor(3, device=device))                        # (N,3)
        idx_z = torch.where(u_b[..., 2] >= 0,
                            torch.tensor(4, device=device),
                            torch.tensor(5, device=device))                        # (N,3)
        idx_sel = torch.stack([idx_x, idx_y, idx_z], dim=-1)                       # (N,3,3)

        # Gather selected face centers 
        offset_tot_sel = torch.gather(offset_tot_q, 2, idx_sel.unsqueeze(-1).expand(-1, -1, -1, 3))  # (N,3,3,3)

        # Selected face normals in world frame
        normal_x_w = torch.where(u_b[..., 0].unsqueeze(-1) >= 0, R_wb[..., :, 0], -R_wb[..., :, 0])  # (N,3,3)
        normal_y_w = torch.where(u_b[..., 1].unsqueeze(-1) >= 0, R_wb[..., :, 1], -R_wb[..., :, 1])  # (N,3,3)
        normal_z_w = torch.where(u_b[..., 2].unsqueeze(-1) >= 0, R_wb[..., :, 2], -R_wb[..., :, 2])  # (N,3,3)
        normal_w_sel = torch.stack([normal_x_w, normal_y_w, normal_z_w], dim=2)                                      # (N,3,3,3)

        # Face areas per axis and projected reference area A_ref 
        Lx, Ly, Lz = local_boundary[..., 0], local_boundary[..., 1], local_boundary[..., 2]  # (N,3)
        A_x = Ly * Lz                                                                        # (N,3)
        A_y = Lx * Lz                                                                        # (N,3)
        A_z = Lx * Ly                                                                        # (N,3)
        A_axis = torch.stack([A_x, A_y, A_z], dim=2)                                         # (N,3,3)
        A_ref = torch.stack([A_x * u_b.abs()[..., 0],
                            A_y * u_b.abs()[..., 1],
                            A_z * u_b.abs()[..., 2]], dim=2)                                # (N,3,3)

        # Face forces (on 3 selected windward faces per body)
        F_faces = -(Cd * q_dyn).unsqueeze(-1).unsqueeze(-1) * A_ref.unsqueeze(-1) * v_hat_w.unsqueeze(2)  # (N,3,3,3)

        # Lever arm to each selected face center (quad frame → world frame)
        # r_w = R_wq * (face_center in quad frame), quad origin as reference
        r_qw = torch.einsum("nij,nbfj->nbfi", R_wq, offset_tot_sel)          # (N,3,3,3)

        # torque를 각 바디 COM 기준으로 변환: r_body_w = r_qw - (x_body_w - x_quad_w)
        quad_pos_w    = body_pos_w[:, quad_index, :]      # (N,3)
        arm_pos_w     = body_pos_w[:, arm_index, :]       # (N,3)
        gripper_pos_w = body_pos_w[:, gripper_index, :]   # (N,3)
        x_b_w = torch.stack([quad_pos_w, arm_pos_w, gripper_pos_w], dim=1)    # (N,3,3)
        d_b_w = x_b_w - quad_pos_w.unsqueeze(1)                                   # (N,3,3)
        r_w = r_qw - d_b_w.unsqueeze(2)                                           # (N,3,3,3)

        # Face torques about quad origin in world frame
        Tau_faces = torch.cross(r_w, F_faces, dim=-1)                        # (N,3,3,3)

        # Sum forces/torques across the 3 selected faces (x,y,z)
        F_total = F_faces.sum(dim=2)                                        # (N,3,3)
        Tau_total = Tau_faces.sum(dim=2)                                    # (N,3,3)

        F_total_b   = torch.einsum("nbij,nbj->nbi", R_bw, F_total)     # (N,3,3)
        Tau_total_b = torch.einsum("nbij,nbj->nbi", R_bw, Tau_total)   # (N,3,3)

        asset.set_external_force_and_torque(
            forces=F_total_b, torques=Tau_total_b, body_ids=body_ids, is_global=False
        )

        
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
    
    wind_effect_application = EventTerm(
        func=EventFns.apply_wind_effect,
        mode="interval",
        is_global_time=True,
        interval_range_s=(  
            CONFIG["ENV"]["PHYSICS_DT"],
            CONFIG["ENV"]["PHYSICS_DT"],
        ),  
        params={  
            "asset_cfg": SceneEntityCfg(name="Robot"),  
            "wind_enabled": CONFIG["EVENT"]["WIND_ENABLED"],  
        },  
    )

# Alias for easier access
throw_reset = EventFns.throw_reset
wind_generator = EventFns.wind_generator
apply_wind_effect = EventFns.apply_wind_effect
