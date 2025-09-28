from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.assets import AssetBase, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from dataclasses import dataclass
from isaaclab.utils import configclass
from abc import abstractmethod

import stabilization.tasks.manager_based.stabilization.mdp as mdp
import stabilization.tasks.manager_based.stabilization.envs as envs
import isaaclab.envs as env
import torch
import math
from stabilization.tasks.manager_based.stabilization.config import load_parameters
import isaaclab.utils.math as math_utils  

# Load configuration from YAML file
CONFIG = load_parameters()

class ManipulatorActionFns:
    
    @staticmethod
    def mix_rate_thrust_to_rotor_forces(
        required_accel: torch.Tensor,    # (N, 1)
        mass: torch.Tensor,              # (N, 1)
        momentum_sp_b: torch.Tensor,     # (N, 3)
        rotor_xy: torch.Tensor,          # (4, 2)
        rotor_dirs: torch.Tensor,        # (4,)
        k_f_rpm2: torch.Tensor,          # (1,)
        k_m_rpm2: torch.Tensor,          # (1,)
        com_pos_q: torch.Tensor,         # (N, 3)  
        ) -> torch.Tensor:
        
        # Handle device and dtype
        device = required_accel.device
        dtype = required_accel.dtype
        N = required_accel.shape[0]
        
        # Set tensors
        momentum_sp_b = momentum_sp_b.to(device=device, dtype=dtype) # (N, 3)
        mass = mass.to(device=device, dtype=dtype)                   # (N, 1)
        x = rotor_xy[:, 0].to(device=device, dtype=dtype)            # (4,)
        y = rotor_xy[:, 1].to(device=device, dtype=dtype)            # (4,)
        rotor_dirs = rotor_dirs.to(device=device, dtype=dtype)       # (4,)
        k_m_rpm2 = k_m_rpm2.to(device=device, dtype=dtype)           # (1,)
        k_f_rpm2 = k_f_rpm2.to(device=device, dtype=dtype)           # (1,)
        c = rotor_dirs * (k_m_rpm2 / k_f_rpm2)                       # (4,)

        # Apply COM of total system
        rC = com_pos_q.to(device=device, dtype=dtype)                # (N,3) 
        xC = rC[:, 0:1]                                              # (N,1) 
        yC = rC[:, 1:2]                                              # (N,1) 
        x_e = x.view(1, 4).expand(N, 4)                                 
        y_e = y.view(1, 4).expand(N, 4)                                 
        c_b = c.view(1, 4).expand(N, 4)
        M = torch.stack([torch.ones_like(x_e), y_e, -x_e, c_b], dim=1)  

        # Mixing matrix 
        M = torch.stack([torch.ones_like(x_e), y_e, -x_e, c_b], dim=1)  

        # Desired thrust, torque vector
        total_thrust = (required_accel * mass).squeeze(-1)            # (N,)
        Tx = momentum_sp_b[:, 0] + (yC.squeeze(-1) * total_thrust)    # (N,) 
        Ty = momentum_sp_b[:, 1] - (xC.squeeze(-1) * total_thrust)    # (N,) 
        Tz = momentum_sp_b[:, 2]                                      # (N,) 
        T = torch.stack([total_thrust, Tx, Ty, Tz], dim=1)            # (N,4) 

        # Solve equations
        rotor_forces = torch.linalg.solve(M, T.unsqueeze(-1)).squeeze(-1)  # (N,4) 
        
        return rotor_forces
    
    @staticmethod
    def desaturate_rotor_forces(
        rotor_forces: torch.Tensor,  # (N,4)
        k_f_rpm2: torch.Tensor,      # (1,)
        w_min_rpm: torch.Tensor,     # (1,)            
        w_max_rpm: torch.Tensor,     # (1,)
        ) -> torch.Tensor:

        device = rotor_forces.device
        dtype = rotor_forces.dtype

        # Set tensors
        kf = k_f_rpm2.view(1, 1).to(device=device, dtype=dtype)       # (1,1)
        rpm2_signed = rotor_forces / kf                               # (N,4)
        
        # Compute signed rpm
        rpm_abs = torch.sqrt(torch.clamp(torch.abs(rpm2_signed), min=0.0))  # (N,4)
        rpm_sign = torch.sign(rpm2_signed)                                  # (N,4)
        rpm_cmd = rpm_sign * rpm_abs                                        # (N,4) 

        # Rename for clarity
        wmin = w_min_rpm
        wmax = w_max_rpm
        wmin2 = wmin * wmin
        wmax2 = wmax * wmax

        rpm_out = rpm_cmd.clone()

        # 1st pair: motor 0 and 2
        i, j = 0, 2
        wi = rpm_out[:, i]
        wj = rpm_out[:, j]

        # Sum of squares (represents thrust)
        Si = torch.sign(wi) * wi * wi + torch.sign(wj) * wj * wj  # (N,)

        # Determine in/out of bounds
        in_i = (wi >= wmin) & (wi <= wmax)
        in_j = (wj >= wmin) & (wj <= wmax)

        # If both in range, keep it
        keep_mask = in_i & in_j

        # If only one is out of range
        only_i_out = (~in_i) & in_j
        only_j_out = in_i & (~in_j)

        # If only i is out, clamp i and recompute j to preserve S
        wi_clamped = torch.where(wi < wmin, wmin, torch.where(wi > wmax, wmax, wi))
        wj2_target = Si - wi_clamped * wi_clamped
        wj_new = torch.sqrt(torch.clamp(wj2_target, min=wmin2, max=wmax2))
        rpm_out[:, i] = torch.where(only_i_out, wi_clamped, rpm_out[:, i])
        rpm_out[:, j] = torch.where(only_i_out, wj_new,     rpm_out[:, j])

        # If only j is out, clamp j and recompute i to preserve S
        wj_clamped = torch.where(wj < wmin, wmin, torch.where(wj > wmax, wmax, wj))
        wi2_target = Si - wj_clamped * wj_clamped
        wi_new = torch.sqrt(torch.clamp(wi2_target, min=wmin2, max=wmax2))
        rpm_out[:, j] = torch.where(only_j_out, wj_clamped, rpm_out[:, j])
        rpm_out[:, i] = torch.where(only_j_out, wi_new,     rpm_out[:, i])

        # If both are out of range
        both_out = (~in_i) & (~in_j)
        
        if both_out.any():
            
            # Branch A: If one is over max and the other under min
            over_i_under_j = (wi > wmax) & (wj < wmin)
            over_j_under_i = (wj > wmax) & (wi < wmin)
            
            # if i is over and j is under
            wi_first = torch.clamp(wi, max=wmax)
            wj2_t   = Si - wi_first * wi_first
            wj_first = torch.sqrt(torch.clamp(wj2_t, min=wmin2, max=wmax2))
            rpm_out[:, i] = torch.where(both_out & over_i_under_j, wi_first, rpm_out[:, i])
            rpm_out[:, j] = torch.where(both_out & over_i_under_j, wj_first, rpm_out[:, j])

            # if j is over and i is under
            wj_first = torch.clamp(wj, max=wmax)
            wi2_t   = Si - wj_first * wj_first
            wi_first = torch.sqrt(torch.clamp(wi2_t, min=wmin2, max=wmax2))
            rpm_out[:, j] = torch.where(both_out & over_j_under_i, wj_first, rpm_out[:, j])
            rpm_out[:, i] = torch.where(both_out & over_j_under_i, wi_first, rpm_out[:, i])

            # Branch B: both over or both under
            remains = both_out & (~over_i_under_j) & (~over_j_under_i)
            
            if remains.any():
                
                # Choose which to clamp first based on which is further out of bounds 
                dist_i = torch.where(wi > wmax, wi - wmax, torch.where(wi < wmin, wmin - wi, torch.zeros_like(wi)))
                dist_j = torch.where(wj > wmax, wj - wmax, torch.where(wj < wmin, wmin - wj, torch.zeros_like(wj)))
                choose_i = dist_i >= dist_j

                # If i is selected (more out of bounds)
                wi_c = torch.where(wi > wmax, wmax, torch.where(wi < wmin, wmin, wi))
                wj2  = Si - wi_c * wi_c
                wj_c = torch.sqrt(torch.clamp(wj2, min=wmin2, max=wmax2))
                rpm_out[:, i] = torch.where(remains & choose_i, wi_c, rpm_out[:, i])
                rpm_out[:, j] = torch.where(remains & choose_i, wj_c, rpm_out[:, j])

                # If j is selected (more out of bounds)
                wj_c = torch.where(wj > wmax, wmax, torch.where(wj < wmin, wmin, wj))
                wi2  = Si - wj_c * wj_c
                wi_c = torch.sqrt(torch.clamp(wi2, min=wmin2, max=wmax2))
                rpm_out[:, j] = torch.where(remains & (~choose_i), wj_c, rpm_out[:, j])
                rpm_out[:, i] = torch.where(remains & (~choose_i), wi_c, rpm_out[:, i])

        # 2nd pair: motor 1 and 3
        i, j = 1, 3
        wi = rpm_out[:, i]
        wj = rpm_out[:, j]
        Si = torch.sign(wi) * wi * wi + torch.sign(wj) * wj * wj

        in_i = (wi >= wmin) & (wi <= wmax)
        in_j = (wj >= wmin) & (wj <= wmax)

        keep_mask = in_i & in_j

        only_i_out = (~in_i) & in_j
        only_j_out = in_i & (~in_j)

        wi_clamped = torch.where(wi < wmin, wmin, torch.where(wi > wmax, wmax, wi))
        wj2_target = Si - wi_clamped * wi_clamped
        wj_new = torch.sqrt(torch.clamp(wj2_target, min=wmin2, max=wmax2))
        rpm_out[:, i] = torch.where(only_i_out, wi_clamped, rpm_out[:, i])
        rpm_out[:, j] = torch.where(only_i_out, wj_new,     rpm_out[:, j])

        wj_clamped = torch.where(wj < wmin, wmin, torch.where(wj > wmax, wmax, wj))
        wi2_target = Si - wj_clamped * wj_clamped
        wi_new = torch.sqrt(torch.clamp(wi2_target, min=wmin2, max=wmax2))
        rpm_out[:, j] = torch.where(only_j_out, wj_clamped, rpm_out[:, j])
        rpm_out[:, i] = torch.where(only_j_out, wi_new,     rpm_out[:, i])

        both_out = (~in_i) & (~in_j)
        if both_out.any():
            over_i_under_j = (wi > wmax) & (wj < wmin)
            over_j_under_i = (wj > wmax) & (wi < wmin)

            wi_first = torch.clamp(wi, max=wmax)
            wj2_t   = Si - wi_first * wi_first
            wj_first = torch.sqrt(torch.clamp(wj2_t, min=wmin2, max=wmax2))
            rpm_out[:, i] = torch.where(both_out & over_i_under_j, wi_first, rpm_out[:, i])
            rpm_out[:, j] = torch.where(both_out & over_i_under_j, wj_first, rpm_out[:, j])

            wj_first = torch.clamp(wj, max=wmax)
            wi2_t   = Si - wj_first * wj_first
            wi_first = torch.sqrt(torch.clamp(wi2_t, min=wmin2, max=wmax2))
            rpm_out[:, j] = torch.where(both_out & over_j_under_i, wj_first, rpm_out[:, j])
            rpm_out[:, i] = torch.where(both_out & over_j_under_i, wi_first, rpm_out[:, i])

            remains = both_out & (~over_i_under_j) & (~over_j_under_i)
            if remains.any():
                dist_i = torch.where(wi > wmax, wi - wmax, torch.where(wi < wmin, wmin - wi, torch.zeros_like(wi)))
                dist_j = torch.where(wj > wmax, wj - wmax, torch.where(wj < wmin, wmin - wj, torch.zeros_like(wj)))
                choose_i = dist_i >= dist_j

                wi_c = torch.where(wi > wmax, wmax, torch.where(wi < wmin, wmin, wi))
                wj2  = Si - wi_c * wi_c
                wj_c = torch.sqrt(torch.clamp(wj2, min=wmin2, max=wmax2))
                rpm_out[:, i] = torch.where(remains & choose_i, wi_c, rpm_out[:, i])
                rpm_out[:, j] = torch.where(remains & choose_i, wj_c, rpm_out[:, j])

                wj_c = torch.where(wj > wmax, wmax, torch.where(wj < wmin, wmin, wj))
                wi2  = Si - wj_c * wj_c
                wi_c = torch.sqrt(torch.clamp(wi2, min=wmin2, max=wmax2))
                rpm_out[:, j] = torch.where(remains & (~choose_i), wj_c, rpm_out[:, j])
                rpm_out[:, i] = torch.where(remains & (~choose_i), wi_c, rpm_out[:, i])

        # Convert back to forces
        rotor_forces_out = torch.clamp((rpm_out * rpm_out) * kf, min=0.0)
        
        return rotor_forces_out
        
        
    @staticmethod
    def thrust_to_rpm(
        rotor_forces: torch.Tensor, # (N, 4)
        w_min_rpm: torch.Tensor,    # (1,)
        w_max_rpm: torch.Tensor,    # (1,)
        k_f: torch.Tensor,          # (1,)
        ) -> torch.Tensor:

        # Handle device, dtype, and dimensions
        device = rotor_forces.device
        dtype = rotor_forces.dtype
        N = rotor_forces.shape[0]
        
        # Set tensors
        w_min_rpm = w_min_rpm.view(1, 1).expand(N, 1).to(device=device, dtype=dtype) # (N, 1)
        w_max_rpm = w_max_rpm.view(1, 1).expand(N, 1).to(device=device, dtype=dtype) # (N, 1)
        k_f = k_f.view(1, 1).expand(N, 1).to(device=device, dtype=dtype)             # (N, 1)

        # Convert RPM to rad/s
        w_min = w_min_rpm * (2 * math.pi / 60) # (N, 1)
        w_max = w_max_rpm * (2 * math.pi / 60) # (N, 1)
        
        # Compute rotor speeds
        w = torch.sqrt(rotor_forces / k_f)       # (N, 4)
        w = torch.clamp(w, min=w_min, max=w_max) # (N, 4)
        
        # Convert rad/s to RPM
        w_rpm = w * (60 / (2 * math.pi)) # (N, 4)
        
        return w_rpm # (N, 4)
    
    @staticmethod
    def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:

        # Quaternion format
        w = quat[:, 0]; x = quat[:, 1]; y = quat[:, 2]; z = quat[:, 3]

        # Yaw extraction
        t0 = 2.0 * (w * z + x * y)
        t1 = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(t0, t1)
        
        return yaw.view(-1, 1)
    
    
@configclass
class ManipulatorBaseControllerCfg(ActionTermCfg):
        
    asset_name: str = "Robot"
    arm_length: float = 1
    
    k_f_rpm2: float = 9.6e-09
    k_m_rpm2: float = 6.8e-11
    
    w_min_rpm: float = 5000.0
    w_max_rpm: float = 25000.0
    
    rotor_dirs: list[float] = [1.0, -1.0, 1.0, -1.0] # CW: +1, CCW: -1
    rotor_xy_normalized: list[list[float]] = [ 
        [+0.09057, +0.084212],  # Front right
        [+0.09057, -0.084212],  # Front left
        [-0.09057, -0.084212],  # Back left
        [-0.09057, +0.084212]]  # Back right
    
    # Frame settings
    quad_com_pos_q    = [0, 0, 0]
    arm_com_pos_a     = [0.04, 0, 0]
    gripper_com_pos_g = [0.02, 0, 0]
    arm_pos_q         = [0, 0, -0.05]
    gripper_pos_a     = [0.2054747, 0, 0]
    
class ManipulatorBaseController(ActionTerm):
    
    def __init__(self, cfg: ManipulatorBaseControllerCfg, env: ManagerBasedEnv):
        
        super().__init__(cfg, env)
        
        self._asset: AssetBase = self._env.scene[self.cfg.asset_name]
        self._device = self._asset.device
        self._dtype = torch.float32
        
        # Set tensors
        self._arm_length = torch.tensor(self.cfg.arm_length, device=self._device, dtype=self._dtype) # (1,)
        self._rotor_dirs = torch.tensor(self.cfg.rotor_dirs, device=self._device, dtype=self._dtype)  # (4,)
        self._rotor_xy = torch.tensor(self.cfg.rotor_xy_normalized, device=self._device, dtype=self._dtype) * self._arm_length # (4,2)

        # Set conversion factors
        self._rad_to_rpm = torch.tensor(60 / (2 * math.pi), device=self._device, dtype=self._dtype)
        self._rpm_to_rad = torch.tensor((2 * math.pi) / 60, device=self._device, dtype=self._dtype)

        # Convert coefficients to rad/s
        self._k_f_rpm2 = torch.tensor(cfg.k_f_rpm2, device=self._device, dtype=self._dtype)
        self._k_m_rpm2 = torch.tensor(cfg.k_m_rpm2, device=self._device, dtype=self._dtype)
        self._k_f = self._k_f_rpm2 * (self._rad_to_rpm ** 2)
        self._k_m = self._k_m_rpm2 * (self._rad_to_rpm ** 2) 
        
        # Convert min/max rpm to rad/s
        self._w_min_rpm = torch.tensor(cfg.w_min_rpm, device=self._device, dtype=self._dtype)
        self._w_max_rpm = torch.tensor(cfg.w_max_rpm, device=self._device, dtype=self._dtype)
        self._w_min = self._w_min_rpm * self._rpm_to_rad
        self._w_max = self._w_max_rpm * self._rpm_to_rad

        # Initialize action tensors
        N = self._env.num_envs
        self._N = N
        self._raw = torch.zeros(N, 4, device=self._device)          # (N,4) raw actions
        self._thrust = torch.zeros(N, 1, 3, device=self._device)    # (N,1,3) total thrust force
        self._moment = torch.zeros(N, 1, 3, device=self._device)    # (N,1,3) total moment/torque
        self._omega = torch.zeros(N, 4, device=self._device)        # (N,4) processed motor speeds (rad/s)

        # Get body IDs
        ids, names = self._asset.find_bodies(".*", preserve_order=True)
        quad_index = names.index("quadrotor_visual")
        arm_index = names.index("arm_visual")
        gripper_index = names.index("gripper_visual") 
        
        self._quad_index = quad_index  
        self._arm_index = arm_index  
        self._gripper_index = gripper_index  
        
        self._quad_ids = [int(ids[quad_index])]       
        self._arm_ids = [int(ids[arm_index])]
        self._gripper_ids = [int(ids[gripper_index])]
        
        # Get mass properties
        mass_all = self._asset.root_physx_view.get_masses()  
        mass_all = torch.as_tensor(mass_all, device=self._device, dtype=self._dtype)  # (N, B)  
        self._quad_mass    = mass_all[:, quad_index:quad_index+1]  # (N,1)  
        self._arm_mass     = mass_all[:, arm_index:arm_index+1]    # (N,1)  
        self._gripper_mass = mass_all[:, gripper_index:gripper_index+1]  # (N,1)  
        self._mass   = self._quad_mass + self._arm_mass + self._gripper_mass # (N,1)
        
        # Get inertia properties
        inertia_all = self._asset.data.default_inertia  # (N, B, 9)  
        self._quad_inertia = inertia_all[:, quad_index][:, [0, 4, 8]].to(self._device, self._dtype)  # (N,3)  
        self._arm_inertia = inertia_all[:, arm_index][:, [0, 4, 8]].to(self._device, self._dtype)   # (N,3)  
        self._gripper_inertia = inertia_all[:, gripper_index][:, [0, 4, 8]].to(self._device, self._dtype)  # (N,3)  
        
        # Get Center of Mass of each elements
        self._quad_com_pos_q    = torch.tensor(cfg.quad_com_pos_q, device=self._device, dtype=self._dtype).expand(N,-1)     # (N, 3)
        self._arm_com_pos_a     = torch.tensor(cfg.arm_com_pos_a, device=self._device, dtype=self._dtype).expand(N,-1)  # (N, 3)
        self._gripper_com_pos_g = torch.tensor(cfg.gripper_com_pos_g, device=self._device, dtype=self._dtype).expand(N,-1)  # (N, 3)
        
        # Get relative position of each frames
        self._arm_pos_q         = torch.tensor(cfg.arm_pos_q, device=self._device, dtype=self._dtype).expand(N,-1)  # (N, 3)
        self._gripper_pos_a     = torch.tensor(cfg.gripper_pos_a, device=self._device, dtype=self._dtype).expand(N,-1)  # (N, 3)
        
        self.ManipulatorCascadeController = mdp.ManipulatorCascadeController(
            num_envs = CONFIG["SCENE"]["NUM_ENVS"],
            dt = CONFIG["ENV"]["PHYSICS_DT"],
            cfg = mdp.ManipulatorCascadeControllerCfg(),
            dtype = self._dtype,
        )
    
    def _update_mass_inertia(self):
        
        N = self._N
        
        # Get quaternion data of each elements
        body_quat_w    = self._asset.data.body_quat_w       # (N, B, 4)  
        
        # Get quaternion of each elements
        quad_quat_w    = body_quat_w[:, self._quad_index, :]      # (N,4)
        arm_quat_w     = body_quat_w[:, self._arm_index, :]       # (N,4)
        gripper_quat_w = body_quat_w[:, self._gripper_index,:]    # (N,4)
        
        # Compute Rotation matrix
        self._R_wq = math_utils.matrix_from_quat(quad_quat_w)     # (N,3,3)
        self._R_wa = math_utils.matrix_from_quat(arm_quat_w)      # (N,3,3)
        self._R_wg = math_utils.matrix_from_quat(gripper_quat_w)  # (N,3,3)

        R_qw = self._R_wq.transpose(1, 2)                         
        R_aw = self._R_wa.transpose(1, 2)                         

        self._R_qa = torch.bmm(R_qw, self._R_wa)            # arm from quad
        self._R_ag = torch.bmm(R_aw, self._R_wg)            # gripper from arm
        self._R_qg = torch.bmm(R_qw, self._R_wg)            # gripper from quad

        # Get Center of Mass in quadrotor frame
        self._arm_com_pos_q = self._arm_pos_q + torch.bmm(self._R_qa, self._arm_com_pos_a.unsqueeze(-1)).squeeze(-1)       
        self._gripper_pos_q   = self._arm_pos_q + torch.bmm(self._R_qa, self._gripper_pos_a.unsqueeze(-1)).squeeze(-1)        
        self._gripper_com_pos_q = self._gripper_pos_q + torch.bmm(self._R_qg, self._gripper_com_pos_g.unsqueeze(-1)).squeeze(-1)  
        
        # Get Center of Mass of total system in quadrotor frame
        self._total_com_pos_q = (
            self._quad_com_pos_q * self._quad_mass          
            + self._arm_com_pos_q * self._arm_mass             
            + self._gripper_com_pos_q * self._gripper_mass      
        ) / self._mass                                         

        # Get inertia tensors in quadrotor frame
        Jq_local = torch.diag_embed(self._quad_inertia)           # (N,3,3)  
        Ja_local = torch.diag_embed(self._arm_inertia)            # (N,3,3)  
        Jg_local = torch.diag_embed(self._gripper_inertia)        # (N,3,3)  

        self._quad_inertia_q    = Jq_local                       
        self._arm_inertia_q     = torch.bmm(               
            torch.bmm(self._R_qa, Ja_local), self._R_qa.transpose(1, 2)
        )                                                          
        self._gripper_inertia_q = torch.bmm(                       
            torch.bmm(self._R_qg, Jg_local), self._R_qg.transpose(1, 2)
        )                                                          

        # Get inertia tensors in origin of quadrotor frame
        I3 = torch.eye(3, device=self._device, dtype=self._dtype).unsqueeze(0).expand(N, 3, 3)  

        r_q = self._quad_com_pos_q                                  # (N,3)  
        r_a = self._arm_com_pos_q                                   # (N,3)  
        r_g = self._gripper_com_pos_q                               # (N,3)  

        m_q = self._quad_mass.view(N, 1, 1)                         # (N,1,1)  
        m_a = self._arm_mass.view(N, 1, 1)                          # (N,1,1)  
        m_g = self._gripper_mass.view(N, 1, 1)                      # (N,1,1)  

        r_q2 = (r_q * r_q).sum(dim=1).view(N, 1, 1)                 # (N,1,1)  
        r_a2 = (r_a * r_a).sum(dim=1).view(N, 1, 1)                 # (N,1,1)  
        r_g2 = (r_g * r_g).sum(dim=1).view(N, 1, 1)                 # (N,1,1)   

        rrT_q = torch.bmm(r_q.unsqueeze(-1), r_q.unsqueeze(1))      # (N,3,3)  
        rrT_a = torch.bmm(r_a.unsqueeze(-1), r_a.unsqueeze(1))      # (N,3,3)  
        rrT_g = torch.bmm(r_g.unsqueeze(-1), r_g.unsqueeze(1))      # (N,3,3)   

        J_q_O = self._quad_inertia_q + m_q * (r_q2 * I3 - rrT_q)    # (N,3,3)  
        J_a_O = self._arm_inertia_q  + m_a * (r_a2 * I3 - rrT_a)    # (N,3,3)  
        J_g_O = self._gripper_inertia_q + m_g * (r_g2 * I3 - rrT_g) # (N,3,3)  

        self._J_O = J_q_O + J_a_O + J_g_O                           # (N,3,3) 
    
    @property
    def action_dim(self) -> int:
        return 4
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._omega
    
    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        
        actions = actions.to(device=self._device, dtype=self._dtype)
        actions = torch.clamp(actions, min=-1.0, max=1.0)
        self._update_mass_inertia()
        self._raw = actions
        
        # Observations
        asset_cfg = mdp.SceneEntityCfg(name=self.cfg.asset_name)
        pos_w = mdp.ObservationFns.get_current_pos_w(self._env, asset_cfg)  # (N, 3)
        pos_sp_w = mdp.ObservationFns.get_spawn_pos_w(self._env, asset_cfg) # (N, 3)
        vel_w = mdp.ObservationFns.get_lin_vel_w(self._env, asset_cfg)      # (N, 3)
        quat_w = mdp.ObservationFns.get_quaternion_w(self._env, asset_cfg)  # (N, 4)
        ang_vel_b = mdp.ObservationFns.get_ang_vel_b(self._env, asset_cfg)  # (N, 3)

        # Cascaded control
        vel_sp_w = self.ManipulatorCascadeController.position_control(pos_w, pos_sp_w)
        acc_sp_w = self.ManipulatorCascadeController.velocity_control(vel_w, vel_sp_w)
        yaw_sp = torch.zeros_like(acc_sp_w[:, :1])  # (N, 1)
        # yaw_sp = ActionFns._quat_to_yaw(quat_w) # (N, 1) hold inital yaw
        quat_sp_w, t_norm = self.ManipulatorCascadeController.acc_yaw_to_quaternion_thrust(acc_sp_w, yaw_sp)
        ang_vel_sp_b = self.ManipulatorCascadeController.attitude_control(quat_w, quat_sp_w)
        momentum_sp_b = self.ManipulatorCascadeController.body_rate_control(torch.diagonal(self._J_O, dim1=-2, dim2=-1).contiguous(), ang_vel_b, ang_vel_sp_b)

        # Add residual thrust command from action   
        res = self._raw
        res_thrust = res[:, 0:1]
        res_torque = res[:, 1:4]
        F_per_rotor_max = self._k_f * (self._w_max ** 2)
        F_total_max = 4.0 * F_per_rotor_max
        delta_accel = (res_thrust * F_total_max) / self._mass
        t_norm = torch.clamp(t_norm + delta_accel, min=0.0)
        torque_limit = self.ManipulatorCascadeController._torque_limit
        momentum_sp_b = torch.clamp(momentum_sp_b + res_torque * torque_limit, -torque_limit, torque_limit)

        # Mixing
        self._mixed_thrust = ManipulatorActionFns.mix_rate_thrust_to_rotor_forces(
            required_accel = t_norm,
            momentum_sp_b = momentum_sp_b,
            rotor_xy = self._rotor_xy,
            rotor_dirs = self._rotor_dirs,
            k_f_rpm2 = self._k_f_rpm2,
            k_m_rpm2 = self._k_m_rpm2,
            mass = self._mass,
            com_pos_q = self._total_com_pos_q,  
        )  # (N, 4)
        
        self._mixed_thrust_desaturated = ManipulatorActionFns.desaturate_rotor_forces(
            rotor_forces = self._mixed_thrust,
            k_f_rpm2 = self._k_f_rpm2,
            w_min_rpm = self._w_min_rpm,
            w_max_rpm = self._w_max_rpm,
        ) # (N, 4)
        
        self._mixed_rpm = ManipulatorActionFns.thrust_to_rpm(
            rotor_forces = self._mixed_thrust_desaturated,
            w_min_rpm = self._w_min_rpm,
            w_max_rpm = self._w_max_rpm,
            k_f = self._k_f,
        ) # (N, 4)
        
        self._omega = self._mixed_rpm * self._rpm_to_rad # (N, 4)
        
        return self._omega
        
    def apply_actions(self):

        Fz = (self._omega ** 2 * self._k_f).sum(dim=1) # (N,)
        self._thrust.zero_()
        self._thrust[:, 0, 2] = Fz
        
        x = self._rotor_xy[:, 0]
        y = self._rotor_xy[:, 1]
        tau_x = (y * self._mixed_thrust_desaturated).sum(dim=1) # (N,)
        tau_y = (-x * self._mixed_thrust_desaturated).sum(dim=1) # (N,)
        tau_z = (self._rotor_dirs * self._k_m / self._k_f * self._mixed_thrust_desaturated).sum(dim=1) # (N,)
        self._moment.zero_()
        self._moment[:, 0, 0] = tau_x
        self._moment[:, 0, 1] = tau_y
        self._moment[:, 0, 2] = tau_z

        self._asset.set_external_force_and_torque(
            body_ids = self._quad_ids,
            forces = self._thrust,
            torques = self._moment,
            is_global = False,
        )
    
        