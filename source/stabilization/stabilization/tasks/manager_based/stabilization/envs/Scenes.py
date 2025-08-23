from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

"""
Scene configuration
This module includes quadrotor, groud plane, and dome light configurations.
If configclass is not used, many errors may occur in the simulation.
"""

# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.utils.html#module-isaaclab.utils.configclass
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.scene.html#isaaclab.scene.InteractiveSceneCfg 

@configclass
class StabilizationSceneCfg(InteractiveSceneCfg):
    """Configuration for a Quadrotor scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Configuration for the Crazyflie quadcopter
    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationCfg
    
    Robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd",

            # Rigid properties
            # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.schemas.html#isaaclab.sim.schemas.RigidBodyPropertiesCfg

            rigid_props=sim_utils.RigidBodyPropertiesCfg( 
                disable_gravity=False,             
                max_depenetration_velocity=10.0,   
                angular_damping=0.1,       
                enable_gyroscopic_forces=True,     
            ),
            
            # Entire drone properties
            # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.schemas.html#isaaclab.sim.schemas.ArticulationRootPropertiesCfg

            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
        ),

        # Initial state configuration for the quadrotor
        # https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/assets/articulation/articulation_cfg.html#ArticulationCfg.InitialStateCfg

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5), # z potision above ground
            joint_pos={
                ".*": 0.0,
            },
            joint_vel={
                "m1_joint": 0.0,
                "m2_joint": 0.0,
                "m3_joint": 0.0,
                "m4_joint": 0.0,
            },
        ),
        
        # Actuators configuration for the quadrotor
        # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html#isaaclab.actuators.ImplicitActuatorCfg
        
        actuators={
            "dummy": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
            ),
        },
    )