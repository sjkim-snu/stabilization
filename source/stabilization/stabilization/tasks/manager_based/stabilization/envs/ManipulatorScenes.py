from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
import stabilization.tasks.manager_based.stabilization.models as models
import importlib.resources as ilr

"""
Scene configuration
This module includes quadrotor, groud plane, and dome light configurations.
If configclass is not used, many errors may occur in the simulation.
Links below provide more information about the classes used in this file.
"""

# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.utils.html#module-isaaclab.utils.configclass
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.scene.html#isaaclab.scene.InteractiveSceneCfg 

@configclass
class ManipulatorSceneCfg(InteractiveSceneCfg):

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
    
    # Configuration for a simple 0-DOF aerial manipulator
    Robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ilr.files(models).joinpath("AerialManipulator_0dof.usd")),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0, 
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            # Default joint positions and velocities is 0.0
            joint_pos={
                ".*": 0.0,
            },
            joint_vel={
                "prop1": 0.0,
                "prop2": -0.0,
                "prop3": 0.0,
                "prop4": -0.0,
            },
        ),

        # Available joints: 
        actuators={ 
            "dummy": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )