# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# import stabilization_env_cfg.mdp as mdp


"""
Scene definition
This module includes quadrotor, groud plane, and dome light configurations
"""
 
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

    # quadrotor
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
        actuators={
            "dummy": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
            ),
        },
    )
    """Configuration for the Crazyflie quadcopter."""
    


"""
Action configuration
This module defines the action space for the stabilization task
For more details, refer to the Isaac Lab documentation :
https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#module-isaaclab.envs.mdp.actions
"""

# @configclass
# class ActionsCfg:

#     # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.actions.actions_cfg.JointEffortActionCfg
#     joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)


# @configclass
# class ObservationsCfg:
#     """Observation specifications for the MDP."""

#     @configclass
#     class PolicyCfg(ObsGroup):
#         """Observations for policy group."""

#         # observation terms (order preserved)
#         joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
#         joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

#         def __post_init__(self) -> None:
#             self.enable_corruption = False
#             self.concatenate_terms = True

#     # observation groups
#     policy: PolicyCfg = PolicyCfg()


# @configclass
# class EventCfg:
#     """Configuration for events."""

#     # reset
#     reset_cart_position = EventTerm(
#         func=mdp.reset_joints_by_offset,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
#             "position_range": (-1.0, 1.0),
#             "velocity_range": (-0.5, 0.5),
#         },
#     )

#     reset_pole_position = EventTerm(
#         func=mdp.reset_joints_by_offset,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
#             "position_range": (-0.25 * math.pi, 0.25 * math.pi),
#             "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
#         },
#     )


# @configclass
# class RewardsCfg:
#     """Reward terms for the MDP."""

#     # (1) Constant running reward
#     alive = RewTerm(func=mdp.is_alive, weight=1.0)
#     # (2) Failure penalty
#     terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
#     # (3) Primary task: keep pole upright
#     pole_pos = RewTerm(
#         func=mdp.joint_pos_target_l2,
#         weight=-1.0,
#         params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
#     )
#     # (4) Shaping tasks: lower cart velocity
#     cart_vel = RewTerm(
#         func=mdp.joint_vel_l1,
#         weight=-0.01,
#         params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
#     )
#     # (5) Shaping tasks: lower pole angular velocity
#     pole_vel = RewTerm(
#         func=mdp.joint_vel_l1,
#         weight=-0.005,
#         params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
#     )


# @configclass
# class TerminationsCfg:
#     """Termination terms for the MDP."""

#     # (1) Time out
#     time_out = DoneTerm(func=mdp.time_out, time_out=True)
#     # (2) Cart out of bounds
#     cart_out_of_bounds = DoneTerm(
#         func=mdp.joint_pos_out_of_manual_limit,
#         params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
#     )


# ##
# # Environment configuration


# @configclass
# class StabilizationEnvCfg(ManagerBasedRLEnvCfg):
#     # Scene settings
#     scene: StabilizationSceneCfg = StabilizationSceneCfg(num_envs=2, env_spacing=4.0)
#     # Basic settings
#     observations: ObservationsCfg = ObservationsCfg()
#     actions: ActionsCfg = ActionsCfg()
#     events: EventCfg = EventCfg()
#     # MDP settings
#     rewards: RewardsCfg = RewardsCfg()
#     terminations: TerminationsCfg = TerminationsCfg()

#     # Post initialization
#     def __post_init__(self) -> None:
#         """Post initialization."""
#         # general settings
#         self.decimation = 2
#         self.episode_length_s = 5
#         # viewer settings
#         self.viewer.eye = (8.0, 0.0, 5.0)
#         # simulation settings
#         self.sim.dt = 1 / 120
#         self.sim.render_interval = self.decimation