"""
This script runs only the scene for quadrotor stabilization in Isaac Sim.
usage : python3 SceneRunner.py
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run the quadrotor stabilization RL environment.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import stabilization.tasks.manager_based.stabilization.envs as envs
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

def main():
    """Main entry."""
    
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(cfg=sim_cfg) 
    
    # Set camera view
    sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.5])
    
    # Apply the scene configuration
    scene_cfg = envs.StabilizationSceneCfg(num_envs=10, env_spacing=0.2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    while simulation_app.is_running():
        sim.step()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()