# Copyright (c) 2022-2025, The Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""
Run the quadrotor hover-stabilization RL environment (manager-based).

Example:
    ./isaaclab.sh -p scripts/air/run_quad_hover_env.py --num_envs 512 --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run the quadrotor stabilization RL environment.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from stabilization_env_cfg import StabilizationEnvCfg


def main():
    """Main entry."""
    # 1) Create env config
    env_cfg = StabilizationEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device  # e.g., "cuda:0" or "cpu"

    # 2) Build environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    print("-" * 80)
    print(f"[INFO] QuadStabilizeEnv created on device={env.device}, num_envs={env.num_envs}, "
          f"action_dim={env.action_manager.action.shape[-1]}, step_dt={env.step_dt:.5f}s")

    # 3) Simple random policy loop (for smoke test)
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # Hard reset every N steps just like tutorial (optional)
            if step % 300 == 0:
                env.reset()
                print("-" * 80)
                print(f"[INFO] Environment reset at step={step}")

            # # Sample random actions in [-1, 1] (matches our rates+thrust normalized action)
            # actions = 2.0 * torch.rand_like(env.action_manager.action) - 1.0

            # # Step
            # obs, rew, terminated, truncated, info = env.step(actions)

            # # Print a compact status for env-0
            # # Observations are concatenated as: [vx, vy, vz, roll, pitch, yaw]
            # o0 = obs["policy"][0]
            # vx, vy, vz, roll, pitch, yaw = [float(x) for x in o0.tolist()]
            # r0 = float(rew[0].item())
            # print(f"[Env0] v=({vx:+.2f},{vy:+.2f},{vz:+.2f}) m/s | "
            #       f"rpy=({roll:+.2f},{pitch:+.2f},{yaw:+.2f}) rad | "
            #       f"rew={r0:+.3f}")

            # step += 1

    # 4) Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
