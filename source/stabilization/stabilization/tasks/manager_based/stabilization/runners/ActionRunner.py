"""
This script runs scene and action.
usage : python3 ActionRunner.py
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run quadrotor scene with Action-only debug loop.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
import stabilization.tasks.manager_based.stabilization.envs as envs
import stabilization.tasks.manager_based.stabilization.mdp as mdp

class _FakeEnv:
    """Minimal env shim required by an ActionTerm.

    Provides: scene, num_envs, dt
    """
    def __init__(self, scene: InteractiveScene, sim: sim_utils.SimulationContext):
        self.scene = scene
        self.num_envs = scene.num_envs
        self.dt = sim.get_physics_dt()


def _make_actions(mode: str, n: int, step: int, dt: float) -> torch.Tensor:
    """Return (n,4) actions in [-1, 1] for the chosen test mode.

    These are *policy* actions; ActionTerm will map them to motor speeds/thrusts internally.
    """
    # Base hover-ish level (centered). You may tweak this if your mapping expects other ranges.
    base = 0.2  # try 0.2~0.6 depending on your k_f/k_m scaling
    a = torch.full((n, 4), base)

    t = step * dt
    if mode == "hover":
        return a
    elif mode == "roll":
        # Roll: increase motors on +y arm, decrease on -y arm (assuming XY cross layout)
        a[:, 0] += 0.2  # front-left
        a[:, 1] -= 0.2  # front-right
        a[:, 2] += 0.2  # rear-right
        a[:, 3] -= 0.2  # rear-left
    elif mode == "pitch":
        # Pitch: front down / rear up (sign may need flipping depending on rotor_pos_xy)
        a[:, 0] -= 0.2
        a[:, 1] -= 0.2
        a[:, 2] += 0.2
        a[:, 3] += 0.2
    elif mode == "yaw":
        # Yaw: counter-torque pair. Alternate sign each second for visibility.
        s = 1.0 if int(t) % 2 == 0 else -1.0
        a[:, 0] += 0.25 * s
        a[:, 1] -= 0.25 * s
        a[:, 2] += 0.25 * s
        a[:, 3] -= 0.25 * s
    return a.clamp(-1.0, 1.0)


def main() -> None:
    # --- Simulation context
    
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(cfg=sim_cfg)

    # Camera pose (eye, target)
    sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.5])

    # --- Scene
    scene_cfg = envs.StabilizationSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=0.2,
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()
    
    # Minimal env shim for ActionTerm
    env = _FakeEnv(scene, sim)

    # --- Build Action term (no managers)
    act_cfg = mdp.BaseControllerCfg(
        asset_name="Robot",
    )
    action_term = mdp.BaseController(act_cfg, env)

    # scene.reset()  # enable later if asset supports reset

    step = 0
    while simulation_app.is_running():
        actions = _make_actions(args_cli.mode, env.num_envs, step, sim.get_physics_dt())
        action_term.process_actions(actions)
        action_term.apply_actions()
        sim.step()
        step += 1


if __name__ == "__main__":
    main()
