import argparse
from isaaclab.app import AppLauncher

MANUAL_ACTION = [-0.939, -0.937, -0.939, -0.939]

parser = argparse.ArgumentParser(description="Run quadrotor scene with Action-only (manual actions in code).")
parser.add_argument("--num_envs", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveScene
import stabilization.tasks.manager_based.stabilization.envs as envs
import stabilization.tasks.manager_based.stabilization.mdp as mdp

class _FakeEnv:
    def __init__(self, scene, sim):
        self.scene = scene
        self.num_envs = scene.num_envs
        self.dt = sim.get_physics_dt()

def quat_to_rpy_xyzw(q):
    x, y, z, w = q.unbind(-1)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (torch.pi / 2), torch.asin(sinp))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1)

def main():
    sim = SimulationContext()
    sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.5])

    scene_cfg = envs.StabilizationSceneCfg()
    scene_cfg.num_envs = args_cli.num_envs
    scene_cfg.env_spacing = 2.0
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()

    env = _FakeEnv(scene, sim)

    act_cfg = mdp.BaseControllerCfg(asset_name="Robot")
    ctrl = mdp.BaseController(act_cfg, env)
    device = ctrl._asset.device

    root = ctrl._asset.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    root[:, 2] += 1.0
    ctrl._asset.write_root_pose_to_sim(root[:, :7])
    ctrl._asset.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))
    scene.write_data_to_sim()

    a = torch.tensor(MANUAL_ACTION, dtype=torch.float32, device=device).clamp(-1.0, 1.0)
    actions = a.view(1, 4).repeat(env.num_envs, 1)

    step = 0
    while simulation_app.is_running():
        ctrl.process_actions(actions)
        ctrl.apply_actions()
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

        rs = ctrl._asset.data.root_state_w
        p = rs[0, :3]
        q = rs[0, 3:7]
        rpy = quat_to_rpy_xyzw(q)
        if step % 30 == 0:
            print(f"pos=({p[0].item():.3f}, {p[1].item():.3f}, {p[2].item():.3f}) "
                  f"rpy=({rpy[0].item():.3f}, {rpy[1].item():.3f}, {rpy[2].item():.3f})")
        eye = (p + torch.tensor([2.0, 2.0, 1.2], device=p.device)).tolist()
        target = (p + torch.tensor([0.0, 0.0, 0.2], device=p.device)).tolist()
        sim.set_camera_view(eye, target)
        step += 1

if __name__ == "__main__":
    main()
