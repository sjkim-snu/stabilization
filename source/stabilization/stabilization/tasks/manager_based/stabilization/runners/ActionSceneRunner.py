import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.scene import InteractiveScene
import isaaclab.sim as sim_utils
import torch

import stabilization.tasks.manager_based.stabilization.envs as envs
import stabilization.tasks.manager_based.stabilization.mdp as mdp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sim = sim_utils.SimulationContext()
sim.set_camera_view(eye=[4.0, 4.0, 3.0], target=[0.0, 0.0, 0.0])

scene_cfg = envs.StabilizationSceneCfg()

if hasattr(scene_cfg, "num_envs"):
    scene_cfg.num_envs = args.num_envs
if hasattr(scene_cfg, "env_spacing"):
    scene_cfg.env_spacing = 0.2
if hasattr(scene_cfg, "scene_name"):
    scene_cfg.scene_name = "QuadStabilizeScene"

scene = InteractiveScene(cfg=scene_cfg)
scene.reset()
print("Scene entities:", list(scene.keys())) 

class _FakeEnv:
    def __init__(self, scene, device):
        self.scene = scene
        self.device = device
        self.num_envs = getattr(scene, "num_envs", 1)

env = _FakeEnv(scene, device)

act_cfg = mdp.BaseControllerCfg(
    class_type=mdp.BaseController,
    robot_entity_name="Robot",
)
action_term = mdp.BaseController(act_cfg, env)

N = env.num_envs
actions = 0.2 * torch.ones(N, 4, device=device)

i = 0
while simulation_app.is_running():
    action_term.process_action(actions)  
    action_term.apply_action()          
    sim.step()
    i += 1
