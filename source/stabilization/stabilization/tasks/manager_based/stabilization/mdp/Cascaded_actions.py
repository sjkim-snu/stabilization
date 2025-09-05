from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.assets import AssetBase, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from dataclasses import dataclass
from isaaclab.utils import configclass
from abc import abstractmethod

import stabilization.tasks.manager_based.stabilization.envs as envs
import isaaclab.envs as env
import torch
import math

@configclass
class BaseControllerCfg(ActionTermCfg):
    asset_name: str = "Robot"
    
    