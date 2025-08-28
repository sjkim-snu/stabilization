import os
import yaml 

_DEFAULT_CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_parameters(path=None):

    """
    Load configuration parameters from a YAML file.
    If path is provided, it will be used instead of the default config.yaml.
    """
    
    cfg_path = path or _DEFAULT_CFG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data

__all__ = ["load_parameters"]
