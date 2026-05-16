# src/utils/config.py

import os
import yaml


# def resolve_path(path_value, script_path):
#     if os.path.isabs(path_value):
#         return path_value
#     return os.path.abspath(os.path.join(script_path, path_value))

def load_config(config_path, script_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths against the experiments script directory
    # cfg['data']['data_path'] = resolve_path(cfg['data']['data_path'], script_path)
    # cfg['data']['reactome_path'] = resolve_path(cfg['data']['reactome_path'], script_path)
    # cfg['logging']['run_dir'] = resolve_path(cfg['logging']['run_dir'], script_path)
    
    # Validate params
    train = cfg['data']['train']
    val = cfg['data']['val']
    test = cfg['data']['test']

    if abs(train + val + test - 1.0) > 1e-6:    # avoid underflow
        raise ValueError("train, val, and test must sum to 1.0")

    return cfg


# TODO: add more param validation / setting of default params