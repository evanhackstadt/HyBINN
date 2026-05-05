# src/utils/config.py

import yaml

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Validate params
    train = cfg['data']['train']
    val = cfg['data']['val']
    test = cfg['data']['test']

    if abs(train + val + test - 1.0) > 1e-6:    # avoid underflow
        raise ValueError("train, val, and test must sum to 1.0")

    return cfg


# TODO: add more param validation / setting of default params