# Generated using Claude Sonnett 4.6

import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Sets all relevant random seeds for reproducibility across a full training run.
    Call this once at the top of main() before any data splitting or model instantiation.
    
    Args:
        seed (int): the random seed value, typically from config['data']['random_seed']
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)