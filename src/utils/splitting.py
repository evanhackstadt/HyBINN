# src/utils/splitting.py

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def stratified_train_test_split(events, config):
    """
    Performs class-stratified train/test split on the full dataset
    
    Args:
        events (array): full survival data (y) to be stratified
        config (dict): config dict containing test size and seed parameters
    
    Returns:
        (train_val_idx, test_idx): indices for the train+val and test partitions of the dataset
    """
    
    test_size = config['data']['test']
    seed = config['data']['random_seed']
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(sss.split(X=np.zeros(len(events)), y=events))   # stratify on events; X placeholder
    
    return train_val_idx, test_idx


def get_stratified_kfold_indices(events, config):
    """
    Performs class-stratified train/validation split for each fold of the dataset
    
    Args:
        events (array): train+val survival data (y) to be stratified
        config (dict): config dict containing folds and seed parameters
    
    Returns:
        (train_idx, val_idx): indices for the train and validation sets for each fold
    """
    
    folds = config['training']['folds']
    seed = config['data']['random_seed']
    
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    
    for train_idx, val_idx in skf.split(X=np.zeros(len(events)), y=events):
        yield train_idx, val_idx