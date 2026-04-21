import torch
from torch.utils.data import Subset, Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from utils.splitting import stratified_train_test_split, get_stratified_kfold_indices


class SurvivalDataset(Dataset):
    
    def __init__(self, x_mapped, x_unmapped, times, events):
        """
        Args:
            x_mapped:   numpy array (n_patients, n_mapped_genes)
            x_unmapped: numpy array (n_patients, n_unmapped_genes)
            times:      numpy array (n_patients,)
            events:     numpy array (n_patients,)
        """
        
        # convert numpy --> tensors
        self.x_mapped   = torch.tensor(x_mapped, dtype=torch.float32)
        self.x_unmapped = torch.tensor(x_unmapped, dtype=torch.float32)
        self.y_time     = torch.tensor(times, dtype=torch.float32)
        self.y_event    = torch.tensor(events, dtype=torch.float32)
        
        self.n_samples = len(events)
        self.mapped_dim = self.x_mapped.shape[1]
        self.unmapped_dim = self.x_unmapped.shape[1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        item = {
            'X_mapped': self.x_mapped[idx],
            'X_unmapped': self.x_unmapped[idx],
            'y_time': self.y_time[idx],
            'y_event': self.y_event[idx]
        }
        return item     # returns a dict



def get_dataloader(dataset, indices, config, shuffle=False):
    """
    Wraps a Subset and returns a DataLoader
    
    Args:
        dataset (SurvivalDataset): the full dataset object to extract indices from
        indices (array): the indices to be wrapped into a DataLoader
        config (dict): config dictionary containing batch_size parameter
        shuffle (bool): shuffle param for DataLoader
    
    Returns:
        DataLoader wrapping the index-specified subset of the dataset
    """
    
    batch_size = config['data']['batch_size']
    
    loader = DataLoader(Subset(dataset, indices), 
                        batch_size=batch_size, shuffle=shuffle)
    
    return loader


# TO DELETE:
'''

def get_dataloaders(x_mapped, x_unmapped, times, events,
                    train, val, test, batch_size, random_seed):
    """
    Instantiates SurvivalDataset and splits it into three DataLoaders
    
    Args:
        x_mapped (np array): mapped gene data 2D array
        x_unmapped (np array): unmapped gene data 2D array
        times (np array): survival times array
        events (np array): survival events array (1=died, 0=censored)
        train (float): proportion of data for training set
        val (float): proportion of data for validation set
        test (float): proportion of data for testing set
        batch_size (int): number of samples in each batch
        random_seed (int): seed for random split of dataset
    
    Returns:
        train (shuffled), val, and test DataLoaders
    """
    
    if abs(train + val + test - 1.0) > 1e-6:    # avoid underflow
        raise ValueError("train, val, and test must sum to 1.0")
    
    dataset = SurvivalDataset(x_mapped, x_unmapped, times, events)
    y_labels = dataset.y_event.tolist()

    # Stratified Split
    
    # First carve out test set
    trainval_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=test,
        stratify=y_labels,
        random_state=random_seed
    )
    
    # Then split trainval into train/val
    trainval_labels = [y_labels[i] for i in trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val,
        stratify=trainval_labels,
        random_state=random_seed
    )
    
    # Now create dataloaders using stratified indexes
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, 
                              shuffle=True, num_workers=0,drop_last=True)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size,
                              shuffle=False, num_workers=0, drop_last=False)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=batch_size,
                              shuffle=False, num_workers=0, drop_last=False)
    
    return train_loader, val_loader, test_loader

'''