import torch
from torch.utils.data import Subset, Dataset, DataLoader


class SurvivalDataset(Dataset):
    
    def __init__(self, x_mapped, x_unmapped, x_clinical, times, events):
        """
        Args:
            x_mapped:   numpy array (n_patients, n_mapped_genes)
            x_unmapped: numpy array (n_patients, n_unmapped_genes)
            x_clinical: numpy array (n_patients, 3)
            times:      numpy array (n_patients,)
            events:     numpy array (n_patients,)
        """
        
        # convert numpy --> tensors
        self.x_mapped   = torch.tensor(x_mapped, dtype=torch.float32)
        self.x_unmapped = torch.tensor(x_unmapped, dtype=torch.float32)
        self.x_clinical = torch.tensor(x_clinical, dtype=torch.float32)
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
            'X_clinical': self.x_clinical[idx],
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