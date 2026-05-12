import torch.nn as nn


class StandaloneBranch(nn.Module):
    """Wraps a single branch + survival head for standalone testing."""
    
    def __init__(self, branch, head):
        super().__init__()
        self.branch = branch
        self.head = head

    def forward(self, x_mapped, x_unmapped):
        emb = self.branch(x_mapped, x_unmapped)
        return self.head(emb)