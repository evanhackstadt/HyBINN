import torch.nn as nn


class StandaloneBranch(nn.Module):
    """Wraps a single branch + survival head for standalone testing."""
    
    def __init__(self, branch, head):
        super().__init__()
        self.branch = branch
        self.head = head

    def forward(self, x_mapped, x_unmapped, x_clinical):
        emb = self.branch(x_mapped, x_unmapped, x_clinical)
        return self.head(emb)