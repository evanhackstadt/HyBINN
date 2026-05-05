import torch.nn as nn


# GeneBranch - takes unmapped genes (residual) --> outputs 64-node embedding
'''
7823 → [Linear → BN → ReLU → Dropout(0.5)] → 1024
     → [Linear → BN → ReLU → Dropout(0.5)] → 256
     → [Linear] → 64 (embedding, no activation)
'''
class GeneBranch(nn.Module):
    
    def __init__(self, in_nodes, hidden_nodes_1, hidden_nodes_2, out_nodes):
        super(GeneBranch, self).__init__()
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
        # input genes --> hidden layer 1
        self.fc1 = nn.Linear(in_nodes, hidden_nodes_1)
        self.bn1 = nn.BatchNorm1d(hidden_nodes_1)
        # hidden layer 1 --> hidden layer 2
        self.fc2 = nn.Linear(hidden_nodes_1, hidden_nodes_2)
        self.bn2 = nn.BatchNorm1d(hidden_nodes_2)
        # hidden layer 2 --> output layer (embedding)
        self.fc3 = nn.Linear(hidden_nodes_2, out_nodes, bias=False)
    
    
    def forward(self, x_mapped, x_unmapped):
        # Gene branch is for unmapped genes
        x = x_unmapped
        
        # input genes --> hidden layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # hidden layer 1 --> hidden layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        # hidden layer 2 --> output layer (embedding)
        x = self.fc3(x)
        
        return x