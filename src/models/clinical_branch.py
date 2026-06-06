import torch.nn as nn


# ClinicalBranch - takes T/N/M pathologic stage variables --> expands into 64-node embedding
'''
3 → [Linear → BN → ReLU → Dropout(0.3)] → 16
  → [Linear → BN → ReLU → Dropout(0.3)] → 32
  → [Linear, no activation] → 64 (embedding)
'''
class ClinicalBranch(nn.Module):
    
    def __init__(self, in_nodes, hidden_nodes_1, hidden_nodes_2, embedding_nodes):
        super(ClinicalBranch, self).__init__()
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)    # lighter dropout
        
        # input variables --> hidden layer 1
        self.fc1 = nn.Linear(in_nodes, hidden_nodes_1)
        self.bn1 = nn.BatchNorm1d(hidden_nodes_1)
        # hidden layer 1 --> hidden layer 2
        self.fc2 = nn.Linear(hidden_nodes_1, hidden_nodes_2)
        self.bn2 = nn.BatchNorm1d(hidden_nodes_2)
        # hidden layer 2 --> output layer (embedding)
        self.fc3 = nn.Linear(hidden_nodes_2, embedding_nodes, bias=False)
        
        if embedding_nodes == 1:    # risk score output
            self.fc3.weight.data.uniform_(-0.001, 0.001)
    
    def forward(self, x_mapped, x_unmapped, x_clinical):
        x = x_clinical
        
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