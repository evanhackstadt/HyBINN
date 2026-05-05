import torch
import torch.nn as nn
import torch.nn.functional as F


# BINNBranch - takes genes mapped to pathways --> outputs 64-node embedding
'''
mapped_genes → [sparse Linear → Tanh] → pathways
             → [Linear → Tanh → Dropout(0.5)] → hidden
             → [Linear → Tanh → Dropout(0.5)] → 64 (embedding)
'''
class BINNBranch(nn.Module):
    
    def __init__(self, in_nodes, pathway_nodes, hidden_nodes, out_nodes, pathway_mask):
        super(BINNBranch, self).__init__()
        
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.register_buffer("mask", torch.tensor(pathway_mask))    # save pathway mask
        
        # input genes --> pathway layer
        self.sc1 = nn.Linear(in_nodes, pathway_nodes)
        # pathway layer --> hidden layer (superpathways)
        self.fc2 = nn.Linear(pathway_nodes, hidden_nodes)
        # hidden layer --> hidden layer 2 (embedding)
        self.fc3 = nn.Linear(hidden_nodes, out_nodes, bias=False)
    
    
    def forward(self, x_mapped, x_unmapped):
        # BINN branch is for mapped genes
        x = x_mapped
        
        # apply the mask matrix and update sc1's weights
        masked_weights = self.sc1.weight * self.mask.T
        
        # manually calculate sc1: mask input genes --> pathway layer
        x = self.tanh(F.linear(x, masked_weights, self.sc1.bias))
        self.pathway_activations = x  # store it for interpretability
        
        # the following layers are normal
        # x = self.dropout(x)           # TODO: test with/without
        x = self.tanh(self.fc2(x))
        # x = self.dropout(x)
        x = self.tanh(self.fc3(x))
        
        return x