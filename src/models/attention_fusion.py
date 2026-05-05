import torch
import torch.nn as nn
import torch.nn.functional as F


# AttentionFusion - takes embeddings from all 3 branches --> attention mechanism --> fused embedding
'''
e_i = Linear(64 → 1)(h_i)          # scalar attention energy per branch
[a1, a2, a3] = Softmax([e1, e2, e3])  # attention weights, sum to 1
z = a1*h1 + a2*h2 + a3*h3          # weighted combination, shape (batch, 64)
'''
class AttentionFusion(nn.Module):
    
    def __init__(self, embed_dim):
        super(AttentionFusion, self).__init__()
        
        self.softmax = nn.Softmax(dim=1)
        
        # embeddings --> scalar risk score output
        self.attn_binn     = nn.Linear(embed_dim, 1, bias=False)
        self.attn_gene     = nn.Linear(embed_dim, 1, bias=False)
        self.attn_clinical = nn.Linear(embed_dim, 1, bias=False)
    
    
    def forward(self, h1, h2, h3):
        # input is embeddings from each branch (binn, gene, clinical)
        
        # embeddings --> scalar risk score output
        e1 = self.attn_binn(h1)
        e2 = self.attn_gene(h2)
        e3 = self.attn_clinical(h3)
        
        # softmax
        energies = torch.cat([e1, e2, e3], dim=1)   # (batch, 3)
        a = self.softmax(energies)                  # (batch, 3)
        
        # a[:,i] is (batch,) — unsqueeze to broadcast over embed_dim
        z = a[:,0:1]*h1 + a[:,1:2]*h2 + a[:,2:3]*h3  # (batch, embed_dim)
        
        self.attn_weights = a   # store for interpretability
        return z