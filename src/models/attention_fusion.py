import torch
import torch.nn as nn


# AttentionFusion - takes embeddings from muultiple branches --> attention mechanism --> fused embedding
'''
e_i = Linear(64 → 1)(h_i)          # scalar attention energy per branch
[a1, a2, a3] = Softmax([e1, e2, e3])  # attention weights, sum to 1
z = a1*h1 + a2*h2 + a3*h3          # weighted combination, shape (batch, 64)
'''
class AttentionFusion(nn.Module):
    
    def __init__(self, embed_dim, n_branches):
        super(AttentionFusion, self).__init__()
        
        self.softmax = nn.Softmax(dim=1)
        self.n_branches = n_branches
        
        # embeddings --> scalar risk score output
        # dynamically fuse however many branches we are using
        self.attn_heads = nn.ModuleList([nn.Linear(embed_dim, 1, bias=False) for _ in range(n_branches)])
    
    
    def forward(self, h):
        # input is a list of embeddings from either 1, 2, or all 3 branches (ordered BINN, Gene, Clinical)
        
        # edge case: if attention fusion accidentally called on only 1 branch, skip
        if self.n_branches == 1:
            return h
        
        # generate a scalar risk score from each branch's embeddings independently
        

        
        # OLD EMBEDDING ATTENTION FUSION, TO BE REPLACED
        
        # embeddings --> scalar risk score output
        energies = []
        for head, hi in zip(self.attn_heads, h):
            ei = head(hi)
            energies.append(ei)
        
        # softmax
        energies = torch.cat(energies, dim=1)   # (batch, n_branches)
        a = self.softmax(energies)              # (batch, n_branches)
        
        z = 0                           # (batch, embed_dim)
        for i, hi in enumerate(h):
            # a[:,i] is (batch,) — unsqueeze to broadcast over embed_dim
            z += a[:,i:i+1] * hi        
        
        self.attn_weights = a   # store for interpretability
        return z
