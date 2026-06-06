import torch
import torch.nn as nn
import numpy as np
from models.binn_branch import BINNBranch
from models.gene_branch import GeneBranch
from models.clinical_branch import ClinicalBranch
from models.attention_fusion import AttentionFusion
from models.survival_head import SurvivalHead


# HyBINN - assembles branches, performs attention-based fusion, outputs survival
'''
genes, clinical data → binn, gene, clinical branches 
                     → attention fusion 
                     → survival head
'''
class HyBINN(nn.Module):
    
    def __init__(self, branches: dict, embedding_nodes):
        super(HyBINN, self).__init__()
        
        self.emb_dim = embedding_nodes
        self.n_branches = len(branches.keys())
        
        # branches = {'binn': BINNBranch(...), 'gene': GeneBranch(...), 'clinical': ClinicalBranch(...)}
        self.branches = nn.ModuleDict(branches)
        
        # Learned combination weights — initialized to equal weighting
        # Softmax ensures they sum to 1 and stay positive
        self.fusion_logits = nn.Parameter(torch.zeros(self.n_branches))
        
        self.attention_fusion = AttentionFusion(embed_dim=embedding_nodes,
                                                n_branches=self.n_branches)
        self.survival_head = SurvivalHead(in_nodes=embedding_nodes)
    
    
    def forward(self, x_mapped, x_unmapped, x_clinical):
        
        # Propogate along whichever branches were used
        branches_out = []
        for branch in self.branches.values():
            branches_out.append(branch(x_mapped, x_unmapped, x_clinical))
        
        # Fusion 
        
        # if branches --> risk scores
        if self.emb_dim == 1:
            # Population-level weights (same for all patients)
            weights = torch.softmax(self.fusion_logits, dim=0)
        
            r_final = 0
            for weight, risk_score in zip(weights, branches_out):
                r_final += weight*risk_score
            
            # Store for interpretability
            self.risk_scores = [r.detach() for r in branches_out]
            self.weights     = weights.detach()
            
            return r_final
        
        # if branches --> embeddings
        else:
            z   = self.attention_fusion(branches_out)
            out = self.survival_head(z)
            return out