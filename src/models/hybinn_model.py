import torch.nn as nn
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
        
        # branches = {'binn': BINNBranch(...), 'clinical': ClinicalBranch(...)}
        self.branches = nn.ModuleDict(branches)
        
        self.attention_fusion = AttentionFusion(embed_dim=embedding_nodes,
                                                n_branches=len(branches.keys()))
        self.survival_head = SurvivalHead(in_nodes=embedding_nodes)
    
    
    def forward(self, x_mapped, x_unmapped, x_clinical):
        
        # propogate along whichever branches were used
        branches_out = []
        for branch in self.branches.values():
            branches_out.append(branch(x_mapped, x_unmapped, x_clinical))
        
        # fusion
        z   = self.attention_fusion(branches_out)
        out = self.survival_head(z)
        
        return out