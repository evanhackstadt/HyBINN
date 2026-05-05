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
    
    def __init__(self,
                 binn_in_nodes, binn_pathway_nodes, binn_hidden_nodes, pathway_mask,
                 gene_in_nodes, gene_hidden_nodes_1, gene_hidden_nodes_2,
                 clinical_in_nodes, clinical_hidden_nodes_1, clinical_hidden_nodes_2,
                 embedding_nodes,
                 out_nodes):
        super(HyBINN, self).__init__()
        
        self.binn_branch = BINNBranch(binn_in_nodes, binn_pathway_nodes,
                                      binn_hidden_nodes, out_nodes, pathway_mask)
        self.gene_branch = GeneBranch(gene_in_nodes, gene_hidden_nodes_1,
                                      gene_hidden_nodes_2, out_nodes)
        self.clinical_branch = ClinicalBranch(clinical_in_nodes, clinical_hidden_nodes_1, 
                                              clinical_hidden_nodes_2, out_nodes)
        self.attention_fusion = AttentionFusion(embedding_nodes)
        self.survival_head = SurvivalHead(embedding_nodes)
    
    
    def forward(self, x_mapped, x_unmapped, x_clinical):
        # propogate along branches
        h_binn     = self.binn_branch(x_mapped, x_unmapped)
        h_gene     = self.gene_branch(x_mapped, x_unmapped)
        h_clinical = self.clinical_branch(x_clinical)
        
        # fusion
        z   = self.attention_fusion(h_binn, h_gene, h_clinical)
        out = self.survival_head(z)
        
        return out