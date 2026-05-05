import torch.nn as nn


# SurvivalHead - 64-node embedding output from BINN/gene branch --> 1 scalar risk score output
class SurvivalHead(nn.Module):
    
    def __init__(self, in_nodes):
        super(SurvivalHead, self).__init__()
        
        # input embedding --> output scalar risk score
        self.fc = nn.Linear(in_nodes, 1, bias=False)
        self.fc.weight.data.uniform_(-0.001, 0.001)    # start CoxPH weights close to 0
    
    
    def forward(self, x):
        out = self.fc(x)
        return out