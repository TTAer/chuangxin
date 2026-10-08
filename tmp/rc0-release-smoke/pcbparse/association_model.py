"""Relation-specific message passing over pin/component/wire-group factors."""
import torch
from torch import nn
from .association_features import NODE_FEATURES, EDGE_FEATURES

class ElectricalAssociationNet(nn.Module):
    def __init__(self, hidden=48, layers=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(NODE_FEATURES,hidden),nn.SiLU())
        self.updates = nn.ModuleList(nn.Sequential(nn.Linear(hidden*3,hidden),nn.SiLU(),nn.LayerNorm(hidden)) for _ in range(layers))
        self.decoder = nn.Sequential(nn.Linear(hidden*2+EDGE_FEATURES,hidden),nn.SiLU(),nn.Linear(hidden,1))
    def forward(self,nodes,component_relation,incidence_relation,pairs,edge_features):
        h = self.encoder(nodes)
        for update in self.updates:
            h = h + update(torch.cat((h, component_relation @ h, incidence_relation @ h),dim=1))
        return self.decoder(torch.cat((h[pairs[:,0]],h[pairs[:,1]],edge_features),dim=1)).flatten()
