"""Learned local topology features fused with geometric arm lengths."""
import torch
from torch import nn


class TopologyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.appearance=nn.Sequential(nn.Conv2d(6,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.GroupNorm(4,32),nn.SiLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.GroupNorm(4,64),nn.SiLU(),nn.AdaptiveAvgPool2d(1))
        self.geometry=nn.Sequential(nn.Linear(4,16),nn.SiLU())
        self.fusion=nn.Sequential(nn.Linear(80,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,patch,geometry):
        return self.fusion(torch.cat((self.appearance(patch).flatten(1),self.geometry(geometry)),1))
