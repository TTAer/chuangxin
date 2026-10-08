"""Preserve two-dimensional local evidence before learned downsampling."""
from torch import nn


class SpatialEvidenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.GELU(),nn.Conv2d(8,8,3,padding=1),nn.GELU(),
                                    nn.Conv2d(8,16,3,stride=2,padding=1),nn.GELU(),nn.Conv2d(16,16,3,stride=2,padding=1),nn.GELU(),
                                    nn.AdaptiveAvgPool2d((5,5)),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(400,32),nn.GELU(),nn.Linear(32,2))
    def forward(self,patches):return self.head(self.features(patches))
