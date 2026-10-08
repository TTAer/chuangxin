"""Coordinate-conditioned dual heatmap network (training imports PyTorch only)."""
import torch
from torch import nn
import torch.nn.functional as F


def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU(),
                         nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU())


class PinHeatmapNet(nn.Module):
    """RGB + symbol mask + local coordinates; tip/base heads share an encoder."""
    def __init__(self):
        super().__init__()
        self.enc1 = block(6, 16)
        self.enc2 = block(16, 32)
        self.enc3 = block(32, 64)
        self.dec2 = block(96, 32)
        self.dec1 = block(48, 16)
        self.head = nn.Conv2d(16, 2, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.enc3(F.avg_pool2d(b, 2))
        d = self.dec2(torch.cat((F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b), 1))
        e = self.dec1(torch.cat((F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a), 1))
        return self.head(e)
