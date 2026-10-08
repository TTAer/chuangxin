"""SideStripNet: normal-context compression followed by dilated axial inference."""
import torch
from torch import nn

class SideStripNet(nn.Module):
    def __init__(self,dilated=True):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),
            nn.Conv2d(16,24,(3,7),padding=(1,3),groups=8),nn.GroupNorm(4,24),nn.SiLU())
        self.compress=nn.Sequential(nn.Conv1d(24*8,32,1),nn.SiLU())
        self.axial=nn.Sequential(*[nn.Sequential(nn.Conv1d(32,32,5,padding=2*d,dilation=d,groups=32),
            nn.Conv1d(32,32,1),nn.GroupNorm(4,32),nn.SiLU()) for d in ((1,2,4) if dilated else (1,1,1))])
        self.head=nn.Conv1d(32,2,1)

    def forward(self,x):
        x=self.encoder(x)
        # Exactly retain 8 normal bins; axial resolution remains one source pixel.
        x=x.reshape(x.shape[0],24,8,8,256).mean(dim=3).flatten(1,2)
        return self.head(self.axial(self.compress(x)))
