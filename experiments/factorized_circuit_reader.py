"""Single-pass image reader with shared visual features and typed slot heads."""
import torch
from torch import nn


class FactorizedCircuitReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1,16,3,2,1), nn.GELU(),
                                     nn.Conv2d(16,32,3,1,1), nn.GELU())
        self.local = nn.Sequential(nn.Conv2d(32,24,3,2,1), nn.GELU(),
                                   nn.Conv2d(24,24,3,2,1), nn.GELU(), nn.Flatten(),
                                   nn.Linear(24*5*5,64), nn.GELU())
        self.crossing = nn.Linear(64,2)
        self.interval = nn.Linear(64,3)
        self.horizontal = nn.Sequential(nn.Linear(32*32,64), nn.GELU(), nn.Linear(64,4))
        self.vertical = nn.Sequential(nn.Linear(32*32,64), nn.GELU(), nn.Linear(64,4))
        centers = [[36+36*j,36+36*i] for i in range(6) for j in range(6)]
        centers += [[36+36*j,54+36*i] for i in range(5) for j in range(6)]
        offsets = torch.arange(-16,17,2)
        y,x = torch.meshgrid(offsets,offsets,indexing='ij')
        grid = torch.tensor(centers)[:,None,None,:]+torch.stack((x,y),-1)[None]
        self.register_buffer('grid',grid.float().reshape(1,66,17*17,2)/254*2-1)

    def forward(self,image):
        if image.shape[-2:] != (256,256): raise ValueError('Native 256px fixed grammar only')
        f = self.encoder(image); b = len(image)
        patch = nn.functional.grid_sample(f,self.grid.expand(b,-1,-1,-1).to(f.dtype),align_corners=True)
        patch = patch.permute(0,2,1,3).reshape(b*66,32,17,17)
        local = self.local(patch).reshape(b,66,64)
        h = self.horizontal(nn.functional.avg_pool1d(f.mean(-1),4).flatten(1))
        v = self.vertical(nn.functional.avg_pool1d(f.mean(-2),4).flatten(1))
        return torch.stack((h,v),1), self.crossing(local[:,:36]), self.interval(local[:,36:])

    @torch.no_grad()
    def generate(self,image):
        counts,cross,interval = self(image); counts=counts.argmax(-1)+3
        result = torch.full((len(image),68),7,dtype=torch.long,device=image.device)
        result[:,:2]=counts
        for i in range(6):
            for j in range(6):
                active=(i<counts[:,0])&(j<counts[:,1]);k=i*6+j
                result[:,2+k]=torch.where(active,cross[:,k].argmax(-1)+8,7)
        for i in range(5):
            for j in range(6):
                active=(i<counts[:,0]-1)&(j<counts[:,1]);k=i*6+j
                result[:,38+k]=torch.where(active,interval[:,k].argmax(-1)+10,7)
        return result
