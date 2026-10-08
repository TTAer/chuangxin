"""Dense directional wire/pin/junction/ink perception; PyTorch training only."""
import torch
from torch import nn
import torch.nn.functional as F

class StripBlock(nn.Module):
    def __init__(self,channels,strip=True):
        super().__init__(); k=15 if strip else 3
        self.local=nn.Conv2d(channels,channels,3,padding=1,groups=channels)
        self.horizontal=nn.Conv2d(channels,channels,(1,k),padding=(0,k//2),groups=channels)
        self.vertical=nn.Conv2d(channels,channels,(k,1),padding=(k//2,0),groups=channels)
        self.gate=nn.Conv2d(channels,3,1)
        self.mix=nn.Sequential(nn.Conv2d(channels,channels,1),nn.GroupNorm(4,channels),nn.SiLU())
    def forward(self,x):
        gate=self.gate(x).softmax(1)
        y=self.local(x)*gate[:,0:1]+self.horizontal(x)*gate[:,1:2]+self.vertical(x)*gate[:,2:3]
        return x+self.mix(y)

def stage(a,b,strip):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),StripBlock(b,strip))

class DirectionalFieldNet(nn.Module):
    """Multi-scale anisotropic context, gated branch fusion, five dense heads."""
    def __init__(self,strip=True):
        super().__init__()
        self.first=stage(3,16,strip); self.second=stage(16,32,strip); self.third=stage(32,64,strip)
        self.decode2=stage(96,32,strip); self.decode1=stage(48,16,strip)
        self.wire_head=nn.Conv2d(16,2,1); self.keypoint_head=nn.Conv2d(16,2,1); self.ink_head=nn.Conv2d(16,1,1)
    def forward(self,x):
        a=self.first(x); b=self.second(F.avg_pool2d(a,2)); c=self.third(F.avg_pool2d(b,2))
        d=self.decode2(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1))
        e=self.decode1(torch.cat((F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
        return torch.cat((self.wire_head(e),self.keypoint_head(e),self.ink_head(e)),1)

def soft_skeleton(image,iterations=3):
    def erode(x): return torch.minimum(-F.max_pool2d(-x,(3,1),1,(1,0)),-F.max_pool2d(-x,(1,3),1,(0,1)))
    def opened(x): return F.max_pool2d(erode(x),3,1,1)
    skeleton=F.relu(image-opened(image))
    for _ in range(iterations):
        image=erode(image); delta=F.relu(image-opened(image)); skeleton=skeleton+F.relu(delta-skeleton*delta)
    return skeleton

def centerline_loss(probability,target):
    """Soft-clDice style continuity loss, not an electrical-topology guarantee."""
    predicted=soft_skeleton(probability); truth=soft_skeleton(target)
    dims=(1,2,3)
    precision=((predicted*target).sum(dims)+1)/(predicted.sum(dims)+1)
    recall=((truth*probability).sum(dims)+1)/(truth.sum(dims)+1)
    return 1-(2*precision*recall+1e-6)/(precision+recall+1e-6)
