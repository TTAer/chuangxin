"""Renderer-anchored whole-raster instruction decoding; no per-image GT boxes."""
import torch
from torch import nn
from experiments.circuit_program import CircuitProgramNet

class AnchoredCircuitProgramNet(CircuitProgramNet):
    def __init__(self):
        super().__init__();points=[[64.,64.],[64.,64.]]
        points += [[18.+18*j,18.+18*i] for i in range(6) for j in range(6)]
        points += [[18.+18*j,27.+18*i] for i in range(5) for j in range(6)]
        # Strided padded CNN centers occur at pixels 0,8,...,120 on 128 input.
        self.register_buffer('instruction_grid',torch.tensor(points).reshape(1,68,1,2)/120*2-1)
        self.local_projection=nn.Linear(96,96);self.local_head=nn.Linear(96,14)

    def encode(self,image):
        if image.shape[-2:]!=(128,128):raise ValueError('Current renderer grammar requires whole 128x128 raster')
        feature=self.encoder(image);h,w=feature.shape[-2:]
        y,x=torch.meshgrid(torch.linspace(-1,1,h,device=image.device),torch.linspace(-1,1,w,device=image.device),indexing='ij')
        coords=torch.stack((x,y),-1).reshape(1,h*w,2)
        memory=feature.flatten(2).transpose(1,2)+self.xy(coords)
        local=nn.functional.grid_sample(feature,self.instruction_grid.to(feature.dtype).expand(len(image),-1,-1,-1),align_corners=True).squeeze(-1).transpose(1,2)
        local=torch.cat((feature.mean((-2,-1))[:,None].expand(-1,2,-1),local[:,2:]),1)
        return memory,local

    def decode(self,memory,tokens):
        raster,local=memory
        query=self.token(tokens)+self.position+self.local_projection(local)
        return self.head(self.decoder(query,raster))+self.local_head(local)
