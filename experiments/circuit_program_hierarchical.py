"""Image-only structural prefix, then renderer-anchored program denoising."""
import torch
from torch import nn
from experiments.circuit_program_anchored import AnchoredCircuitProgramNet

class HierarchicalCircuitProgramNet(AnchoredCircuitProgramNet):
    def __init__(self):
        super().__init__()
        self.horizontal_count=nn.Sequential(nn.Linear(96*16,64),nn.GELU(),nn.Linear(64,4))
        self.vertical_count=nn.Sequential(nn.Linear(96*16,64),nn.GELU(),nn.Linear(64,4))

    def encode(self,image):
        memory,local=super().encode(image);feature=self.encoder(image)
        counts=torch.stack((self.horizontal_count(feature.mean(-1).flatten(1)),self.vertical_count(feature.mean(-2).flatten(1))),1)
        return memory,local,counts

    def decode(self,memory,tokens):
        raster,local,counts=memory;logits=super().decode((raster,local),tokens)
        prefix=logits.new_full((len(tokens),2,14),-1e4);prefix[:,:,3:7]=counts
        return torch.cat((prefix,logits[:,2:]),1)
