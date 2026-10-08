"""Contrast normalization and explicit signed stroke-orientation channels."""
import torch
from torch import nn
from experiments.factorized_circuit_reader import FactorizedCircuitReader


class ContrastOnlyReader(FactorizedCircuitReader):
    def canonical_contrast(self,image):
        baseline=image.float().flatten(1).median(1).values[:,None,None,None]
        return ((image-baseline)/(1-baseline).clamp_min(.1)).clamp(0,1)

    def forward(self,image):return super().forward(self.canonical_contrast(image))


class ContrastEdgeReader(ContrastOnlyReader):
    def __init__(self):
        super().__init__();self.encoder[0]=nn.Conv2d(3,16,3,2,1)
        self.register_buffer('sobel',torch.tensor([[[-1.,0,1],[-2,0,2],[-1,0,1]],
                                                 [[-1.,-2,-1],[0,0,0],[1,2,1]]])[:,None]/8)

    def forward(self,image):
        contrast=self.canonical_contrast(image)
        gradients=nn.functional.conv2d(nn.functional.pad(contrast,(1,1,1,1),mode='replicate'),self.sobel.to(contrast.dtype))
        return FactorizedCircuitReader.forward(self,torch.cat((contrast,gradients),1))
