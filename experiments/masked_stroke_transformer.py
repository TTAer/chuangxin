"""Patch-token masked reconstruction encoder, then direct legal-partition head."""
import torch
from torch import nn


class MaskedStrokeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Linear(3*8*8,64)
        self.position=nn.Parameter(torch.randn(1,36,64)*.02)
        self.mask_token=nn.Parameter(torch.zeros(1,1,64))
        layer=nn.TransformerEncoderLayer(64,4,128,dropout=.1,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(64)
        self.reconstruction=nn.Linear(64,64)
        self.partition=nn.Linear(64,15)

    def patches(self,image):
        return image.reshape(-1,3,6,8,6,8).permute(0,2,4,1,3,5).reshape(-1,36,192)

    def encode(self,image,mask=None):
        tokens=self.embedding(self.patches(image))
        if mask is not None:tokens=torch.where(mask[:,:,None],self.mask_token,tokens)
        return self.norm(self.encoder(tokens+self.position))

    def forward(self,image):
        return self.partition(self.encode(image).mean(1))

    def reconstruct(self,image,mask):
        return self.reconstruction(self.encode(image,mask))
