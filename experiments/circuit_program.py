"""Whole-raster conditioned masked executable-program denoiser (fresh weights)."""
import torch
from torch import nn
import math

LENGTH=68;MASK=13

class CircuitProgramNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(1,24,3,2,1),nn.GELU(),nn.Conv2d(24,48,3,2,1),nn.GELU(),nn.Conv2d(48,96,3,2,1),nn.GELU())
        self.xy=nn.Linear(2,96);self.token=nn.Embedding(14,96);self.position=nn.Parameter(torch.randn(1,LENGTH,96)*.02)
        self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(96,4,192,.1,batch_first=True),2)
        self.head=nn.Linear(96,14);self.projection=nn.Sequential(nn.Linear(96,96),nn.GELU(),nn.Linear(96,64))

    def encode(self,image):
        feature=self.encoder(image);h,w=feature.shape[-2:]
        y,x=torch.meshgrid(torch.linspace(-1,1,h,device=image.device),torch.linspace(-1,1,w,device=image.device),indexing='ij')
        coordinates=torch.stack((x,y),-1).reshape(1,h*w,2)
        return feature.flatten(2).transpose(1,2)+self.xy(coordinates)

    def forward(self,image,tokens):
        return self.decode(self.encode(image),tokens)

    def decode(self,memory,tokens):
        return self.head(self.decoder(self.token(tokens)+self.position,memory))

    @torch.no_grad()
    def generate(self,image,steps=6):
        memory=self.encode(image);b=len(image);tokens=torch.full((b,LENGTH),MASK,dtype=torch.long,device=image.device)
        for step in range(steps):
            logits=self.decode(memory,tokens).float();legal=torch.zeros_like(logits,dtype=torch.bool);legal[:,:2,3:7]=True
            counts=logits[:,:2,3:7].argmax(-1)+3
            if step:counts=tokens[:,:2]
            for i in range(6):
                for j in range(6):
                    active=(i<counts[:,0])&(j<counts[:,1]);slot=2+i*6+j
                    legal[:,slot,7]=~active;legal[:,slot,8]=active;legal[:,slot,9]=active
            for i in range(5):
                for j in range(6):
                    active=(i<counts[:,0]-1)&(j<counts[:,1]);slot=38+i*6+j
                    legal[:,slot,7]=~active
                    for code in (10,11,12):legal[:,slot,code]=active
            probability=logits.masked_fill(~legal,-1e4).softmax(-1);confidence,pred=probability.max(-1)
            if step:pred=torch.where(tokens!=MASK,tokens,pred)
            confidence[:,:2]=2;confidence[tokens!=MASK]=3
            keep=min(LENGTH,math.ceil(LENGTH*(step+1)/steps));indices=confidence.topk(keep,dim=1).indices
            tokens.fill_(MASK);tokens.scatter_(1,indices,pred.gather(1,indices))
        return tokens
