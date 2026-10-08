"""PortLift: max-product wire paths -> multi-port component incidence.

Device ports are observations, never edges through the device. Component
shared-net relation is decoded directly, NOT with a connected component union.
"""
import torch
from torch import nn

class PortLiftNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(1,12,3,padding=1),nn.GELU(),nn.Conv2d(12,20,3,stride=2,padding=1),nn.GELU(),
                                   nn.Conv2d(20,24,3,stride=2,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(24,32),nn.GELU(),nn.Linear(32,1))

    def forward(self,patches,edges,mask,ports):
        b,e=patches.shape[:2];flat=patches.flatten(0,1)
        orbit=torch.cat([torch.rot90(flat,k,(-2,-1)) for k in range(4)])
        logits=self.encoder(orbit).reshape(4,b,e).mean(0);g=logits.sigmoid()*mask
        left=nn.functional.one_hot(edges[:,:,0],12).to(g.dtype);right=nn.functional.one_hot(edges[:,:,1],12).to(g.dtype)
        a=torch.einsum('be,bei,bej->bij',g,left,right);a=a+a.transpose(1,2)
        relation=a+torch.eye(12,device=g.device)[None]
        # Floyd-style maximum path-product; no grounding inverse or learned net slots.
        for k in range(12):relation=torch.maximum(relation,relation[:,:,k,None]*relation[:,None,k,:])
        onehot=nn.functional.one_hot(ports,12).to(g.dtype)
        incidence=torch.einsum('bcpi,bij,bdqj->bcpdq',onehot,relation,onehot).amax((2,4))
        return relation,incidence,logits

def balanced_bce(pred,truth,mask):
    p=pred.clamp(1e-6,1-1e-6);pos=truth*mask;neg=(1-truth)*mask
    return -(pos*p.log()).sum()/pos.sum().clamp_min(1)-(neg*(1-p).log()).sum()/neg.sum().clamp_min(1)
