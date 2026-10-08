"""Visual diffusivities -> grounded Laplacian Green response; no net slots."""
import torch
from torch import nn


class ProbeFlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.GELU(),nn.Conv2d(16,24,3,stride=2,padding=1),nn.GELU(),
                                   nn.Conv2d(24,32,3,stride=2,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,48),nn.GELU())
        self.projection=nn.Sequential(nn.Linear(48,48),nn.GELU(),nn.Linear(48,32))
        self.conductance=nn.Linear(48,1)

    def forward(self,patches,edges,edge_mask):
        batch,count=patches.shape[:2]
        logits=self.conductance(self.encoder(patches.flatten(0,1))).reshape(batch,count)
        conductance=logits.sigmoid().pow(4)*edge_mask
        # 12 observed physical wire runs at most in this procedural feasibility bank.
        n=12;left=nn.functional.one_hot(edges[:,:,0],n).to(patches.dtype);right=nn.functional.one_hot(edges[:,:,1],n).to(patches.dtype)
        a=torch.einsum('be,bei,bej->bij',conductance,left,right);a=a+a.transpose(1,2)
        laplacian=torch.diag_embed(a.sum(-1))-a
        green=torch.linalg.inv(laplacian+.003*torch.eye(n,device=a.device)[None])
        norm=green.diagonal(dim1=1,dim2=2).clamp_min(1e-6).sqrt()
        response=(green/(norm[:,:,None]*norm[:,None,:])).clamp(1e-6,1-1e-6)
        return response,logits


def partition_loss(response,truth,valid):
    n=valid.shape[1];mask=valid[:,:,None]*valid[:,None,:]*(1-torch.eye(n,device=valid.device)[None])
    pos=mask*truth;neg=mask*(1-truth)
    # Separately balanced positive/negative reachability; singleton-only pages
    # contribute negative evidence, never fabricated positive pairs.
    positive=-(response.log()*pos).sum()/pos.sum().clamp_min(1)
    negative=-((1-response).log()*neg).sum()/neg.sum().clamp_min(1)
    return positive+negative
