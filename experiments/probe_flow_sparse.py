"""R2: rotation-invariant observed diffusivities with exact zero conductance."""
import torch
from torch import nn
from experiments.probe_flow import ProbeFlowNet


class SparseProbeFlowNet(ProbeFlowNet):
    def forward(self,patches,edges,edge_mask):
        batch,count=patches.shape[:2];flat=patches.flatten(0,1)
        # Average orientation orbits before conductivity inference. All four
        # directions are treated identically, without a learned orientation ID.
        orbit=torch.cat([torch.rot90(flat,k,(-2,-1)) for k in range(4)])
        features=self.encoder(orbit).reshape(4,batch*count,-1).mean(0)
        logits=self.conductance(features).reshape(batch,count)
        positive=logits.relu().square()
        conductance=positive/(1+positive)*edge_mask
        n=12;left=nn.functional.one_hot(edges[:,:,0],n).to(patches.dtype);right=nn.functional.one_hot(edges[:,:,1],n).to(patches.dtype)
        a=torch.einsum('be,bei,bej->bij',conductance,left,right);a=a+a.transpose(1,2)
        laplacian=torch.diag_embed(a.sum(-1))-a
        green=torch.linalg.inv(laplacian+.003*torch.eye(n,device=a.device)[None])
        norm=green.diagonal(dim1=1,dim2=2).clamp_min(1e-6).sqrt()
        response=(green/(norm[:,:,None]*norm[:,None,:])).clamp(1e-6,1-1e-6)
        return response,logits
