"""Learn radial ink profiles with permutation-invariant angular evidence aggregation."""
import torch
from torch import nn
import numpy as np,cv2


class RadialEvidenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        y,x=torch.meshgrid(torch.arange(-16,17),torch.arange(-16,17),indexing='ij')
        radius=(x.float().square()+y.float().square()).sqrt()
        angle=torch.remainder(torch.atan2(y.float(),x.float())+torch.pi/8,2*torch.pi)
        sectors=(angle/(torch.pi/4)).long().clamp_max(7)
        edges=[0,2,4,6,8,10,12,14,17];masks=[]
        for s in range(8):
            for lo,hi in zip(edges[:-1],edges[1:]):
                m=((sectors==s)&(radius>=lo)&(radius<hi)&(radius>0)).float();masks.append(m/max(1,float(m.sum())))
        self.register_buffer('masks',torch.stack(masks).flatten(1))
        core=(radius<2).float();self.register_buffer('core_mask',core.flatten()/core.sum())
        self.sector=nn.Sequential(nn.Linear(8,32),nn.GELU(),nn.Linear(32,32),nn.GELU())
        self.classifier=nn.Sequential(nn.Linear(65,32),nn.GELU(),nn.Linear(32,2))

    def forward(self,patches):
        evidence=(patches.flatten(1).float()@self.masks.T).reshape(-1,8,8)
        f=self.sector(evidence)
        core=patches.flatten(1).float()@self.core_mask
        return self.classifier(torch.cat((f.mean(1),f.amax(1),core[:,None]),1))


def crossing_patches(raw,aligned,geometry,mode):
    """Observed centers only; no GT count, position or transform. Return stable slot identities."""
    if not geometry['ready']:return np.empty((0,1,33,33),np.float32),[]
    rows=geometry['observed_rows'];cols=geometry['observed_columns']
    sx=float(np.median(np.diff(cols))/36);sy=float(np.median(np.diff(rows))/36)
    im=raw if mode=='raw' else aligned
    bg=float(np.median(im));ink=np.clip((bg-im.astype(np.float32))/max(1,bg),0,1)
    axis=np.arange(-16,17,dtype=np.float32);patches=[];slots=[]
    for i,y in enumerate(rows):
        for j,x in enumerate(cols):
            if mode=='canonical':x,y=36+36*j,36+36*i;scale_x=scale_y=1.
            else:scale_x,scale_y=sx,sy
            xx=np.broadcast_to((x+axis*scale_x)[None],(33,33)).copy();yy=np.broadcast_to((y+axis*scale_y)[:,None],(33,33)).copy()
            patches.append(cv2.remap(ink,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)[None]);slots.append(2+i*6+j)
    return np.stack(patches),slots
