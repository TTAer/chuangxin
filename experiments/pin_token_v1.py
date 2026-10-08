"""Experimental token role + endpoint association; never imported by RC0."""
import re
import numpy as np

ROLES=('designator','pin_number','pin_name','component_model','net_label','other')
K=8

def text_features(text):
    text=str(text); n=max(1,len(text))
    return np.array([min(n,32)/32,sum(c.isdigit() for c in text)/n,
        sum(c.isalpha() for c in text)/n,float(text.isdigit()),float(text.isalpha()),
        float(bool(re.fullmatch(r'(?:U|IC|R|C|Q|J|P|D|LED)\d+',text,re.I))),
        float('/' in text),float('_' in text),float('+' in text),float('-' in text),
        float(text.upper() in ('GND','VCC','VDD','VSS')),float(text.upper()==text)],np.float32)

def candidates(token_box,body,points):
    """Image-coordinate geometry; candidate selection never reads identity labels."""
    center=np.array([(token_box[0]+token_box[2])/2,(token_box[1]+token_box[3])/2])
    points=np.asarray(points,np.float32); width=max(1,body[2]-body[0]); height=max(1,body[3]-body[1])
    distances=np.linalg.norm(points-center,axis=1)
    order=np.argsort(distances,kind='stable')[:K]
    features=np.zeros((K,12),np.float32); valid=np.zeros(K,bool)
    for j,i in enumerate(order):
        x,y=points[i]; dx,dy=center-points[i]
        features[j]=[dx/width,dy/height,abs(dx)/width,abs(dy)/height,
            distances[i]/max(width,height),(x-body[0])/width,(y-body[1])/height,
            (center[0]-body[0])/width,(center[1]-body[1])/height,
            (token_box[2]-token_box[0])/width,(token_box[3]-token_box[1])/height,
            float(body[0]<=center[0]<=body[2] and body[1]<=center[1]<=body[3])]
        valid[j]=True
    return order,features,valid

def make_model(visual=True):
    import torch
    from torch import nn
    class PinTokenNet(nn.Module):
        def __init__(self):
            super().__init__(); self.visual=visual
            self.encoder=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
            self.token=nn.Sequential(nn.Linear(44 if visual else 12,48),nn.ReLU())
            self.role=nn.Linear(48,len(ROLES))
            self.affinity=nn.Sequential(nn.Linear(60,48),nn.ReLU(),nn.Linear(48,1))
            self.null=nn.Linear(48,1)
        def forward(self,patch,text,pairs,valid):
            embedding=self.encoder(patch).flatten(1) if self.visual else None
            token=self.token(torch.cat([embedding,text],1) if self.visual else text)
            role=self.role(token)
            scores=self.affinity(torch.cat([token[:,None,:].expand(-1,K,-1),pairs],2)).squeeze(-1)
            scores=scores.masked_fill(~valid,-1e4)
            return role,torch.cat([scores,self.null(token)],1)
    return PinTokenNet()
