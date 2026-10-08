"""Oracle supervision adapter; frozen NetSetQuery architecture is unchanged."""
import torch
from torch import nn
from pcbparse.core import assignment


def oracle_loss(outputs,pins,wires,joint=False,complete=True):
    p=outputs['pin_logits'][0];w=outputs['wire_logits'][0]
    selected=[(p,pins)]+([(w,wires)] if joint else [])
    groups=torch.unique(torch.cat([labels[labels>=0] for _,labels in selected]),sorted=True)
    q=p.shape[-1]-1
    if len(groups)>q:raise ValueError('Query overflow')
    if not len(groups):return p.sum()*0
    dice=0
    for logits,labels in selected:
        known=labels>=0
        if not known.any():continue
        prob=logits[known].softmax(-1)[:,:q];target=(labels[known,None]==groups[None]).to(prob.dtype)
        dice=dice+(2*(prob.T@target)+1e-6)/(prob.sum(0)[:,None]+target.sum(0)[None]+1e-6)
    pairs=assignment(dice.detach().cpu().tolist());mapping={int(groups[g]):slot for slot,g in pairs};loss=p.sum()*0
    for logits,labels in selected:
        known=labels>=0
        if known.any():
            aligned=torch.tensor([mapping[int(x)] for x in labels[known]],device=p.device)
            loss=loss+nn.functional.cross_entropy(logits[known],aligned)
    loss=loss+1-torch.stack([dice[slot,g] for slot,g in pairs]).mean()/len(selected)
    existence=outputs['net_existence_logits'][0];active=torch.zeros_like(existence)
    active[[slot for slot,_ in pairs]]=1
    if complete:
        loss=loss+.2*nn.functional.binary_cross_entropy_with_logits(existence,active)
    else:
        loss=loss+.1*nn.functional.softplus(-existence[active.bool()]).mean()
    return loss
