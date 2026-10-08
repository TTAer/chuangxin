"""Whole-page pin/net set prediction prototype, separate from frozen runtime."""
import math
import torch
from torch import nn
from pcbparse.core import assignment


class NetSetQueryDecoder(nn.Module):
    def __init__(self,queries=128,width=64):
        super().__init__();self.queries=queries;self.width=width
        self.input=nn.Linear(12,width);self.kind=nn.Embedding(2,width)
        layer=nn.TransformerEncoderLayer(width,4,width*2,dropout=0,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.query=nn.Parameter(torch.randn(queries,width)*.02)
        decoder=nn.TransformerDecoderLayer(width,4,width*2,dropout=0,batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(decoder,2)
        self.query_key=nn.Linear(width,width);self.pin_key=nn.Linear(width,width);self.wire_key=nn.Linear(width,width)
        self.abstain=nn.Linear(width,1);self.existence=nn.Linear(width,1)

    def forward(self,pins,wires):
        if pins.shape[1]==0:raise ValueError('At least one pin token is required')
        if pins.shape[0]!=wires.shape[0]:raise ValueError('Batch mismatch')
        memory=torch.cat([self.input(pins)+self.kind.weight[0],self.input(wires)+self.kind.weight[1]],1)
        memory=self.encoder(memory)
        query=self.decoder(self.query[None].expand(pins.shape[0],-1,-1),memory)
        q=self.query_key(query);p=memory[:,:pins.shape[1]];w=memory[:,pins.shape[1]:]
        def head(tokens,key):
            scores=torch.einsum('bnd,bqd->bnq',key(tokens),q)/math.sqrt(self.width)
            return torch.cat([scores,self.abstain(tokens)],-1)
        return {'pin_logits':head(p,self.pin_key),'wire_logits':head(w,self.wire_key),
                'net_existence_logits':self.existence(query).squeeze(-1)}


def partial_net_loss(outputs,labels):
    """Hungarian-matched partial pin-set loss; -1 is unknown, never disconnected.

    Unknown membership and unobserved nets cannot supervise empty queries.
    Wire ownership is not supervised by this pin-only prototype loss.
    """
    logits=outputs['pin_logits'];losses=[]
    for b in range(len(logits)):
        known=labels[b]>=0
        if not known.any():continue
        groups=torch.unique(labels[b,known],sorted=True)
        q=logits.shape[-1]-1
        if len(groups)>q:raise ValueError('More observed target nets than query slots; no silent truncation')
        probs=logits[b,known].softmax(-1)[:,:q]
        target=(labels[b,known,None]==groups[None]).to(probs.dtype)
        intersection=probs.T@target
        dice=(2*intersection+1e-6)/(probs.sum(0)[:,None]+target.sum(0)[None]+1e-6)
        pairs=assignment(dice.detach().cpu().tolist())
        mapping={int(groups[g]):slot for slot,g in pairs}
        aligned=torch.tensor([mapping[int(x)] for x in labels[b,known]],device=logits.device)
        ce=nn.functional.cross_entropy(logits[b,known],aligned)
        selected=torch.stack([dice[slot,g] for slot,g in pairs])
        # Positive existence only: missing supervision must not label empty slots.
        existence=outputs['net_existence_logits'][b,[slot for slot,_ in pairs]]
        losses.append(ce+1-selected.mean()+.1*nn.functional.softplus(-existence).mean())
    return torch.stack(losses).mean() if losses else logits.sum()*0
