"""Conservative component-relation bounds over observed attachment candidates.

Bounds are conditional on candidate completeness and correct physical relation.
No component-pair closure and no target-informed candidate selection.
"""
import torch

def component_bounds(wire_relation,candidates,candidate_mask,port_mask,candidate_complete=None):
    # [B,N,N], [B,C,P,K] wire IDs; masks are binary. Missing evidence -> unknown.
    if candidates.ndim!=4 or candidate_mask.shape!=candidates.shape or port_mask.shape!=candidates.shape[:3]:raise ValueError('Invalid candidate shapes')
    b,c,p,k=candidates.shape;n=wire_relation.shape[-1]
    if p<1 or k<1 or candidates.dtype not in (torch.int32,torch.int64):raise ValueError('Nonempty integer candidate tensor required')
    if not torch.isfinite(wire_relation).all() or (wire_relation<0).any() or (wire_relation>1).any():raise ValueError('Wire relation must be finite in [0,1]')
    if wire_relation.shape!=(b,n,n) or candidates.min()<0 or candidates.max()>=n:raise ValueError('Invalid wire relation or candidate IDs')
    if not torch.all((candidate_mask==0)|(candidate_mask==1)) or not torch.all((port_mask==0)|(port_mask==1)):raise ValueError('Masks must be binary')
    if candidate_complete is None:candidate_complete=torch.zeros_like(port_mask)
    if candidate_complete.shape!=port_mask.shape or not torch.all((candidate_complete==0)|(candidate_complete==1)):raise ValueError('Invalid completeness evidence mask')
    lower=wire_relation.new_zeros((b,c,c));upper=lower.clone()
    for pi in range(p):
        for pj in range(p):
            left=candidates[:,:,pi,:];right=candidates[:,:,pj,:]
            batch=torch.arange(b,device=left.device)[:,None,None,None,None]
            values=wire_relation[batch,left[:,:,None,:,None],right[:,None,:,None,:]]
            valid=candidate_mask[:,:,pi,None,:,None]*candidate_mask[:,None,:,pj,None,:]
            active=port_mask[:,:,pi,None]*port_mask[:,None,:,pj]
            has=valid.bool().any((-2,-1))
            complete=candidate_complete[:,:,pi,None].bool()&candidate_complete[:,None,:,pj].bool()
            has=has&complete
            minimum=values.masked_fill(~valid.bool(),1).amin((-2,-1));maximum=values.masked_fill(~valid.bool(),0).amax((-2,-1))
            low=torch.where(has,minimum,0)*active;high=torch.where(has,maximum,1)*active
            lower=torch.maximum(lower,low);upper=torch.maximum(upper,high)
    # A component with no observable port cannot have a justified negative.
    exists=port_mask.bool().any(-1);unknown=(~exists[:,:,None])|(~exists[:,None,:]);upper=torch.where(unknown,1,upper)
    return lower,upper
