"""Image-only separable wire-union residual; symmetric diagonal evidence, not certainty."""
import numpy as np


def residual_scores(patches):
    p=np.asarray(patches,dtype=np.float32)[:,0]
    arms=[2,3,4,5,27,28,29,30]
    h=np.median(p[:,:,arms],axis=2)
    v=np.quantile(p[:,arms,:],.25,axis=1)
    predicted=h[:,:,None]+v[:,None,:]-h[:,:,None]*v[:,None,:]
    extra=np.maximum(p-predicted,0)
    axis=np.arange(-16,17);y,x=np.meshgrid(axis,axis,indexing='ij')
    # Absolute ink cutoff avoids estimating broad wires from low background noise.
    hx=np.max(np.where((v>.25)&(abs(axis)[None]<=8),abs(axis)[None],0),axis=1)
    hy=np.max(np.where((h>.25)&(abs(axis)[None]<=8),abs(axis)[None],0),axis=1)
    scores=[]
    for sx,sy in ((-1,-1),(-1,1),(1,-1),(1,1)):
        mask=(sx*x[None]>=np.maximum(2,hx+1)[:,None,None])&(sy*y[None]>=np.maximum(2,hy+1)[:,None,None])&((x*x+y*y)[None]<=121)
        count=mask.sum((1,2));value=(extra*mask).sum((1,2))/np.maximum(1,count)
        scores.append(np.where(count>=3,value,np.nan))
    return np.min(np.stack(scores,axis=1),axis=1)


def decisions(scores,low,high):
    return np.where(scores>=high,9,np.where(scores<=low,8,7)).astype(np.int64)
