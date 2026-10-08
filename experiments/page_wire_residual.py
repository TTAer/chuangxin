"""Estimate clean wire profiles from page overhangs, avoiding local component arms."""
import numpy as np,cv2
from experiments.radial_evidence_net import crossing_patches


def page_scores(raw,aligned,geometry):
    patches,slots=crossing_patches(raw,aligned,geometry,'raw')
    if not slots:return np.empty(0),[]
    rows=geometry['observed_rows'];cols=geometry['observed_columns'];sx=float(np.median(np.diff(cols))/36);sy=float(np.median(np.diff(rows))/36)
    bg=float(np.median(raw));ink=np.clip((bg-raw.astype(np.float32))/max(1,bg),0,1);axis=np.arange(-16,17,dtype=np.float32)
    hs=[];vs=[]
    # Full-length overhang is a renderer assumption, not a generic schematic guarantee.
    for y in rows:
        xx=np.broadcast_to((cols[-1]+np.array([13,14,15],np.float32)*sx)[None],(33,3)).copy();yy=np.broadcast_to((y+axis*sy)[:,None],(33,3)).copy()
        hs.append(cv2.remap(ink,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0).mean(1))
    for x in cols:
        yy=np.broadcast_to((rows[-1]+np.array([13,14,15],np.float32)*sy)[:,None],(3,33)).copy();xx=np.broadcast_to((x+axis*sx)[None],(3,33)).copy()
        vs.append(cv2.remap(ink,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0).mean(0))
    h=np.median(hs,axis=0);v=np.median(vs,axis=0);predicted=h[:,None]+v[None,:]-h[:,None]*v[None,:]
    extra=np.maximum(patches[:,0]-predicted[None],0);a=np.arange(-16,17);y,x=np.meshgrid(a,a,indexing='ij')
    hx=int(np.max(np.where((v>.25)&(abs(a)<=8),abs(a),0)));hy=int(np.max(np.where((h>.25)&(abs(a)<=8),abs(a),0)))
    values=[]
    for sign_x,sign_y in ((-1,-1),(-1,1),(1,-1),(1,1)):
        mask=(sign_x*x>=max(2,hx+1))&(sign_y*y>=max(2,hy+1))&(x*x+y*y<=121)
        values.append((extra*mask[None]).sum((1,2))/mask.sum() if mask.sum()>=3 else np.full(len(slots),np.nan))
    return np.min(np.stack(values,axis=1),axis=1),slots
