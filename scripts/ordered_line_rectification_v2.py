"""V2 lower relative prominence, preserves absolute noise floor. Bilateral ordered line observations and monotone rectification without uniform spacing."""
import numpy as np,cv2
from scripts.circuit_geometry_guard_v2 import register_guarded_v2


def ordered_peaks(profile):
    smooth=cv2.GaussianBlur(profile.astype(np.float32).reshape(-1,1),(1,3),.5).ravel()
    ids=np.flatnonzero(smooth>max(.05,float(smooth.max())*.10))
    if not len(ids):return None
    groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
    groups=[g for g in groups if len(g)<=15]
    if not 3<=len(groups)<=6:return None
    centers=np.array([np.average(g,weights=smooth[g]) for g in groups])
    if np.any(np.diff(centers)<12) or np.any(np.diff(centers)>60):return None
    return centers


def register_ordered(image):
    if image.shape!=(256,256):raise ValueError('Whole256 raster only')
    _,previous=register_guarded_v2(image)
    orientation=previous.get('orientation',[])
    witnessed=[d for d in orientation if d.get('witnessed')]
    if not witnessed or any(not d.get('orthogonal',False) for d in witnessed):
        return image.copy(),{'ready':False,'reason':'direction evidence not orthogonal','orientation':orientation}
    bg=float(np.median(image));ink=np.clip((bg-image.astype(float))/max(1,bg),0,1)
    soft=cv2.GaussianBlur(ink.astype(np.float32),(3,3),.7)
    bounds=[];ys,xs=np.where(ink>.4)
    if len(xs)>=20:bounds.append((int(xs.min()),int(xs.max()),int(ys.min()),int(ys.max()),'strong'))
    weak=cv2.morphologyEx((soft>.1).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(weak,8)
    boxes=[s for s in stats[1:] if s[4]>=25 and max(s[2],s[3])>=25]
    if boxes:bounds.append((min(s[0] for s in boxes),max(s[0]+s[2]-1 for s in boxes),min(s[1] for s in boxes),max(s[1]+s[3]-1 for s in boxes),'weak'))
    def locate(lo,hi,rows):
        sides=[]
        for side in ('near','far'):
            candidates=[]
            for a,b in ((.015,.035),(.035,.055)):
                start,end=(int(round(lo+a*(hi-lo))),int(round(lo+b*(hi-lo)))) if side=='near' else (int(round(hi-b*(hi-lo))),int(round(hi-a*(hi-lo))))
                end=max(start+2,end)
                p=ink[:,start:end].mean(1) if rows else ink[start:end,:].mean(0)
                points=ordered_peaks(p)
                if points is not None:candidates.append(points)
            if not candidates:return None
            if len(candidates)==2 and (len(candidates[0])!=len(candidates[1]) or max(abs(candidates[0]-candidates[1]))>1.5):return None
            sides.append(np.mean(candidates,axis=0))
        if len(sides[0])!=len(sides[1]) or max(abs(sides[0]-sides[1]))>1.5:return None
        return (sides[0]+sides[1])/2
    for lo_x,hi_x,lo_y,hi_y,source in bounds:
        rows=locate(lo_x,hi_x,True);cols=locate(lo_y,hi_y,False)
        if rows is None or cols is None:continue
        def mapping(points):
            canonical=36+36*np.arange(len(points));axis=np.arange(256,dtype=float)
            mapped=np.interp(axis,canonical,points);step=float(np.median(np.diff(points))/36)
            mapped[axis<canonical[0]]=points[0]+(axis[axis<canonical[0]]-canonical[0])*step
            mapped[axis>canonical[-1]]=points[-1]+(axis[axis>canonical[-1]]-canonical[-1])*step
            return mapped.astype(np.float32)
        x,y=mapping(cols),mapping(rows);map_x=np.broadcast_to(x[None],(256,256)).copy();map_y=np.broadcast_to(y[:,None],(256,256)).copy()
        aligned=cv2.remap(image,map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=int(bg))
        return aligned,{'ready':True,'observed_rows':rows.tolist(),'observed_columns':cols.tolist(),'bounds_source':source,
                        'orientation':orientation,'scope':'ordered orthogonal lines, bilateral overhang agreement; not independently verified'}
    return image.copy(),{'ready':False,'reason':'bilateral ordered profiles disagree','orientation':orientation}
