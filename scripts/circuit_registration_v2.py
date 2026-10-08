"""Image-only multiband analog line profiles with subpixel lattice fitting."""
import numpy as np,cv2


def profile_centers(profile):
    smooth=cv2.GaussianBlur(profile.astype(np.float32).reshape(-1,1),(1,3),.5).ravel()
    threshold=max(.05,float(smooth.max())*.25)
    indices=np.flatnonzero(smooth>threshold)
    if not len(indices):return None
    groups=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    groups=[g for g in groups if len(g)<=15]
    if not 3<=len(groups)<=6:return None
    centers=np.array([np.average(g,weights=smooth[g]) for g in groups])
    slope,intercept=np.polyfit(np.arange(len(centers)),centers,1)
    residual=float(np.max(np.abs(centers-(intercept+slope*np.arange(len(centers))))))
    if not 18<=slope<=47 or residual>1.5:return None
    return {'centers':centers.tolist(),'scale':float(slope/36),'translation':float(intercept-slope),'residual':residual}


def register_image_v2(image):
    if image.shape!=(256,256):raise ValueError('Whole256 raster required')
    background=float(np.median(image));ink=np.clip((background-image.astype(float))/max(1,background),0,1)
    ys,xs=np.where(ink>.4)
    if len(xs)<20:return image.copy(),{'ready':False,'reason':'insufficient ink'}
    def locate(low,high,axis):
        span=float(high-low);candidates=[]
        for start,end in ((.02,.05),(.045,.07)):
            a=int(round(low+start*span));b=max(a+2,int(round(low+end*span)))
            profile=ink[:,a:b].mean(1) if axis=='rows' else ink[a:b,:].mean(0)
            result=profile_centers(profile)
            if result:candidates.append(result)
        if not candidates:return None
        if len(candidates)==2:
            a,b=candidates
            if len(a['centers'])!=len(b['centers']) or max(abs(np.array(a['centers'])-b['centers']))>1.5:return None
        return min(candidates,key=lambda x:x['residual'])
    rows=locate(xs.min(),xs.max(),'rows');cols=locate(ys.min(),ys.max(),'columns')
    if rows is None or cols is None:return image.copy(),{'ready':False,'reason':'multiband profile lacks consistent lattice'}
    sx,sy=cols['scale'],rows['scale'];tx,ty=cols['translation'],rows['translation']
    inverse=np.array([[1/sx,0,-tx/sx],[0,1/sy,-ty/sy]],np.float32)
    aligned=cv2.warpAffine(image,inverse,(256,256),flags=cv2.INTER_LINEAR,borderValue=int(background))
    return aligned,{'ready':True,'scale_x':sx,'scale_y':sy,'translation_x':tx,'translation_y':ty,
                    'observed_rows':rows['centers'],'observed_columns':cols['centers'],
                    'row_residual':rows['residual'],'column_residual':cols['residual'],
                    'scope':'multiband overhang profiles; uniform orthogonal grid assumption'}
