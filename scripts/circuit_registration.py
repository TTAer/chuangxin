"""Image-only registration of translated/scaled orthogonal grammar pages."""
import numpy as np,cv2


def run_centers(signal):
    ids=np.flatnonzero(signal)
    if not len(ids):return np.array([],float)
    groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
    return np.array([float(g.mean()) for g in groups if len(g)<=15])


def register_image(image):
    if image.shape!=(256,256):raise ValueError('Whole256 raster required')
    mask=image<float(np.median(image))*.6
    ys,xs=np.where(mask)
    if len(xs)<20:return image.copy(),{'ready':False,'reason':'insufficient ink'}
    spanx=float(xs.max()-xs.min());spany=float(ys.max()-ys.min())
    # Probe isolated line overhangs before the first vertical/horizontal crossing.
    x=int(round(xs.min()+spanx*10/224));y=int(round(ys.min()+spany*10/224))
    rows=run_centers(mask[:,max(0,x-1):x+2].mean(1)>.25)
    cols=run_centers(mask[max(0,y-1):y+2,:].mean(0)>.25)
    if not (3<=len(rows)<=6 and 3<=len(cols)<=6):
        return image.copy(),{'ready':False,'reason':'line probes outside grammar','rows':rows.tolist(),'columns':cols.tolist()}
    dy=np.diff(rows);dx=np.diff(cols);sy=float(np.median(dy)/36);sx=float(np.median(dx)/36)
    if min(sx,sy)<.5 or max(sx,sy)>1.3 or max(np.ptp(dx),np.ptp(dy))>3:
        return image.copy(),{'ready':False,'reason':'nonuniform line grid'}
    tx=float(cols[0]-36*sx);ty=float(rows[0]-36*sy)
    inverse=np.array([[1/sx,0,-tx/sx],[0,1/sy,-ty/sy]],np.float32)
    aligned=cv2.warpAffine(image,inverse,(256,256),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=int(np.median(image)))
    return aligned,{'ready':True,'scale_x':sx,'scale_y':sy,'translation_x':tx,'translation_y':ty,
                    'observed_rows':rows.tolist(),'observed_columns':cols.tolist(),
                    'scope':'image-only overhang probes; uniform orthogonal grid assumption'}
