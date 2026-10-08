"""Multicontrast direction witnesses and weak-line bounding fallback; image only."""
import numpy as np,cv2
from scripts.circuit_geometry_guard import weighted_median
from scripts.circuit_registration_v2 import profile_centers,register_image_v2


def orientation_evidence(mask):
    lines=cv2.HoughLinesP(mask.astype(np.uint8)*255,1,np.pi/180,20,minLineLength=25,maxLineGap=8)
    h=[];v=[];hw=[];vw=[]
    for line in ([] if lines is None else lines.reshape(-1,4)):
        x1,y1,x2,y2=map(float,line);dx=abs(x2-x1);dy=abs(y2-y1);length=float(np.hypot(dx,dy));a=float(np.degrees(np.arctan2(dy,dx)))
        if a<=30:h.append(a);hw.append(length)
        elif a>=60:v.append(90-a);vw.append(length)
    result={'horizontal_segments':len(h),'vertical_segments':len(v),'horizontal_support':sum(hw),'vertical_support':sum(vw)}
    result['witnessed']=len(h)>=2 and len(v)>=2 and sum(hw)>=80 and sum(vw)>=40
    if result['witnessed']:
        result['horizontal_deviation_deg']=weighted_median(h,hw);result['vertical_deviation_deg']=weighted_median(v,vw)
        result['orthogonal']=max(result['horizontal_deviation_deg'],result['vertical_deviation_deg'])<=3
    return result


def register_weak_bounds(image,ink):
    mask=(ink>.10).astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(mask,8)
    boxes=[s for s in stats[1:] if s[4]>=25 and max(s[2],s[3])>=25]
    if not boxes:return image.copy(),{'ready':False,'reason':'no weak-line components'}
    lo_x=min(s[0] for s in boxes);hi_x=max(s[0]+s[2]-1 for s in boxes)
    lo_y=min(s[1] for s in boxes);hi_y=max(s[1]+s[3]-1 for s in boxes)
    def locate(low,high,rows):
        candidates=[]
        for a,b in ((.02,.05),(.045,.07)):
            start=int(round(low+a*(high-low)));end=max(start+2,int(round(low+b*(high-low))))
            p=ink[:,start:end].mean(1) if rows else ink[start:end,:].mean(0)
            fit=profile_centers(p)
            if fit:candidates.append(fit)
        if not candidates:return None
        if len(candidates)==2 and (len(candidates[0]['centers'])!=len(candidates[1]['centers']) or max(abs(np.array(candidates[0]['centers'])-candidates[1]['centers']))>1.5):return None
        return min(candidates,key=lambda d:d['residual'])
    rows=locate(lo_x,hi_x,True);cols=locate(lo_y,hi_y,False)
    if rows is None or cols is None:return image.copy(),{'ready':False,'reason':'weak-line bounds lack consistent profiles'}
    sx,sy=cols['scale'],rows['scale'];tx,ty=cols['translation'],rows['translation']
    affine=np.array([[1/sx,0,-tx/sx],[0,1/sy,-ty/sy]],np.float32)
    aligned=cv2.warpAffine(image,affine,(256,256),flags=cv2.INTER_LINEAR,borderValue=int(np.median(image)))
    return aligned,{'ready':True,'scale_x':sx,'scale_y':sy,'translation_x':tx,'translation_y':ty,
                    'observed_rows':rows['centers'],'observed_columns':cols['centers'],'bounds_source':'weak connected line components'}


def register_guarded_v2(image):
    if image.shape!=(256,256):raise ValueError('Whole256 raster required')
    bg=float(np.median(image));ink=np.clip((bg-image.astype(float))/max(1,bg),0,1)
    strong=orientation_evidence(ink>.4);evidence=[{'cut':.4,**strong}]
    if strong['witnessed'] and not strong['orthogonal']:
        return image.copy(),{'ready':False,'reason':'nonorthogonal strong direction evidence','orientation':evidence}
    soft=cv2.GaussianBlur(ink.astype(np.float32),(3,3),.7)
    if not strong['witnessed']:
        for cut in (.18,.10):evidence.append({'cut':cut,**orientation_evidence(soft>cut)})
        witnessed=[d for d in evidence if d['witnessed']]
        if not witnessed:return image.copy(),{'ready':False,'reason':'insufficient multicontrast directions','orientation':evidence}
        if any(not d['orthogonal'] for d in witnessed):
            return image.copy(),{'ready':False,'reason':'nonorthogonal or conflicting multicontrast directions','orientation':evidence}
    aligned,info=register_image_v2(image)
    if not info['ready']:aligned,info=register_weak_bounds(image,soft)
    info['orientation']=evidence;info['scope']='multicontrast direction and uniform grid assumptions; not independently verified'
    return aligned,info
