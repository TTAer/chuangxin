"""Whole-page orientation witnesses before uniform orthogonal registration."""
import numpy as np,cv2
from scripts.circuit_registration_v2 import register_image_v2


def weighted_median(values,weights):
    order=np.argsort(values);values=np.asarray(values)[order];weights=np.asarray(weights)[order]
    return float(values[np.searchsorted(np.cumsum(weights),weights.sum()/2)])


def register_guarded(image):
    background=float(np.median(image));mask=(image<background*.6).astype(np.uint8)*255
    lines=cv2.HoughLinesP(mask,1,np.pi/180,20,minLineLength=25,maxLineGap=8)
    h=[];v=[];hw=[];vw=[]
    for line in ([] if lines is None else lines.reshape(-1,4)):
        x1,y1,x2,y2=map(float,line);dx=abs(x2-x1);dy=abs(y2-y1);length=float(np.hypot(dx,dy))
        angle=float(np.degrees(np.arctan2(dy,dx)))
        if angle<=30:h.append(angle);hw.append(length)
        elif angle>=60:v.append(90-angle);vw.append(length)
    evidence={'horizontal_segments':len(h),'vertical_segments':len(v),
              'horizontal_support':float(sum(hw)),'vertical_support':float(sum(vw))}
    if len(h)<2 or len(v)<2 or sum(hw)<80 or sum(vw)<40:
        return image.copy(),{'ready':False,'reason':'insufficient full-page orientation witnesses','orientation':evidence}
    evidence['horizontal_deviation_deg']=weighted_median(h,hw)
    evidence['vertical_deviation_deg']=weighted_median(v,vw)
    if max(evidence['horizontal_deviation_deg'],evidence['vertical_deviation_deg'])>3:
        return image.copy(),{'ready':False,'reason':'nonorthogonal full-page line geometry','orientation':evidence}
    aligned,info=register_image_v2(image);info['orientation']=evidence
    info['scope']='orientation-checked uniform orthogonal grid; not independently verified'
    return aligned,info
