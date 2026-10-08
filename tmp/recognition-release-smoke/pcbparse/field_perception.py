"""Overlapping ONNX dense-field inference with an actual-ink wire gate."""
import json
from pathlib import Path
import cv2
import numpy as np

def field_input(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    ink=1-gray
    binary=(gray<.86).astype(np.uint8)
    horizontal=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((1,7),np.uint8)).astype(np.float32)
    vertical=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((7,1),np.uint8)).astype(np.float32)
    return np.ascontiguousarray(np.stack((ink,horizontal,vertical)))

def tile_starts(length,size,stride):
    if length<=size: return [0]
    starts=list(range(0,length-size+1,stride))
    if starts[-1]!=length-size: starts.append(length-size)
    return starts

class FieldPerception:
    def __init__(self,path,threads=2):
        import onnxruntime as ort
        path=Path(path); self.path=str(path); self.metadata=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        self.size=self.metadata['size']; self.threshold=self.metadata.get('wire_threshold',.5)
        opts=ort.SessionOptions(); opts.intra_op_num_threads=threads; opts.inter_op_num_threads=threads
        self.session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    def __call__(self,image):
        h,w=image.shape[:2]; size=self.size; stride=size//2
        # Prevent unbounded CPU inference; the calling parser retains its normal
        # mask when the optional model cannot run within this explicit budget.
        positions=[(x,y) for y in tile_starts(h,size,stride) for x in tile_starts(w,size,stride)]
        if len(positions)>1600: raise ValueError('dense field tile budget exceeded')
        padded=cv2.copyMakeBorder(image,0,max(0,size-h),0,max(0,size-w),cv2.BORDER_CONSTANT,value=(255,255,255))
        ph,pw=padded.shape[:2]; total=np.zeros((5,ph,pw),np.float32); counts=np.zeros((ph,pw),np.float32)
        blend=np.outer(np.hanning(size),np.hanning(size)).astype(np.float32); blend=np.maximum(blend,.05)
        for start in range(0,len(positions),8):
            batch=positions[start:start+8]
            tensors=np.stack([field_input(padded[y:y+size,x:x+size]) for x,y in batch])
            logits=self.session.run(None,{'image':tensors})[0]
            if logits.shape!=(len(batch),5,size,size) or not np.isfinite(logits).all(): raise ValueError('invalid dense field output')
            probabilities=1/(1+np.exp(-np.clip(logits,-30,30)))
            for (x,y),prob in zip(batch,probabilities): total[:,y:y+size,x:x+size]+=prob*blend; counts[y:y+size,x:x+size]+=blend
        fields=total[:,:h,:w]/counts[:h,:w].clip(min=1e-6)[None]
        visible=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<220
        wire=(np.maximum(fields[0],fields[1])>=self.threshold)&visible
        report={'module':'DirectionalFieldNet','model':self.path,'heads':['horizontal_wire','vertical_wire','pin_tip','junction','ink'],'tiles':len(positions),'wire_threshold':self.threshold,'actual_ink_gate':True,'junction_policy':'dense junction output diagnostic only; TopologyPerception retains electrical contact decisions','wire_pixels':int(wire.sum()),'official_equivalence':False}
        return wire.astype(np.uint8)*255,fields,report
