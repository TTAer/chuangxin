"""ONNX YOLO detector, local weights only and no PyTorch requirement at inference."""
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class OnnxDetector:
    def __init__(self,path,confidence=.25,threads=2):
        path=Path(path)
        if not path.is_file(): raise ValueError('detector ONNX model is missing: '+str(path))
        options=ort.SessionOptions(); options.intra_op_num_threads=threads; options.inter_op_num_threads=threads
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        self.input=self.session.get_inputs()[0]
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        self.names=meta['names']; self.size=meta['imgsz']; self.confidence=confidence

    def __call__(self,image):
        h,w=image.shape[:2]; s=self.size; ratio=min(s/h,s/w)
        rw,rh=round(w*ratio),round(h*ratio); ox=(s-rw)//2; oy=(s-rh)//2
        canvas=np.full((s,s,3),114,dtype=np.uint8)
        canvas[oy:oy+rh,ox:ox+rw]=cv2.resize(image,(rw,rh))
        tensor=np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
        raw=self.session.run(None,{self.input.name:tensor})[0]
        if raw.ndim!=3 or raw.shape[1]!=4+len(self.names): raise ValueError('unsupported detector output layout')
        pred=raw[0].T; classes=pred[:,4:].argmax(axis=1); scores=pred[:,4:].max(axis=1)
        keep=scores>=self.confidence; pred=pred[keep]; classes=classes[keep]; scores=scores[keep]
        xywh=pred[:,:4].copy(); xywh[:,:2]-=xywh[:,2:]/2
        xywh[:,0]=(xywh[:,0]-ox)/ratio; xywh[:,1]=(xywh[:,1]-oy)/ratio; xywh[:,2:]/=ratio
        indices=[]
        for category in np.unique(classes):
            ids=np.flatnonzero(classes==category)
            selected=cv2.dnn.NMSBoxes(xywh[ids].tolist(),scores[ids].tolist(),self.confidence,.4)
            indices.extend(int(ids[int(j)]) for j in np.asarray(selected).flatten())
        detections=[]
        for i in indices:
            x,y,bw,bh=xywh[i]; box=[max(0,float(x)),max(0,float(y)),min(w,float(x+bw)),min(h,float(y+bh))]
            if box[0]>=box[2] or box[1]>=box[3]: continue
            detections.append({'bbox':box,'score':float(scores[i]),'category':self.names[int(classes[i])],'pins':[]})
        return detections
