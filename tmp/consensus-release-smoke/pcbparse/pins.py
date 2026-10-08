"""ONNX pin heatmaps with shared training/inference crop transforms."""
from pathlib import Path
import json
import cv2
import numpy as np


def crop_tensor(image, bbox, size=96, margin=.15):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    dx = max(4, (x2-x1)*margin); dy = max(4, (y2-y1)*margin)
    box = [max(0, int(x1-dx)), max(0, int(y1-dy)), min(w, int(x2+dx)+1), min(h, int(y2+dy)+1)]
    a, b, c, d = box
    if c <= a or d <= b:
        raise ValueError('empty pin crop')
    rgb = cv2.resize(image[b:d, a:c], (size, size))[:, :, ::-1].astype(np.float32)/255
    # Same color separation as the parser; mask is useful for colored drawings.
    blue, green, red = cv2.split(image[b:d, a:c].astype(np.int16))
    colored = ((red-green>35)&(red-blue>25)&(red>90))
    mask = colored if colored.sum()>max(5, colored.size*.0002) else cv2.cvtColor(image[b:d,a:c],cv2.COLOR_BGR2GRAY)<170
    mask = cv2.resize(mask.astype(np.float32), (size,size), interpolation=cv2.INTER_NEAREST)
    yy, xx = np.mgrid[:size,:size].astype(np.float32)
    tensor = np.concatenate((rgb.transpose(2,0,1), mask[None], (xx[None]/(size-1)*2-1), (yy[None]/(size-1)*2-1)))
    return np.ascontiguousarray(tensor), box


def peaks(heatmap, threshold=.35, radius=3):
    maximum = cv2.dilate(heatmap, np.ones((radius*2+1,radius*2+1),np.uint8))
    ys, xs = np.where((heatmap >= maximum-1e-7)&(heatmap>=threshold))
    chosen=[]
    for i in np.argsort(heatmap[ys,xs])[::-1]:
        x,y=int(xs[i]),int(ys[i])
        if any((x-a)**2+(y-b)**2<=radius**2 for a,b,_ in chosen): continue
        chosen.append((x,y,float(heatmap[y,x])))
    return chosen


class PinLocator:
    def __init__(self, path='models/pins/pin-heatmap.onnx', threads=2):
        import onnxruntime as ort
        path=Path(path)
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        self.threshold=meta['threshold']; self.size=meta['size']; self.margin=meta['margin']
        opts=ort.SessionOptions(); opts.intra_op_num_threads=threads; opts.inter_op_num_threads=threads
        self.session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
        self.input=self.session.get_inputs()[0].name

    def __call__(self, image, bbox):
        tensor,box=crop_tensor(image,bbox,self.size,self.margin)
        logits=self.session.run(None,{self.input:tensor[None]})[0][0]
        probabilities=1/(1+np.exp(-np.clip(logits,-30,30)))
        a,b,c,d=box
        return [{'x':a+(x+.5)/self.size*(c-a),'y':b+(y+.5)/self.size*(d-b),'score':score}
                for x,y,score in peaks(probabilities[0],self.threshold)]
