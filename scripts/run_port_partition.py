"""Image + existing detector/OCR observations -> offline port graph proposals.

No ground truth is opened. This research entry point does not output a contest
netlist or commit proposals into RC0.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import onnxruntime as ort
from experiments.port_partition import PARTITIONS, PORTS
from pcbparse.vision import read_image, wire_lines
from pcbparse.topology_perception import crossing_candidates, patch_tensor


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image',required=True)
    p.add_argument('--observations',required=True)
    p.add_argument('--model',default='models/candidates/port-partition-v1/matched-seed0/port-graph/model.onnx')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    image=read_image(args.image);h,w=image.shape[:2]
    obs=json.loads(Path(args.observations).read_text(encoding='utf-8'))
    if obs.get('coordinate_system') != 'image_bottom_left':
        # Existing cache names the bottom-left convention differently in some runs.
        if obs.get('coordinate_system') != 'bottom-left':
            raise ValueError('Expected explicitly declared bottom-left observations')
    if obs['image']['height']!=h or obs['image']['width']!=w:
        raise ValueError('Observations/image dimensions differ')
    mask=(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<220).astype(np.uint8)*255
    boxes=[c['bbox'] for c in obs['components'].values()]+[t['bbox'] for t in obs.get('texts',[])]
    for a,b,c,d in boxes:
        x1,x2=max(0,int(a)),min(w,int(np.ceil(c)))
        y1,y2=max(0,int(h-d)),min(h,int(np.ceil(h-b)))
        mask[y1:y2,x1:x2]=0
    segments,_=wire_lines(mask,split_junctions=False)
    candidates=crossing_candidates(segments)
    options=ort.SessionOptions();options.intra_op_num_threads=2
    session=ort.InferenceSession(args.model,sess_options=options,providers=['CPUExecutionProvider'])
    proposals=[]
    for row in candidates:
        input_image=patch_tensor(image,row['x'],row['y'])[:3][None]
        logits=session.run(None,{'image':input_image})[0][0]
        if logits.shape!=(15,):raise ValueError('Expected structured partition model')
        weights=np.exp(logits-logits.max());weights/=weights.sum()
        index=int(logits.argmax());groups=PARTITIONS[index].tolist()
        proposals.append({'point_xy':[row['x'],row['y']], 'segment_ids':row['wires'],
                          'port_order':PORTS,'partition':groups,
                          'partition_weight':float(weights[index]),
                          'decision':'evidence_only_not_committed',
                          'weight_is_calibrated_probability':False})
    result={'schema':'port-partition-image-proposals-v1','coordinate_system':'image_top_left',
            'image_sha256':hashlib.sha256(Path(args.image).read_bytes()).hexdigest(),
            'observations_sha256':hashlib.sha256(Path(args.observations).read_bytes()).hexdigest(),
            'model_sha256':hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
            'input_scope':'pixels and cached detector/OCR observations, no ground truth',
            'wire_segments':segments,'port_partitions':proposals,'runtime_changed':False,
            'limitations':'Physical local proposal system; no pin identity, no flying-label replacement, no contest output; proposal confidence uncalibrated.'}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'segments':len(segments),'port_proposals':len(proposals),'commits':0}))


if __name__=='__main__':main()
