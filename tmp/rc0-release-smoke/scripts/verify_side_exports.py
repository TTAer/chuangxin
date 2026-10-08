"""CPU PyTorch/ONNX parity over fixed real and synthetic bands."""
import json
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from pcbparse.side_strip_model import SideStripNet
from pcbparse.side_strips import SideStripLocator,tensor
torch.set_num_threads(2); gray=np.load(ROOT/'data/processed/side-strips/bands.npz')['gray']; indices=np.linspace(0,len(gray)-1,16,dtype=int)
p=argparse.ArgumentParser(); p.add_argument('--models',nargs='+',default=['base','mixed','local']); p.add_argument('--output',default='output/side-strip-verification/exports.json'); a=p.parse_args()
reports=[]
for name in a.models:
    root=ROOT/'models/candidates'/('side-strip-'+name); meta=json.loads((root/'side-strip.json').read_text(encoding='utf-8'))
    model=SideStripNet(dilated=name in ('base','mixed')); model.load_state_dict(torch.load(root/'best.pt',map_location='cpu',weights_only=True)); model.eval(); locator=SideStripLocator(root/'side-strip.onnx')
    maximum=0.; coordinate=0.
    with torch.no_grad():
        for i in indices:
            x=tensor(gray[i])[None]; expected=model(torch.from_numpy(x)).numpy(); actual=locator.session.run(None,{locator.input:x})[0]
            maximum=max(maximum,float(np.max(np.abs(expected-actual))))
            coordinate=max(coordinate,float(np.max(np.abs(np.tanh(expected[:,1])*32-np.tanh(actual[:,1])*32))))
    assert maximum<.001 and coordinate<.001,(name,maximum,coordinate)
    reports.append({'model':name,'samples':len(indices),'max_logit_error':maximum,'max_normal_coordinate_error_pixels':coordinate,'parameters':meta['parameters'],'onnx_sha256':meta['onnx_sha256']})
out=ROOT/a.output; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(reports,indent=2),encoding='utf-8'); print(json.dumps(reports),flush=True)
