"""Experimental CPU ONNX artifacts; not inserted into pipeline.json."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import onnxruntime as ort
from experiments.pin_token_v1 import make_model

torch.set_num_threads(4)
cache=np.load(ROOT/'data/processed/pin-token-v1/tokens.npz')
report=[]
for variant in ('geometry-text','visual-joint'):
    base=ROOT/'output/training/pin-token-v1'/variant; target=ROOT/'models/candidates/pin-token-v1'/variant; target.mkdir(parents=True,exist_ok=True)
    model=make_model(variant=='visual-joint'); model.load_state_dict(torch.load(base/'best.pt',weights_only=True,map_location='cpu')); model.eval()
    inputs=(torch.from_numpy(cache['patch'][:4]).float()/255,torch.from_numpy(cache['text'][:4]),torch.from_numpy(cache['pairs'][:4]),torch.from_numpy(cache['valid'][:4]))
    names=['patch','text','pairs','valid']; path=target/'pin-token.onnx'
    # Geometry control does not consume the image patch; exporter removes it.
    torch.onnx.export(model,inputs,str(path),input_names=names,output_names=['role_logits','affinity_logits'],opset_version=17,
        dynamic_axes={name:{0:'batch'} for name in names+['role_logits','affinity_logits']})
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']); used={i.name for i in session.get_inputs()}
    maximum=0.; matching=True
    for n in (1,7):
        args=(torch.from_numpy(cache['patch'][:n]).float()/255,torch.from_numpy(cache['text'][:n]),torch.from_numpy(cache['pairs'][:n]),torch.from_numpy(cache['valid'][:n]))
        with torch.no_grad(): expected=[t.numpy() for t in model(*args)]
        actual=session.run(None,{name:t.numpy() for name,t in zip(names,args) if name in used})
        for a,b in zip(actual,expected):
            maximum=max(maximum,float(np.max(np.abs(a-b)))); matching=matching and bool(np.array_equal(a.argmax(1),b.argmax(1)))
    assert maximum<1e-4 and matching
    meta=json.loads((base/'best.json').read_text(encoding='utf-8')); meta.update({'onnx_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'onnx_inputs':sorted(used),'cpu_export_max_absolute_error':maximum,'prediction_argmax_matches':matching,'runtime_enabled':False})
    (target/'pin-token.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); report.append({'variant':variant,'max_absolute_error':maximum,'argmax_matches':matching})
(ROOT/'output/ablation/pin-token-v1-export.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report),flush=True)
