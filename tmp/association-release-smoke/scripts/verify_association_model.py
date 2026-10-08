"""Check export numerics, dynamic shapes and node-permutation equivariance."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np

p=argparse.ArgumentParser(); p.add_argument('--torch',action='store_true'); a=p.parse_args()
cache=ROOT/'data/processed/association'; records=json.loads((cache/'records.json').read_text(encoding='utf-8'))
records=[r for r in records if r['split']=='validation'][:20]
out=ROOT/'output/association-model-verification'; out.mkdir(parents=True,exist_ok=True)
if a.torch:
    import torch
    from pcbparse.association_model import ElectricalAssociationNet
    torch.set_num_threads(4)
    for name,layers in [('association-balanced',2),('association-mlp-balanced',0)]:
        model=ElectricalAssociationNet(layers=layers).eval(); model.load_state_dict(torch.load(ROOT/'output/training'/name/'best.pt',weights_only=True))
        logits={}
        with torch.no_grad():
            for r in records:
                with np.load(cache/r['file']) as z: args=tuple(torch.from_numpy(z[k]) for k in ('nodes','component_relation','incidence_relation','pairs','edge_features'))
                logits[r['file']]=model(*args).numpy()
        np.savez_compressed(out/(name+'-torch.npz'),**logits)
    print('Saved torch reference outputs for 20 differently shaped validation graphs.')
else:
    import onnxruntime as ort
    reports=[]; rng=np.random.default_rng(7)
    for name in ('association-balanced','association-mlp-balanced'):
        session=ort.InferenceSession(str(ROOT/'models/candidates'/name/'association.onnx'),providers=['CPUExecutionProvider']); error=0.; equivariance=0.
        with np.load(out/(name+'-torch.npz')) as expected:
            for r in records:
                with np.load(cache/r['file']) as z: tensors={k:z[k] for k in ('nodes','component_relation','incidence_relation','pairs','edge_features')}
                input_names={i.name for i in session.get_inputs()}
                logits=session.run(None,{k:v for k,v in tensors.items() if k in input_names})[0]; error=max(error,float(np.max(np.abs(logits-expected[r['file']]))))
                n=len(tensors['nodes']); perm=rng.permutation(n); inv=np.argsort(perm); changed=dict(tensors)
                changed['nodes']=tensors['nodes'][perm]
                for k in ('component_relation','incidence_relation'): changed[k]=tensors[k][np.ix_(perm,perm)]
                changed['pairs']=inv[tensors['pairs']].astype(np.int64)
                equivariance=max(equivariance,float(np.max(np.abs(session.run(None,{k:v for k,v in changed.items() if k in input_names})[0]-logits))))
        assert error<1e-4 and equivariance<1e-4,(name,error,equivariance)
        reports.append({'model':name,'graphs':len(records),'max_onnx_torch_error':error,'max_node_permutation_error':equivariance,'passed':True})
    (out/'summary.json').write_text(json.dumps({'records':reports,'passed':True},indent=2),encoding='utf-8'); print(json.dumps(reports,indent=2))
