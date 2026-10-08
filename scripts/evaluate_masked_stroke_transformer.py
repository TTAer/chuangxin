"""Fixed cohort comparison and numerical ONNX verification; development only."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,onnxruntime as ort
from experiments.port_partition import RELATIONS
from scripts.evaluate_port_partition_expanded import score,selected_merge_stats


def main():
    out=ROOT/'output/training/masked-stroke-transformer'
    with np.load(ROOT/'data/processed/port-partition-expanded/relations.npz') as ar:z={k:ar[k] for k in ar.files}
    options=ort.SessionOptions();options.intra_op_num_threads=2
    results=[];exports=[]
    for variant in ('scratch','multisource-pretrained','PPN-expanded-seed0'):
        path=ROOT/('models/candidates/port-partition-expanded/expanded-seed0/port-graph/model.onnx' if variant.startswith('PPN') else f'models/candidates/masked-stroke-transformer/{variant}/model.onnx')
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        if not variant.startswith('PPN'):
            with np.load(out/variant/'export-reference.npz') as ref:
                actual=session.run(None,{'image':ref['image']})[0]
                error=float(np.abs(actual-ref['logits']).max())
                if not np.allclose(actual,ref['logits'],atol=1e-4,rtol=1e-4):raise ValueError('ONNX export mismatch')
                exports.append({'variant':variant,'max_abs_error':error})
        for split in (1,2):
            ids=np.flatnonzero(z['split']==split)
            logits=np.concatenate([session.run(None,{'image':z['images'][part].astype(np.float32)/255})[0] for part in np.array_split(ids,int(np.ceil(len(ids)/64)))])
            p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
            result={'variant':variant,'cohort':'original133' if split==1 else 'automatic668','map':score(RELATIONS[logits.argmax(1)],z['targets'][ids],z['valid'][ids]),
                    'high_weight_merge':{str(t):selected_merge_stats(p@RELATIONS,z['targets'][ids],z['valid'][ids],t) for t in (.95,.99)}}
            results.append(result);print(json.dumps(result),flush=True)
    (out/'evaluation.json').write_text(json.dumps({'results':results,'exports':exports,'independent_evaluation':False,'thresholds_calibrated':False,'runtime_changed':False},indent=2),encoding='utf-8')


if __name__=='__main__':main()
