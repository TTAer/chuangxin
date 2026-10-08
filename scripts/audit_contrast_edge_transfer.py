"""Posthoc frozen historical-model control on the same newly evaluated pages."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from experiments.factorized_circuit_reader import FactorizedCircuitReader
from scripts.train_circuit_program import metrics


def main():
    out=ROOT/'output/contrast-edge-transfer';out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise ValueError('Preserve diagnostic')
    checkpoint=ROOT/'output/training/registered-circuit-v1/best.pt';model=FactorizedCircuitReader().cuda().eval();torch.set_num_threads(4)
    model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda'))
    report={'scope':'posthoc fixed old-model diagnostic; no model selection or new blind claim',
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'cohorts':{}}
    for name in ('validation','stress'):
        with np.load(ROOT/f'data/processed/contrast-edge-v1/{name}.npz') as z:images,gold=z['images'],z['program']
        ready=np.array([r['ready'] for r in json.loads((ROOT/f'data/processed/contrast-edge-v1/{name}-geometry.json').read_text())])
        rows=[]
        with torch.no_grad():
            for i in range(0,len(images),16):
                x=torch.tensor(1-images[i:i+16].astype(np.float32)/255,device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):rows.append(model.generate(x).cpu().numpy())
        pred=np.concatenate(rows);report['cohorts'][name]={'all':metrics(pred,gold),'ready':metrics(pred[ready],gold[ready])}
        np.savez_compressed(out/f'{name}-predictions.npz',program=pred)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
