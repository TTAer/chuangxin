"""One post-training train/holdout audit, including collapse and fixed baselines."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from experiments.netset_query import NetSetQueryDecoder
from scripts.train_netset_oracle import net_metrics


def main():
    torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False)
    bank=ROOT/'data/processed/netset-oracle';out=ROOT/'output/training/netset-oracle'
    manifest=json.loads((bank/'manifest.json').read_text(encoding='utf-8'));results=[]
    for variant in ('full-pin-oracle','full-pin-wire-oracle'):
        model=NetSetQueryDecoder().eval();model.load_state_dict(torch.load(out/variant/'best.pt',weights_only=True,map_location='cpu'))
        records=[]
        with torch.no_grad():
            for row in manifest['rows']:
                with np.load(bank/row['case']/'oracle.npz') as z:d={k:z[k] for k in z.files}
                outputs=model(torch.tensor(d['pins'])[None],torch.tensor(d['wires'])[None])
                metrics=net_metrics(outputs,d['pin_labels'],d['wire_labels'])
                records.append({'case':row['case'],'split':row['split'],**metrics})
                dest=out/variant/'predictions';dest.mkdir(exist_ok=True)
                np.savez_compressed(dest/(row['case']+'.npz'),pin_logits=outputs['pin_logits'][0].numpy(),wire_logits=outputs['wire_logits'][0].numpy(),existence=outputs['net_existence_logits'][0].numpy())
        summary={}
        for split in ('train','development_holdout'):
            rows=[r for r in records if r['split']==split];keys=[k for k,v in rows[0].items() if isinstance(v,(int,float))]
            summary[split]={'pages':len(rows),'mean':{k:float(np.mean([r[k] for r in rows])) for k in keys}}
        results.append({'variant':variant,'wire_head_supervised':variant=='full-pin-wire-oracle','summary':summary,'pages':records})
    result={'results':results,'predictor_feature_order_invariant':True,'net_membership_matching':'Hungarian maximizes per-net Dice then micro member counts; local development metric, NOT verified official equivalence',
            'wire_ownership_metric':'exclusive classification of GT segments into pin-matched net; not endpoint-matching LineF1',
            'no_new_training':True,'runtime_changed':False,'independent_validation':False}
    (out/'post-training-evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='pages'} for r in results]))


if __name__=='__main__':main()
