"""Calibrate on train patches, gate on old development pages, never read old pressure."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.stroke_residual import residual_scores,decisions
from experiments.radial_evidence_net import crossing_patches
from scripts.prepare_circuit_program import execute


def exact_net(p,g):return {frozenset(v) for v in execute(p)[3]}=={frozenset(v) for v in execute(g)[3]}


def main():
    out=ROOT/'output/stroke-residual-development';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Preserve development')
    protocol={'scope':'train/dev only; image-only directional wire subtraction, no old pressure',
              'fixed_rule':{'low':.02,'high':.12},'train_search':'low.005/.01/.02/.03/.04; high.06/.08/.10/.12/.16/.20',
              'selection':'both confident class precision>=.9995; maximize confident coverage; first grid tie',
              'dev_gate':'no incorrect changed slots, no broken exact programs, exact-net>=baseline',
              'abstention':'ambiguous/nonfinite score preserves baseline; geometry rejects no program',
              'limits':'finite orthogonal R/C renderer only; arm contamination and blur may violate separable assumption'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    with np.load(ROOT/'data/processed/radial-evidence-v1/train-patches.npz') as z:p=z['raw'];labels=z['labels']+8
    score=np.concatenate([residual_scores(p[i:i+1024]) for i in range(0,len(p),1024)]);np.savez_compressed(out/'train-scores.npz',scores=score)
    rows=[]
    for low in (.005,.01,.02,.03,.04):
        for high in (.06,.08,.10,.12,.16,.20):
            d=decisions(score,low,high);record={'low':low,'high':high,'coverage':float((d!=7).mean())}
            for c in (8,9):record[f'precision_{c}']=float((labels[d==c]==c).mean()) if (d==c).any() else 0
            record['eligible']=record['precision_8']>=.9995 and record['precision_9']>=.9995;rows.append(record)
    eligible=[r for r in rows if r['eligible']];best=max(eligible,key=lambda r:r['coverage']) if eligible else None
    (out/'train-calibration.json').write_text(json.dumps({'grid':rows,'chosen':best},indent=2),encoding='utf-8')
    rules={'fixed':protocol['fixed_rule']}
    if best:rules['calibrated']={'low':best['low'],'high':best['high']}
    with np.load(ROOT/'data/processed/radial-evidence-v1/validation.npz') as z:raw=z['observed_images'];aligned=z['images'];gold=z['program']
    with np.load(ROOT/'output/radial-evidence-evaluation/validation-predictions.npz') as z:base=z['baseline']
    geometry=json.loads((ROOT/'data/processed/radial-evidence-v1/validation-geometry.json').read_text());result={'protocol':protocol,'rules':{}}
    for name,rule in rules.items():
        pred=base.copy();changes=[]
        for i,g in enumerate(geometry):
            patches,slots=crossing_patches(raw[i,0],aligned[i,0],g,'raw')
            if not slots:continue
            ds=decisions(residual_scores(patches),**rule)
            for s,d in zip(slots,ds):
                if d!=7 and pred[i,s]!=7 and d!=pred[i,s]:changes.append({'case_index':i,'slot':s,'correct_after':bool(d==gold[i,s])});pred[i,s]=d
        before=(base==gold).all(1);after=(pred==gold).all(1);net=float(np.mean([exact_net(p,g) for p,g in zip(pred,gold)]));original=float(np.mean([exact_net(p,g) for p,g in zip(base,gold)]))
        r={'thresholds':rule,'changed_slots':len(changes),'wrong_changed_slots':sum(not d['correct_after'] for d in changes),'fixed_programs':int((~before&after).sum()),'broken_programs':int((before&~after).sum()),'exact_net':net,'baseline_exact_net':original,'changes':changes}
        r['gate_passed']=r['wrong_changed_slots']==0 and r['broken_programs']==0 and net>=original
        result['rules'][name]=r;np.savez_compressed(out/f'{name}-dev-predictions.npz',program=pred);print(json.dumps({'rule':name,**r}),flush=True)
    result['selected_rule']='calibrated' if 'calibrated' in result['rules'] and result['rules']['calibrated']['gate_passed'] else ('fixed' if result['rules']['fixed']['gate_passed'] else None)
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
