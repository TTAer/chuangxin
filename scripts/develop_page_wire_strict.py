"""Second separate hypothesis: use clean page-wide wire profiles; train/dev gate only."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.page_wire_residual import page_scores
from experiments.stroke_residual import decisions
from scripts.develop_stroke_residual import exact_net


def main():
    out=ROOT/'output/page-wire-residual-strict-development';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Preserve development')
    protocol={'scope':'third development attempt, strict zero training errors; public dev reused, no old pressure',
              'change':'estimate shared wire profiles at far right/bottom page overhang across rows/columns; finite full-length-wire assumption',
              'search':'low.001/.003/.005/.01/.02/.03/.04; high.06/.08/.10/.12/.16/.20/.25/.30',
              'selection':'both confident class precision=1 on train, maximize coverage; no confidence guarantee from finite training',
              'dev_gate':'wrong changed slots0, broken programs0, exact-net>=baseline, at least1 correct changed slot',
              'deployment':'ambiguous scores preserve frozen baseline; not independently verified'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    with np.load(ROOT/'output/page-wire-residual-development/train-scores.npz') as z:scores=z['scores']
    with np.load(ROOT/'data/processed/radial-evidence-v1/train-patches.npz') as z:labels=z['labels']+8;pages=z['page_ids'];slots=z['slot_ids']
    assert len(scores)==len(labels) and np.all(np.diff(pages)>=0)
    grid=[]
    for low in (.001,.003,.005,.01,.02,.03,.04):
        for high in (.06,.08,.10,.12,.16,.20,.25,.30):
            d=decisions(scores,low,high);r={'low':low,'high':high,'coverage':float((d!=7).mean())}
            for c in (8,9):r[f'precision_{c}']=float((labels[d==c]==c).mean()) if (d==c).any() else 0
            r['eligible']=r['precision_8']==1 and r['precision_9']==1;grid.append(r)
    eligible=[r for r in grid if r['eligible']];chosen=max(eligible,key=lambda r:r['coverage']) if eligible else None
    (out/'train-calibration.json').write_text(json.dumps({'grid':grid,'chosen':chosen},indent=2),encoding='utf-8')
    if chosen is None:
        (out/'report.json').write_text(json.dumps({'protocol':protocol,'gate_passed':False,'reason':'no train-eligible threshold pair'},indent=2),encoding='utf-8');print('No train-eligible pair',flush=True);return
    with np.load(ROOT/'data/processed/radial-evidence-v1/validation.npz') as z:raw=z['observed_images'];aligned=z['images'];gold=z['program']
    with np.load(ROOT/'output/radial-evidence-evaluation/validation-predictions.npz') as z:base=z['baseline']
    geo=json.loads((ROOT/'data/processed/radial-evidence-v1/validation-geometry.json').read_text());pred=base.copy();changes=[];rule={'low':chosen['low'],'high':chosen['high']}
    for i,g in enumerate(geo):
        score,slots=page_scores(raw[i,0],aligned[i,0],g);ds=decisions(score,**rule)
        for s,d in zip(slots,ds):
            if d!=7 and pred[i,s]!=7 and d!=pred[i,s]:changes.append({'case_index':i,'slot':s,'correct_after':bool(d==gold[i,s])});pred[i,s]=d
    before=(base==gold).all(1);after=(pred==gold).all(1);net=float(np.mean([exact_net(p,g) for p,g in zip(pred,gold)]));original=float(np.mean([exact_net(p,g) for p,g in zip(base,gold)]))
    r={'protocol':protocol,'thresholds':rule,'changed_slots':len(changes),'wrong_changed_slots':sum(not d['correct_after'] for d in changes),'fixed_programs':int((~before&after).sum()),'broken_programs':int((before&~after).sum()),'exact_net':net,'baseline_exact_net':original,'changes':changes}
    r['gate_passed']=r['changed_slots']>0 and r['wrong_changed_slots']==0 and r['broken_programs']==0 and net>=original
    (out/'report.json').write_text(json.dumps(r,indent=2),encoding='utf-8');np.savez_compressed(out/'dev-predictions.npz',program=pred);print(json.dumps(r),flush=True)


if __name__=='__main__':main()
