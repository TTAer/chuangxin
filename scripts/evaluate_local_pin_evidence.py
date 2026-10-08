"""Image-only local OCR additions over frozen RC0 perception, post-inference GT."""
from pathlib import Path
import json
import hashlib
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from experiments.local_pin_evidence import LocalPinEvidence,commit_additions
from pcbparse.semantic_library import SemanticLibrary
from pcbparse.dataset import index_dataset
from pcbparse.vision import read_image
from pcbparse.core import evaluate,validate
from pcbparse.competition import evaluate_qa,export_competition
from pcbparse.__main__ import load,save

def main():
    protocol={'scope':'cached frozen RC0 geometry + actual bounded local OCR; existing output facts locked',
        'crop':'96x48 normal/axial; top/bottom swapped; 2x cubic; max resize; exact catalog number and visible name',
        'proposal_score_min':.7,'number_confidence_min':.95,'name_confidence_min':.9,'nearest_margin_pixels':4,
        'commit':'unique physical wire group, no old-net bridge, no old identity/geometry overwrite',
        'new_models_trained':False,'residual_authority':'proposal-only; prior failed gate forbids learned override of geometry',
        'gt_at_inference':False,'official_equivalence':False}
    out=ROOT/'output/local-pin-evidence'; out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists(): assert load(out/'protocol.json')==protocol
    else: save(out/'protocol.json',protocol)
    parser=LocalPinEvidence(SemanticLibrary()); samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}
    keys=['ComponentF1','PinF1','NetHypergraphF1','NetLineF1','Score_Accuracy_SingleCase_proxy']
    for domain in ('gt10','pcb55'):
        rows=[]
        for folder in sorted((ROOT/'output/designator-stable-catalog'/domain).iterdir()):
            if not (folder/'observations.json').exists(): continue
            obs=load(folder/'observations.json'); before=load(folder/'prediction.json'); case=folder.name
            path=ROOT/'data/raw/official/gt-0930/10GT'/case/(case+'.png') if domain=='gt10' else Path(samples[case]['image'])
            start=time.perf_counter(); proposals,report=parser.collect(obs,read_image(path)); pred,decisions=commit_additions(before,obs,proposals)
            assert not validate(pred)
            # Same proposal-only residual arm intentionally cannot authorize a switch.
            report.update(decisions=decisions,source_observations_sha256=hashlib.sha256((folder/'observations.json').read_bytes()).hexdigest(),
                existing_facts_locked=True,residual_commit_enabled=False)
            save(out/domain/case/'prediction.json',pred); save(out/domain/case/'report.json',report)
            # GT is first read after prediction, never used to crop, identify or commit.
            truth=load(path.with_suffix('.json') if domain=='gt10' else ROOT/'output/benchmark-recognition-pcb55'/case/'golden.json')
            scorer=evaluate_qa if domain=='gt10' else evaluate
            rows.append({'case':case,'baseline':scorer(before,truth),'local_ocr':scorer(pred,truth),'residual_disabled_control':scorer(pred,truth),
                'proposals':len(proposals),'commits':sum(p['committed'] for p in decisions),'crops':report['counts']['crops'],'seconds':time.perf_counter()-start})
            print(json.dumps({'domain':domain,**{k:rows[-1][k] for k in ('case','crops','proposals','commits')}}),flush=True)
            save(out/domain/'progress.json',{'completed':len(rows),'rows':rows})
        means={mode:{key:sum(r[mode][key] for r in rows)/len(rows) for key in keys} for mode in ('baseline','local_ocr','residual_disabled_control')}
        delta={key:means['local_ocr'][key]-means['baseline'][key] for key in keys}
        gate=delta['PinF1']>=.01 and delta['NetHypergraphF1']>0
        save(out/domain/'summary.json',{'domain':domain,'pages':len(rows),'means':means,'delta':delta,'gate_passed':gate,'default_promoted':False,
            'residual_arm':'same committed outputs by authority gate, not an active reranker ablation','crops':sum(r['crops'] for r in rows),'proposals':sum(r['proposals'] for r in rows),'commits':sum(r['commits'] for r in rows),'records':rows})
        print(json.dumps({'domain':domain,'delta':delta,'gate_passed':gate}),flush=True)
if __name__=='__main__': main()
