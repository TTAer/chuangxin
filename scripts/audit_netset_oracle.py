"""Reproducible capacity, source, cache and frozen-baseline audit; no training."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pcbparse.core import assignment,f1


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    bank=ROOT/'data/processed/netset-oracle';out=ROOT/'output/training/netset-oracle'
    manifest=json.loads((bank/'manifest.json').read_text(encoding='utf-8'))
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    assert sha(bank/'manifest.json')==protocol['data_sha256']
    frozen=[]
    for file,key in [('data/splits/rc0-evaluation.json','runtime_sha256'),('output/netset-query-prototype-manifest.json','sha256'),('output/topology-agreement-manifest.json','sha256')]:
        data=json.loads((ROOT/file).read_text(encoding='utf-8'));fail=[p for p,h in data[key].items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
        frozen.append({'manifest':file,'files':len(data[key]),'mismatches':fail});assert not fail,(file,fail)
    expected='04ef04248830666d6a4020c4ee77a73b82d130b4632d9a8e5946a90f996586b4'
    assert sha(ROOT/'submission/graphschematic-rc0-evaluation.zip')==expected
    assert all(sha(ROOT/r['annotation'])==r['annotation_sha256'] for r in manifest['rows'])
    valid=[r['nets'] for r in manifest['capacity_all200']['rows'] if r['valid_schema']]
    capacity={'all200':{k:v for k,v in manifest['capacity_all200'].items() if k!='rows'},
              'valid_schema_only':{'pages':len(valid),'max':max(valid),'P95':float(np.percentile(valid,95)),'P99':float(np.percentile(valid,99))},
              'admitted':{'pages':len(manifest['rows']),'max':max(r['net_count'] for r in manifest['rows'])},
              'note':'schema validity and complete ownership do not establish human-verified annotation correctness'}
    details=[]
    for variant in protocol['variants']:
        pages=[]
        for row in manifest['rows']:
            with np.load(bank/row['case']/'oracle.npz') as d:gold=d['pin_labels']
            with np.load(out/variant/'predictions'/(row['case']+'.npz')) as d:pred=d['pin_logits'].argmax(1);q=d['pin_logits'].shape[1]-1
            sets=[set(np.flatnonzero(pred==s)) for s in sorted(set(pred.tolist())-{q})]
            truth=[set(np.flatnonzero(gold==g)) for g in sorted(set(gold.tolist()))]
            pairs=assignment([[f1(len(a&b),len(a),len(b)) for b in truth] for a in sets])
            exact=sum(sets[i]==truth[j] for i,j in pairs)
            partial=sum(bool(sets[i]&truth[j]) and sets[i]!=truth[j] for i,j in pairs)
            pages.append({'case':row['case'],'split':row['split'],'exact_matched_nets':exact,'partial_matched_nets':partial,
                          'unmatched_or_zero_overlap_true_nets':len(truth)-exact-partial,
                          'hungarian_net_Dice_over_all_true_nets':sum(f1(len(sets[i]&truth[j]),len(sets[i]),len(truth[j])) for i,j in pairs)/len(truth)})
        details.append({'variant':variant,'summary':{s:{k:float(np.mean([r[k] for r in pages if r['split']==s])) for k in pages[0] if k not in ('case','split')} for s in ('train','development_holdout')},'pages':pages})
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    result={'capacity':capacity,'source_annotation_hashes_checked':len(manifest['rows']),'frozen_checks':frozen,'RC0_zip_sha256':expected,
            'coverage_curve':'not_executed_first_gate_failed','real_observation_oracle':'not_executed_first_gate_failed','weak_label_training':'not_executed_first_gate_failed',
            'simple_reference_baselines':{k:float(np.mean([r[k] for r in summary['baselines']])) for k in ('singleton_micro_F1','all_one_micro_F1')},
            'singleton_reference_exceeds_Q_pages':sum(r['pin_count']>manifest['Q'] for r in manifest['rows'] if r['split']=='development_holdout'),
            'baseline_note':'singleton is an abstract partition reference, not a Q-constrained decoder',
            'net_details':details,'independent_validation':False,'official_evaluator_equivalence':False,'runtime_changed':False}
    (out/'final-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='net_details'}))


if __name__=='__main__':main()
