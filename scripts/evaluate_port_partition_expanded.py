"""Fixed old/new ONNX comparison on two development cohorts."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import onnxruntime as ort
from experiments.port_partition import RELATIONS


def score(pred,target,valid,selected=None):
    mask=valid.astype(bool).copy();mask[:,[0,5]]=False
    if selected is not None:mask &= selected
    tp=int(((pred==1)&(target==1)&mask).sum())
    fp=int(((pred==1)&(target==0)&mask).sum())
    fn=int(((pred==0)&(target==1)&mask).sum())
    neg=int(((target==0)&mask).sum())
    return {'f1':2*tp/max(1,2*tp+fp+fn),'tp':tp,'fp':fp,'fn':fn,
            'merge_precision':tp/(tp+fp) if tp+fp else None,
            'false_merge_rate_on_negative_relations':fp/neg if neg else None,
            'known_cross_relations':int(mask.sum())}


def selected_merge_stats(weight,target,valid,threshold):
    known=valid.astype(bool).copy();known[:,[0,5]]=False
    chosen=(weight>=threshold)&known
    tp=int((chosen&(target==1)).sum());fp=int((chosen&(target==0)).sum())
    positives=int((known&(target==1)).sum());negatives=int((known&(target==0)).sum())
    return {'selected_merges':tp+fp,'tp':tp,'fp':fp,
            'merge_precision':tp/(tp+fp) if tp+fp else None,
            'false_merge_fraction_selected':fp/(tp+fp) if tp+fp else None,
            'false_merge_rate_all_known_negatives':fp/negatives if negatives else None,
            'connected_relation_recall':tp/positives if positives else None,
            'selection_coverage':(tp+fp)/int(known.sum()) if known.sum() else None,
            'abstained_positive_relations':positives-tp}


def main():
    out=ROOT/'output/training/port-partition-expanded'
    with np.load(ROOT/'data/processed/port-partition-expanded/relations.npz') as archive:
        z={k:archive[k] for k in archive.files}
    rows=[];exports=[]
    for expanded in (False,True):
        for seed in range(3):
            name=f'expanded-seed{seed}' if expanded else f'matched-seed{seed}'
            model=ROOT/'models/candidates'/('port-partition-expanded' if expanded else 'port-partition-v1')/name/'port-graph/model.onnx'
            options=ort.SessionOptions();options.intra_op_num_threads=2
            session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
            if expanded:
                ref=np.load(out/name/'port-graph/export-reference.npz')
                actual=session.run(None,{'image':ref['image']})[0]
                if not np.allclose(actual,ref['logits'],atol=1e-4,rtol=1e-4):raise ValueError('Export mismatch')
                exports.append({'model':str(model.relative_to(ROOT)),'max_abs_error':float(np.abs(actual-ref['logits']).max())})
            for code in (1,2):
                ids=np.flatnonzero(z['split']==code)
                if not len(ids):continue
                logits=np.concatenate([session.run(None,{'image':z['images'][part].astype(np.float32)/255})[0]
                                       for part in np.array_split(ids,max(1,int(np.ceil(len(ids)/128))))])
                weights=np.exp(logits-logits.max(1,keepdims=True));weights/=weights.sum(1,keepdims=True)
                pred=RELATIONS[logits.argmax(1)]
                marginal=weights@RELATIONS
                row={'model':str(model.relative_to(ROOT)),'expanded':expanded,'seed':seed,
                     'cohort':'original-fixed-133' if code==1 else 'pixel-only-automatic-development',
                     'samples':len(ids),'map':score(pred,z['targets'][ids],z['valid'][ids]),
                     'high_weight_merge':{str(t):selected_merge_stats(marginal,z['targets'][ids],z['valid'][ids],t) for t in (.9,.95)}}
                rows.append(row)
    result={'schema':'expanded-port-partition-evaluation-v1','results':rows,'export_verification':exports,
            'thresholds':'fixed descriptive .9/.95 marginal weights; NOT calibrated or selected for commit',
            'evaluation':'both cohorts development; automatic candidates filtered for available weak supervision; not candidate precision/recall or independent accuracy',
            'runtime_changed':False,'commit_authorized':False}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for cohort in ('original-fixed-133','pixel-only-automatic-development'):
        for expanded in (False,True):
            rs=[r for r in rows if r['cohort']==cohort and r['expanded']==expanded]
            print(json.dumps({'cohort':cohort,'expanded':expanded,'mean_f1':float(np.mean([r['map']['f1'] for r in rs])),
                              'fp':[r['map']['fp'] for r in rs],'precision':[r['map']['merge_precision'] for r in rs]}))


if __name__=='__main__':main()
