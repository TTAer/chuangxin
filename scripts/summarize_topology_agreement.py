"""Reproducible summary tables and existing PPN seed sensitivity; no training."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,onnxruntime as ort
from scripts.audit_topology_agreement import infer,metrics,precision_lower95


def main():
    out=ROOT/'output/topology-agreement';file=out/'local-audit.json';r=json.loads(file.read_text(encoding='utf-8'))
    for v in r['summary'].values():
        for name,m in v.items():
            if isinstance(m,dict) and 'tp' in m and 'fp' in m:m['precision_lower_95_one_sided']=precision_lower95(m['tp'],m['fp'])
    notes={2152:'Plain black four-arm crossing; neither model requires an isolated dot here. Shared visual blind spot, not a justified relabel.',2576:'Nearby green orthogonal bends within the same crop; local crossing context contains multiple adjacent strokes.',2985:'Red dot on a green T; weak source-wire groups disagree with visible junction evidence. Keep suspected weak-label conflict unresolved.'}
    for row in r['shared_error_records']:
        row['crop']=str((out/'shared-error-crops'/(str(row['index'])+'.png')).relative_to(ROOT))
        if row['index'] in notes:row['agent_pixel_observation']=notes[row['index']]
    r['precision_interval_limit']='One-sided exact binomial diagnostic assumes independent relations; within-crop/page correlation violates that assumption. Not a family-level confidence guarantee.'
    file.write_text(json.dumps(r,indent=2),encoding='utf-8')
    lines=['# 三模型关系分歧矩阵','', 'C=连接；S=模型支持分开；U=不确定。RC0 U不是有监督split。每个候选计4条跨方向关系，unknown不计精度。模块级RC0对照使用未遮罩dot/T上下文，不能等同完整RC0候选精度。','']
    for code in ('1','2'):
        lines+=['## '+('固定133' if code=='1' else '自动668'),'', '| RC0 | PPN | MST | 关系数 | 已知连接 | 已知不连接 | unknown |','|---|---|---|---:|---:|---:|---:|']
        for key,v in sorted(r['summary'][code]['confidence_matrix'].items()):
            a,b,c=key.split('|');lines.append(f"| {a} | {b} | {c} | {v['relations']} | {v['known_positive']} | {v['known_negative']} | {v['unknown']} |")
        lines+=['']
    (out/'disagreement-matrix.md').write_text('\n'.join(lines),encoding='utf-8')
    with np.load(ROOT/'data/processed/port-partition-expanded/relations.npz') as ar:z={k:ar[k] for k in ar.files}
    with np.load(out/'local-predictions.npz') as ar:ids=ar['indices'];mst=ar['MST_weights']
    target=z['targets'][ids].astype(bool);known=z['valid'][ids].astype(bool);known[:,[0,5]]=False
    opt=ort.SessionOptions();opt.intra_op_num_threads=2;rows=[]
    for seed in (0,1,2):
        session=ort.InferenceSession(str(ROOT/f'models/candidates/port-partition-expanded/expanded-seed{seed}/port-graph/model.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
        _,p=infer(session,z['images'][ids].astype(np.float32)/255)
        for code in (1,2):
            mask=known&(z['split'][ids,None]==code)
            rows.append({'PPN_seed':seed,'MST_seed':20261006,'cohort':code,'joint_high':metrics((p>=.95)&(mst>=.95),target,mask)})
    (out/'existing-seed-sensitivity.json').write_text(json.dumps({'rows':rows,'no_new_training':True,'default_seed_unchanged':0,'single_MST_seed_limitation':True},indent=2),encoding='utf-8')
    print(json.dumps(rows))


if __name__=='__main__':main()
