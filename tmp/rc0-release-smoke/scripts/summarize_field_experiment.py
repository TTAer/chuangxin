import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
models={}
for name in ('directional-fields-base','directional-fields','directional-fields-local'):
    d=load(f'models/candidates/{name}/field.json')
    models[name]={k:d[k] for k in ('parameters','train_tiles','validation_tiles','best_epoch','strip_context','centerline_loss')}
baseline=load('output/benchmark-official-pin-consensus-live/summary.json')
benchmarks={}
for name in ('base','multisource'):
    d=load('output/benchmark-official-fields'+('-base' if name=='base' else '')+'/summary.json')
    comparisons=[]
    original={r['case']:r for r in baseline['records']}
    for r in d['records']:
        comparisons.append({'case':r['case'],'delta':r['Score_Accuracy_SingleCase_proxy']-original[r['case']]['Score_Accuracy_SingleCase_proxy']})
    benchmarks[name]={'mean_metrics':d['mean_metrics'],'wall_seconds':d['wall_seconds'],'completed':d['completed'],'delta_proxy':d['mean_metrics']['Score_Accuracy_SingleCase_proxy']-baseline['mean_metrics']['Score_Accuracy_SingleCase_proxy'],'improved':sum(r['delta']>1e-9 for r in comparisons),'regressed':sum(r['delta']< -1e-9 for r in comparisons),'same':sum(abs(r['delta'])<=1e-9 for r in comparisons),'per_case':comparisons}
summary={'models':models,'baseline_gt10':baseline['mean_metrics'],'gt10':benchmarks,'verification':{k:load(f'output/field-verification/{k}.json') for k in ('base','multisource','local')},'default_enabled':False,'official_equivalence':False,'independent_final_test':False,'architectural_control':'one seed; unequal parameter counts','data_control':'equal epochs but adding CGHD increases optimizer steps','decision':'keep optional; no robust end-to-end improvement; tip and junction fields remain diagnostic only'}
out=ROOT/'output/ablation/directional-fields.json'; out.write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps({'gt10':benchmarks,'output':str(out)}),flush=True)
