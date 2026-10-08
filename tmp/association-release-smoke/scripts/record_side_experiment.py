"""Record new data availability, architecture ablations and explicit hold policy."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
def save(path,value):
    p=ROOT/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(value,indent=2),encoding='utf-8')
groups={}
for name in ('base','mixed','local'):
    root='models/candidates/side-strip-'+name
    groups[name]={'model':load(root+'/side-strip.json'),'development':{d:load('output/side-strip-side-strip-'+name+'-'+d+'/summary.json') for d in ('gt10','pcb55')}}
record={'architecture':'SideStripNet','groups':groups,'default_enabled':False,'function_tests_passed':76,
        'policy':'hold: single-seed short experiment; local control beats dilated model; modest PCB55 gain and GT10 gain concentrated in one page',
        'data_preparation':load('data/processed/side-strips/preparation.json'),'export_verification':load('output/side-strip-verification/exports.json'),
        'live_verification':load('output/side-strip-verification/live.json'),
        'without_text_structure':{d:load('output/side-strip-side-strip-mixed-'+d+'-raw/summary.json') for d in ('gt10','pcb55')},
        'official_equivalence':False,'frozen_pool_used':False,'caveats':['equal epochs are unequal optimization steps between base and mixed','oracle-band F1 is not whole-image pin-number F1','synthetic families are layouts, not real independent electrical designs']}
save('output/ablation/side-strip.json',record)
source=load('data/raw/CircuitOCR/subset-manifest.json'); summary={k:v for k,v in source.items() if k!='records'}
summary['records']=[{k:v for k,v in r.items() if k!='transcript'} for r in source['records']]
summary['public_metadata']=load('data/raw/CircuitOCR/source-manifest.json')
summary['synthetic_source']=load('data/processed/side-strips/preparation.json')
summary['pixel_hash_definition']='SHA256(str(decoded BGR array shape)+decoded pixel bytes), same as official inventory'
save('output/side-source-audit.json',summary)
print(json.dumps({'default_enabled':False,'source_pairs':source['downloaded'],'groups':list(groups)}),flush=True)
