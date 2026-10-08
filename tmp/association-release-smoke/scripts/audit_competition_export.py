"""Validate every development prediction against competition export schema."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.competition import export_competition,TYPES
from pcbparse.core import validate
rows=[]
for domain in ('gt10','pcb55'):
    for folder in sorted((ROOT/'output/designator-guarded'/domain).iterdir()):
        if not (folder/'prediction.json').exists(): continue
        obs=load(folder/'observations.json'); pred=export_competition(load(folder/'prediction.json'),obs['component_categories'])
        assert not validate(pred) and all(c['type'] in TYPES for c in pred['components'].values())
        rows.append({'domain':domain,'case':folder.name,'valid':True,'ambiguous_M_exported_as_other':sum(c['type']=='m' for c in obs['components'].values())})
save(ROOT/'output/ablation/designator-export.json',{'pages':len(rows),'failures':0,'rows':rows,'official_runner_verified':False})
print({'pages':len(rows),'failures':0,'M_fallback_pages':sum(r['ambiguous_M_exported_as_other']>0 for r in rows)},flush=True)
