"""Eligible official TRAIN200 full oracle, preserving original image split."""
import json,sys,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pcbparse.core import validate,references
from pcbparse.vision import read_image
from scripts.prepare_netset_query_prototype import tokens
from scripts.prepare_masked_stroke_data import pixel_hash


def main():
    out=ROOT/'data/processed/netset-oracle';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Preserve oracle manifest')
    rows=json.loads((ROOT/'output/official-audit/inventory.json').read_text(encoding='utf-8'))['details']['train200']['records']
    blocked={pixel_hash(read_image(p)) for p in (ROOT/'data/raw/official/gt-0930/10GT').glob('*/*.png')}
    seen=set();eligible=[];excluded=[];capacity=[]
    for r in rows:
        g=json.loads((ROOT/r['annotation']).read_text(encoding='utf-8'));capacity.append({'case':r['case'],'nets':len(g['nets']),'valid_schema':not bool(validate(g))})
        reason=None;errors=validate(g)
        refs={c+'.'+p[4:] for c,ps in g['pins'].items() for p in ps};owners=Counter(ref for n in g['nets'].values() for ref in references(n['hyperGraph'])) if not errors else Counter()
        size=len(refs)+sum(len(n['edges']) for n in g['nets'].values())
        if errors:reason='invalid_schema'
        elif any(owners[x]!=1 for x in refs):reason='incomplete_pin_membership'
        elif len(g['nets'])>128:reason='query_overflow'
        elif size>512:reason='explicit_resource_scope_above512_tokens'
        elif len(refs)<2 or len(g['nets'])<2:reason='trivial'
        if reason:excluded.append({'case':r['case'],'reason':reason});continue
        image=read_image(ROOT/r['image']);h,w=image.shape[:2];digest=pixel_hash(image)
        if digest in blocked or digest in seen:
            excluded.append({'case':r['case'],'reason':'GT10_overlap' if digest in blocked else 'pixel_duplicate'});continue
        seen.add(digest);wires=[];wirelabels=[];groups={name:i for i,name in enumerate(g['nets'])};byref={}
        for name,net in g['nets'].items():
            for ref in references(net['hyperGraph']):byref[ref]=groups[name]
            for edge in net['edges'].values():wires.append(edge);wirelabels.append(groups[name])
        obs={'components':g['components'],'pins':g['pins'],'wires':wires,'image':{'width':w,'height':h},'contact_decisions':[]}
        ordered,pins,wiretokens=tokens(obs);labels=np.array([byref[x] for x in ordered],np.int64)
        dest=out/r['case'];dest.mkdir(exist_ok=True)
        np.savez_compressed(dest/'oracle.npz',pins=pins,wires=wiretokens,pin_labels=labels,wire_labels=np.array(wirelabels,np.int64))
        eligible.append({'case':r['case'],'split':'train' if r['training_overlap'] else 'development_holdout','pin_count':len(pins),'wire_count':len(wiretokens),'net_count':len(g['nets']),
                         'pixel_sha256':digest,'annotation':r['annotation'],'annotation_sha256':hashlib.sha256((ROOT/r['annotation']).read_bytes()).hexdigest()})
    counts=np.array([r['nets'] for r in capacity]);manifest={'rows':eligible,'excluded':excluded,'exclusion_counts':dict(Counter(r['reason'] for r in excluded)),
        'split_counts':dict(Counter(r['split'] for r in eligible)),
        'capacity_all200':{'max':int(counts.max()),'P95':float(np.percentile(counts,95)),'P99':float(np.percentile(counts,99)),'rows':capacity},
        'Q':128,'token_limit':512,'scope':'schema-complete eligible subset of existing train200; development holdout, NOT independent verified designs',
        'GT10_used':False,'Freeze_A_used':False,'new_data_downloaded':False,'targets':'complete per admitted schema, annotation correctness remains weak source assumption',
        'input_leakage_policy':'GT coordinates/identity form oracle observations; GT net IDs/membership/wire ownership only targets, never features; input token order has no positional embedding',
        'limitations':'invalid/incomplete/query-overflow pages excluded without repair; design-family independence unproved; labels/text not yet encoded'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in manifest.items() if k not in ('rows','excluded','capacity_all200')}))


if __name__=='__main__':main()
