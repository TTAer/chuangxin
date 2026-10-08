"""Read-only schema/source gate: component incidence is not pin partition."""
import json, hashlib
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'data/raw/CircuitDigitization-connectivity/ground_truth/real_nets_verified.json'
    data=json.loads(source.read_text(encoding='utf-8'))
    local=ROOT/'data/raw/CGHD-v16-subset';records=json.loads((local/'records.json').read_text(encoding='utf-8'))
    index=defaultdict(list)
    for record in records:index[record['sample']].append(record['image'])
    rows=[]
    for key,r in data.items():
        if key.startswith('_'):continue
        stem=key.removesuffix('_jpg').removesuffix('_png').removesuffix('_jpeg')
        occurrence=defaultdict(set);unknown=[];within_duplicates=[]
        for ni,net in enumerate(r['nets']):
            tokens=[(str(t[0]),str(t[1])) for t in net]
            if len(set(tokens))!=len(tokens):within_duplicates.append(ni)
            for cid,role in tokens:
                occurrence[(cid,role)].add(ni)
                if cid not in r['components']:unknown.append(cid)
        repeated=[{'component_id':cid,'role':role,'net_indices':sorted(nets)} for (cid,role),nets in occurrence.items() if len(nets)>1]
        candidates=index.get(stem,[]);sizes=[]
        from PIL import Image
        for path in candidates:
            with Image.open(local/path) as image:size=list(image.size)
            sizes.append({'path':str((local/path).relative_to(ROOT)),'size':size,'size_matches':size==r['img_wh']})
        rows.append({'key':key,'annotation_size':r['img_wh'],'net_count':len(r['nets']),
                     'candidate_name_matches':sizes,'identity_and_pixel_alignment_verified':False,
                     'cross_net_repeated_component_role':repeated,'unknown_component_references':sorted(set(unknown)),
                     'within_net_duplicate_indices':within_duplicates,'training_admissible':False})
    result={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_records':len(rows),
            'local_image_records':len(records),'records_with_name_candidate':sum(bool(r['candidate_name_matches']) for r in rows),
            'records_with_cross_net_repeated_component_role':sum(bool(r['cross_net_repeated_component_role']) for r in rows),
            'training_admissible':0,'independent_verified_designs_added':0,
            'schema_warning':'component e roles recur across nets: incidence hypergraph, not a partition of unique observable pin IDs; no union of component identities across nets',
            'image_warning':'704-square benchmark copies include resizes/flips/rotations/stroke maps; basename and equal dimensions are insufficient pixel/identity provenance',
            'records':rows}
    out=ROOT/'output/probeflow-data-intake/connectivity-admission-audit.json'
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='records'}))

if __name__=='__main__':main()
