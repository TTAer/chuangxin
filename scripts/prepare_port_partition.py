"""Weak four-port relation targets from existing development wire groups."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from pcbparse.dataset import index_dataset
from experiments.port_partition import port_membership, compatible_partitions


def main():
    out = ROOT/'data/processed/port-partition-v1'
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT/'data/processed/topology'
    records = json.loads((source/'records.json').read_text(encoding='utf-8'))
    index = {r['id']: r for r in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}
    z = np.load(source/'patches.npz')
    images, targets, masks, rows, skipped = [], [], [], [], Counter()
    page_cache = {}
    for i, r in enumerate(records):
        page = r['page']
        if page not in page_cache:
            item = index[page]
            raw = json.loads(Path(item['annotation']).read_text(encoding='utf-8'))
            segments = []
            for group, edge in raw['edges'].items():
                if edge.get('state') not in (None, 'active') or edge.get('isFlywire'):
                    continue
                for path in edge.get('keyPointPaths', {}).get('wire', []):
                    segments.extend((group, a, b) for a, b in zip(path, path[1:]) if a != b)
            page_cache[page] = segments
        target, valid, owners = port_membership(page_cache[page], r['point'])
        if valid.sum() < 3:
            skipped['fewer_than_three_known_pair_relations'] += 1
            continue
        if not compatible_partitions(target, valid).any():
            raise ValueError('Inconsistent annotation partition')
        images.append(z['images'][i]); targets.append(target); masks.append(valid)
        rows.append({**r, 'port_wire_groups': owners,
                     'image_sha256': index[page]['sha256'],
                     'annotation_sha256': hashlib.sha256(Path(index[page]['annotation']).read_bytes()).hexdigest()})
    np.savez_compressed(out/'relations.npz', images=np.asarray(images),
                        targets=np.asarray(targets), valid=np.asarray(masks),
                        split=np.asarray([r['split']=='validation' for r in rows], np.int64))
    (out/'records.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    meta = {'schema': 'port-partition-preparation-v1', 'samples': len(rows),
            'split_counts': dict(Counter(r['split'] for r in rows)),
            'pages': {s: len({r['image_sha256'] for r in rows if r['split']==s}) for s in ('train','validation')},
            'skipped': dict(skipped), 'known_relations': int(np.asarray(masks).sum()),
            'source': 'Existing topology crossing candidate cohort; labels re-derived from physical wire group paths',
            'supervision': 'weak source annotations; ambiguous/absent ports unsupervised; no pin point targets',
            'scope': 'conditional local physical connectivity; does not learn flying-label electrical equivalence',
            'split_policy': 'existing exact-image hash split; structure-family independence NOT proven; validation is development',
            'Freeze_A_used': False, 'GT10_used': False,
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'patches.npz', source/'records.json')}}
    train = {r['image_sha256'] for r in rows if r['split']=='train'}
    val = {r['image_sha256'] for r in rows if r['split']=='validation'}
    if train & val:
        raise ValueError('Exact image leakage')
    (out/'preparation.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
