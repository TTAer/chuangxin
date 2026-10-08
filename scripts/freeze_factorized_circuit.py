"""Preserve the controlled experiment and an exact-image electrical ambiguity witness."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scripts.prepare_factorized_circuit import render_native
from scripts.prepare_circuit_program import execute


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    dest=ROOT/'output/factorized-circuit-research-manifest.json'
    if dest.exists():raise ValueError('Preserve research freeze')
    base=np.full(68,7,np.int64);base[:2]=6;base[2:38]=8;base[38:]=10
    changed=base.copy();changed[16]=9
    a,b=render_native(base,7),render_native(changed,7);ra,rb=execute(base)[1],execute(changed)[1]
    assert np.array_equal(a,b) and not np.array_equal(ra,rb)
    out=ROOT/'output/factorized-circuit-audit'
    np.savez_compressed(out/'collision-witness.npz',programs=np.stack((base,changed)),images=np.stack((a,b)),relations=np.stack((ra,rb)))
    Image.fromarray(a).save(out/'collision-identical.png')
    previous=json.loads((ROOT/'output/circuit-program-research-manifest.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in previous['sha256'].items())
    rc0=json.loads((ROOT/'data/splits/rc0-evaluation.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in rc0['runtime_sha256'].items())
    package=ROOT/'submission/graphschematic-rc0-evaluation.zip'
    assert sha(package)=='04ef04248830666d6a4020c4ee77a73b82d130b4632d9a8e5946a90f996586b4'
    files=set()
    for pattern in ('experiments/factorized_circuit_reader.py','scripts/*factorized_circuit*.py','docs/FACTORIZED_CIRCUIT_RESULTS.md','output/factorized-circuit*.log'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    for folder in ('data/processed/factorized-circuit-v1','output/training/factorized-circuit-v1','output/factorized-circuit-audit','output/factorized-circuit-demo'):
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    result={'scope':'native256 fixed renderer, independent slot CNN, identifiability and template audit',
            'status':'FROZEN_RESEARCH_ONLY_STRESS_NO_GO','prior_research_sha_count':len(previous['sha256']),
            'prior_research_unchanged':True,'RC0_runtime_checked':len(rc0['runtime_sha256']),
            'RC0_package_unchanged':True,'real_verified_designs':0,'sha256':hashes}
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    print(json.dumps({'frozen_files':len(hashes),'old_research_preserved':len(previous['sha256']),'RC0_preserved':True}))


if __name__=='__main__':main()
