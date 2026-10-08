"""Check published wheel hashes before an offline install."""
import argparse
import hashlib
import json
from pathlib import Path
def verify(directory):
    base=Path(directory).resolve(); manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    for row in manifest['packages']:
        path=(base/row['file']).resolve()
        if not path.is_relative_to(base) or path.suffix!='.whl': raise ValueError('unsafe wheel path')
        if not path.is_file() or path.stat().st_size!=row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']: raise ValueError('wheel hash mismatch: '+row['file'])
    return {'verified_wheels':len(manifest['packages']),'bytes':sum(r['bytes'] for r in manifest['packages'])}
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('directory',nargs='?',default='vendor/linux-wheels'); a=p.parse_args(); print(json.dumps(verify(a.directory)))
