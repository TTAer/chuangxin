"""Download the author's AITEE archive with hashes and guarded extraction."""
import hashlib
import json
from pathlib import Path
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / 'data/raw/AITEE'
base.mkdir(parents=True, exist_ok=True)
url = 'https://raw.githubusercontent.com/CKnievel/aitee-dataset/main/aitee_dataset.zip'
target = base / 'aitee_dataset.zip'
with requests.get(url, stream=True, timeout=(20, 60)) as response:
    response.raise_for_status()
    with target.open('wb') as handle:
        size = 0
        for chunk in response.iter_content(1024 * 1024):
            size += len(chunk)
            if size > 500 * 1024 ** 2:
                raise ValueError('archive too large')
            handle.write(chunk)
if not zipfile.is_zipfile(target):
    raise ValueError('not a ZIP archive (possibly Git LFS pointer)')
with zipfile.ZipFile(target) as archive:
    if sum(i.file_size for i in archive.infolist()) > 2 * 1024 ** 3:
        raise ValueError('expanded archive too large')
    for member in archive.namelist():
        if not (base / member).resolve().is_relative_to(base.resolve()):
            raise ValueError('unsafe archive path')
    archive.extractall(base)
for name in ('LICENSE', 'README.md', 'classes.txt'):
    response = requests.get('https://raw.githubusercontent.com/CKnievel/aitee-dataset/main/' + name, timeout=30)
    response.raise_for_status()
    (base / name).write_bytes(response.content)
manifest = {'source': url, 'license': 'Apache-2.0', 'bytes': target.stat().st_size,
            'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
(base / 'download.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest))
