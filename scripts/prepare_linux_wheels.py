"""Verified PyPI wheelhouse for CPython 3.12, glibc Linux x86_64.

Bounded parallel HTTP ranges keep slow WSL downloads out of the Docker build.
No source distributions or downloaded code are executed by this script.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.request
from urllib.parse import urlparse
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'vendor/linux-wheels'
PACKAGES={'numpy':'2.2.6','opencv-python':'5.0.0.93','pillow':'11.3.0','rapidocr_onnxruntime':'1.4.4','onnxruntime':'1.30.0','pyclipper':'1.4.0','Shapely':'2.1.2','PyYAML':'6.0.3','six':'1.17.0','tqdm':'4.70.1','flatbuffers':'25.12.19','packaging':'26.3','protobuf':'7.36.2'}
HEADERS={'User-Agent':'GraphSchematic-wheelhouse/1.0'}
def request(url,headers=()): return urllib.request.urlopen(urllib.request.Request(url,headers={**HEADERS,**dict(headers)}),timeout=45)
def choose(item):
    name=item['filename']
    if not name.endswith('.whl') or '/' in name or '\\' in name: return 0
    if name.endswith('none-any.whl'): return 100
    if 'manylinux' not in name or 'x86_64' not in name: return 0
    if not any(tag in name for tag in ('manylinux2014','manylinux_2_17','manylinux_2_27','manylinux_2_28')): return 0
    if '-cp312-cp312-' in name: return 90
    if re.search(r'-cp(?:37|38|39|310|311|312)-abi3-',name): return 80
    return 0
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    BASE.mkdir(parents=True,exist_ok=True); selected=[]; jobs=[]
    for package,version in PACKAGES.items():
        with request(f'https://pypi.org/pypi/{package}/{version}/json') as response: metadata=json.load(response)
        options=[item for item in metadata['urls'] if choose(item)]
        if not options: raise ValueError('No supported binary wheel: '+package)
        item=max(options,key=choose); name=item['filename']; target=(BASE/name).resolve()
        if not target.is_relative_to(BASE.resolve()) or urlparse(item['url']).hostname!='files.pythonhosted.org': raise ValueError('unexpected wheel path/host')
        size=item['size']; checksum=item['digests']['sha256']
        if not 0<size<120*1024**2 or not re.fullmatch('[0-9a-f]{64}',checksum): raise ValueError('invalid size/hash')
        record={'package':package,'version':version,'file':name,'bytes':size,'sha256':checksum,'url':item['url']}; selected.append(record)
        if target.exists() and target.stat().st_size==size and sha(target)==checksum: continue
        parts=BASE/'.parts'/checksum; parts.mkdir(parents=True,exist_ok=True)
        for start in range(0,size,2*1024**2): jobs.append((record,start,min(size-1,start+2*1024**2-1),parts/str(start)))
    def fetch(job):
        record,start,end,path=job
        if path.exists() and path.stat().st_size==end-start+1: return
        for attempt in range(4):
            try:
                with request(record['url'],{'Range':f'bytes={start}-{end}'}.items()) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f"bytes {start}-{end}/{record['bytes']}": raise ValueError('exact HTTP range required')
                    data=response.read(end-start+1)
                if len(data)!=end-start+1: raise OSError('incomplete chunk')
                path.write_bytes(data); return
            except (urllib.error.URLError,OSError):
                if attempt==3: raise
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(fetch,job) for job in jobs]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            if i%10==0: print(json.dumps({'completed_chunks':i,'total_chunks':len(jobs)}),flush=True)
    for record in selected:
        target=BASE/record['file']; checksum=record['sha256']
        if target.exists() and target.stat().st_size==record['bytes'] and sha(target)==checksum: continue
        staged=BASE/(record['file']+'.assembling')
        with staged.open('wb') as output:
            for start in range(0,record['bytes'],2*1024**2): output.write((BASE/'.parts'/checksum/str(start)).read_bytes())
        if staged.stat().st_size!=record['bytes'] or sha(staged)!=checksum: raise ValueError('wheel SHA256 mismatch: '+record['file'])
        staged.replace(target)
    result={'target':'CPython 3.12 glibc Linux x86_64','source':'PyPI JSON and files.pythonhosted.org; SHA256 verified','packages':selected,'total_bytes':sum(r['bytes'] for r in selected)}
    (BASE/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps({'verified_wheels':len(selected),'bytes':result['total_bytes']}),flush=True)
if __name__=='__main__': main()
