"""Bounded author downloads with ZIP path checks and per-file SHA256.

CGHD selects aligned segmentation samples via HTTP Range without downloading
the entire 4.9 GB archive; extracted members still receive ZIP CRC validation.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import urllib.error
import http.client
import zipfile
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
HEADERS={'User-Agent':'GraphSchematic-Research/1.0'}

class RemoteZip(io.RawIOBase):
    def __init__(self,url,size): self.url=url; self.size=size; self.position=0; self.start=-1; self.cached=b''; self.downloaded=0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if self.position<0: raise ValueError('negative offset')
        return self.position
    def read(self,n=-1):
        n=min(self.size-self.position,n if n>=0 else self.size-self.position)
        if n<=0: return b''
        if self.start<=self.position and self.position+n<=self.start+len(self.cached):
            result=self.cached[self.position-self.start:self.position-self.start+n]
        else:
            end=min(self.size-1,self.position+max(n,65536)-1)
            req=urllib.request.Request(self.url,headers={**HEADERS,'Range':f'bytes={self.position}-{end}'})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req,timeout=45) as r:
                        expected=f'bytes {self.position}-{end}/{self.size}'
                        if r.status!=206 or r.headers.get('Content-Range')!=expected: raise ValueError('server did not honor exact range')
                        self.cached=r.read(end-self.position+1)
                except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead):
                    if attempt == 2: raise
                    continue
                if len(self.cached)==end-self.position+1: break
            if len(self.cached)!=end-self.position+1: raise ValueError('incomplete range after retries')
            self.start=self.position; self.downloaded+=len(self.cached); result=self.cached[:n]
        self.position+=len(result); return result

def safe_write(base,name,data):
    path=(base/name).resolve()
    if not path.is_relative_to(base.resolve()): raise ValueError('unsafe archive path')
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
    return {'file':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

def cghd(limit):
    meta=json.loads((ROOT/'output/new-dataset-research/cghd_v16.json').read_text(encoding='utf-8'))
    item=meta['files'][0]; base=ROOT/'data/raw/CGHD-v16-subset'; base.mkdir(parents=True,exist_ok=True)
    remote=RemoteZip(item['links']['self'],item['size']); files=[]; records=[]
    with zipfile.ZipFile(remote) as z:
        names=z.namelist(); lookup=set(names)
        (base/'archive-file-list.json').write_text(json.dumps(names),encoding='utf-8')
        # Keep code/docs for schema inspection only; never execute downloaded code.
        docs=[n for n in names if n.rsplit('/',1)[-1] in ('README.md','LICENSE','classes.json','classes_ports.json','classes_color.json')]
        for name in docs: files.append(safe_write(base,name,z.read(name)))
        selected=sorted(n for n in names if '/segmentation/' in n and Path(n).suffix.lower() in ('.png','.jpg','.jpeg'))
        # Spread selection across drafters rather than taking an alphabetical block.
        groups={}
        for name in selected: groups.setdefault(name.split('/segmentation/')[0],[]).append(name)
        selected=[]
        for i in range(max(map(len,groups.values()),default=0)):
            for group in sorted(groups):
                if i<len(groups[group]): selected.append(groups[group][i])
        jobs=[]
        for seg in selected[:limit]:
            prefix,tail=seg.split('/segmentation/'); stem=Path(tail).stem
            image=next((n for n in names if n.startswith(prefix+'/images/') and Path(n).stem==stem),None)
            annotation=prefix+'/annotations/'+stem+'.xml'; instance=prefix+'/instances/'+stem+'.json'
            if not image or annotation not in lookup: continue
            jobs.append(({'image':image,'segmentation':seg,'annotation':annotation,'instances':instance if instance in lookup else None,'drafter':prefix.rsplit('/',1)[-1],'sample':stem},[z.getinfo(name) for name in [image,seg,annotation]+([instance] if instance in lookup else [])]))
        def fetch(job):
            record,infos=job; rows=[]; downloaded=0
            for info in infos:
                path=(base/info.filename).resolve()
                if not path.is_relative_to(base.resolve()): raise ValueError('unsafe path')
                data=path.read_bytes() if path.exists() else b''
                if len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:
                    if info.compress_size>30*1024**2 or info.file_size>100*1024**2: raise ValueError('member exceeds budget')
                    reader=RemoteZip(item['links']['self'],item['size']); reader.seek(info.header_offset)
                    # Fetch header and compressed member in one validated range.
                    packet=reader.read(30+len(info.filename.encode('utf-8'))+65536+info.compress_size)
                    header=struct.unpack('<4s5H3I2H',packet[:30])
                    if header[0]!=b'PK\x03\x04' or header[2]&1: raise ValueError('invalid/encrypted ZIP member')
                    offset=30+header[-2]+header[-1]; compressed=packet[offset:offset+info.compress_size]
                    data=zlib.decompress(compressed,-15) if info.compress_type==zipfile.ZIP_DEFLATED else compressed if info.compress_type==zipfile.ZIP_STORED else None
                    if data is None or len(data)!=info.file_size or zlib.crc32(data)!=info.CRC: raise ValueError('ZIP member CRC failed')
                    downloaded+=reader.downloaded
                rows.append(safe_write(base,info.filename,data))
            return record,rows,downloaded
        with ThreadPoolExecutor(max_workers=3) as executor:
            for record,rows,downloaded in executor.map(fetch,jobs):
                files.extend(rows); records.append(record); remote.downloaded+=downloaded
                (base/'records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
                if len(records)%10==0: print(json.dumps({'source':'CGHD','paired_samples':len(records),'downloaded_bytes':remote.downloaded}),flush=True)
    manifest={'source':'https://zenodo.org/records/17469897','version':16,'author_license':meta['metadata']['license']['id'],'archive_readme_license':'CC BY-SA 3.0','license_status':'Zenodo metadata and archive README differ; preserve both notices; raw data not redistributed','download_mode':'HTTP Range subset; not full corpus','advertised_images':3269,'local_paired_samples':len(records),'archive_bytes':item['size'],'archive_checksum_advertised':item['checksum'],'full_archive_checksum_verified':False,'extracted_crc_verified':True,'downloaded_bytes':remote.downloaded,'files':files}
    (base/'download.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in manifest.items() if k!='files'}),flush=True)

def amsnet():
    url='https://raw.githubusercontent.com/AMS-Net/ams-net.github.io/main/amsnet_1.0-20240412T024532Z-001.zip'
    base=ROOT/'data/raw/AMSNet-1.0'; base.mkdir(parents=True,exist_ok=True); target=base/'amsnet.zip'
    digest=hashlib.sha256(); size=0
    with urllib.request.urlopen(urllib.request.Request(url,headers=HEADERS),timeout=60) as r, target.open('wb') as f:
        while chunk:=r.read(1024**2):
            size+=len(chunk)
            if size>300*1024**2: raise ValueError('archive budget exceeded')
            digest.update(chunk); f.write(chunk)
    files=[]
    with zipfile.ZipFile(target) as z:
        if sum(x.file_size for x in z.infolist())>1024**3: raise ValueError('expanded archive budget exceeded')
        for member in z.infolist():
            if member.is_dir(): continue
            files.append(safe_write(base,member.filename,z.read(member)))
    for name in ('README.md','LICENSE'):
        url2='https://raw.githubusercontent.com/AMS-Net/ams-net.github.io/main/'+name
        with urllib.request.urlopen(urllib.request.Request(url2,headers=HEADERS),timeout=30) as r: files.append(safe_write(base,name,r.read()))
    images=[f for f in files if Path(f['file']).suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff')]
    manifest={'source':'https://github.com/AMS-Net/ams-net.github.io','version':'1.0 archive','repository_license':'GPL-3.0; third-party circuit figure rights not independently cleared','training_status':'inspection only until data provenance/annotation mapping checked','archive_bytes':size,'archive_sha256':digest.hexdigest(),'image_files':len(images),'files':files}
    (base/'download.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in manifest.items() if k!='files'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('source',choices=['cghd','amsnet']); p.add_argument('--limit',type=int,default=160); a=p.parse_args()
    if a.source=='cghd': cghd(a.limit)
    else: amsnet()
