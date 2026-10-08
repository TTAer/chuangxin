import json
from pathlib import Path
import urllib.request

root=Path(__file__).resolve().parents[1]
sources={
 'cghd_range':'https://zenodo.org/api/records/17469897/files/cghd-zenodo-16.zip/content',
 'amsnet_zip':'https://api.github.com/repos/AMS-Net/ams-net.github.io/contents/amsnet_1.0-20240412T024532Z-001.zip',
 'netlistify_meta':'https://huggingface.co/api/datasets/hanky2397/schematic_images',
}
report={}
for name,url in sources.items():
    try:
        headers={'User-Agent':'GraphSchematic-Research/1.0'}
        if name=='cghd_range': headers['Range']='bytes=4876805041-4876870576'
        with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=30) as r:
            if name=='cghd_range':
                report[name]={'status':r.status,'range':r.headers.get('Content-Range'),'length':r.headers.get('Content-Length')}; r.read(64)
            else:
                d=json.load(r); report[name]={'status':r.status,'size':d.get('size'),'download_url':d.get('download_url'),'gated':d.get('gated'),'cardData':d.get('cardData')}
    except Exception as e: report[name]={'error':str(e)}
print(json.dumps(report,indent=2)); (root/'output/new-dataset-research/probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
