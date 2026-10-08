"""Bounded public intake with repository license and held-out family checks."""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/raw/OpenSchematics-pretrain-sample'
OUT.mkdir(parents=True,exist_ok=True)


def fetch(url,limit=128*1024*1024):
    req=urllib.request.Request(url,headers={'User-Agent':'GraphSchematic-Research/1.0'})
    with urllib.request.urlopen(req,timeout=40) as response:return response.read(limit)


def download(item):
    row=item['row'];repo=row.get('name','');index=item['row_idx'];im=row.get('schematic_image')
    record={'index':index,'repository':repo,'dataset':'bshada/open-schematics','dataset_card_license':'CC-BY-4.0','used_for_pretraining':False}
    if not im or any(s in repo.lower() for s in ('sparkfun','olimex','hatlabs','qwiic','directional','navigation_switch')):
        return dict(record,status='no_image_or_reserved_source_family')
    try:
        metadata=json.loads(fetch('https://api.github.com/repos/'+repo,2*1024*1024))
        license=(metadata.get('license') or {}).get('spdx_id')
        record.update(repository_license=license,repository_url=metadata.get('html_url'))
        if license not in ('MIT','Apache-2.0','BSD-2-Clause','BSD-3-Clause','CC0-1.0','CC-BY-4.0','CC-BY-SA-4.0','GPL-3.0','GPL-2.0','CERN-OHL-S-2.0','CERN-OHL-P-2.0','CERN-OHL-W-2.0'):
            return dict(record,status='source_license_not_resolved')
        body=fetch(im['src'],32*1024*1024);path=OUT/f'{index}.png';path.write_bytes(body)
        record.update(status='downloaded_pending_pixel_audit',image=str(path.relative_to(ROOT)),sha256=hashlib.sha256(body).hexdigest())
    except Exception as e:record.update(status='unavailable',error=type(e).__name__+': '+str(e)[:180])
    return record


def main():
    entries=[]
    for offset in (0,1000,2000,3000,4000,5000):
        url=f'https://datasets-server.huggingface.co/rows?dataset=bshada/open-schematics&config=default&split=train&offset={offset}&length=10'
        try:
            data=json.loads(fetch(url))
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:entries.extend(pool.map(download,data['rows']))
        except Exception as e:entries.append({'offset':offset,'status':'metadata_unavailable','error':str(e)[:180]})
        (OUT/'intake.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
        print(json.dumps({'offset':offset,'downloaded':sum('image' in e for e in entries),'checked':len(entries)}),flush=True)


if __name__=='__main__':main()
