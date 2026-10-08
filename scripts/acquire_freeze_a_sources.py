"""Acquire manufacturer CAD/PDF evidence, never model-generated labels.

Each repository is an intake family, not proof of design independence. No
downloaded code is executed. Archives are bounded and extracted selectively.
"""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request
import urllib.parse
import zipfile
import argparse

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/raw/FreezeA-manufacturer'
REPOS = [
 'sparkfun/SparkFun_Qwiic_Directional_Pad',
 'sparkfun/SparkFun_Qwiic_Navigation_Switch',
 'sparkfun/SparkFun_Particulate_Matter_Sensor_Breakout_BMV080',
 'sparkfun/SparkFun_Qwiic_Current_Sensor_INA2XX',
 'sparkfun/SparkFun_Portability_Shield',
 'sparkfun/SparkFun_Serial_MP3_Player_Shield-MY1690X',
 'sparkfun/SparkFun_Qwiic_5V_Boost_AP3012K',
 'sparkfun/SparkFun_GNSS_Flex_Breakout',
 'sparkfun/SparkFun_GNSS_LG580P',
 'sparkfun/SparkFun_GNSS_DAN-F10N',
 'sparkfun/SparkFun_RFID_USB-C_Reader',
 'sparkfun/SparkFun_RTK_Postcard',
 'sparkfun/SparkFun_BlueSMiRF-v2',
 'sparkfun/SparkFun_Soft_Power_Switch_Mk2',
 'sparkfun/SparkFun_Soft_Power_Switch_USB-C',
 'sparkfun/SparkFun_Qwiic_WAV_Trigger_Pro',
 'sparkfun/SparkFun_Pico_Add_On_Flipper',
 'sparkfun/SparkFun_Qwiic_Fingerprint_Sensor_FPC2534',
 'sparkfun/SparkFun_Qwiic_OLED_1.3in',
 'sparkfun/SparkFun_Qwiic_OLED_1.5in',
 'sparkfun/SparkFun_GNSS_Flex_pHAT',
 'sparkfun/SparkFun_GNSS_LG290P',
 'OLIMEX/ESP32-DevKit-LiPo',
 'hatlabs/SH-ESP32-hardware',
 'OLIMEX/ESP32-S3-DevKit-Lipo',
]

def fetch(url, limit=70_000_000):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'GraphSchematic-evaluation-intake'}), timeout=50) as r:
        data=r.read(limit+1)
    if len(data)>limit: raise ValueError('download exceeds intake bound')
    return data

def save(base, name, data, url):
    path=(base / name).resolve()
    if not path.is_relative_to(base.resolve()): raise ValueError('unsafe archive member')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()!=data: raise ValueError('frozen source changed; use new intake version')
    path.write_bytes(data)
    return {'path':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'source_url':url}

def acquire(repo):
    base=BASE/repo.replace('/','--'); rows=[]; errors=[]
    for branch in ('main','master'):
        url=f'https://codeload.github.com/{repo}/zip/refs/heads/{branch}'
        try:
            archive=fetch(url)
            with zipfile.ZipFile(io.BytesIO(archive)) as z:
                selected=[]
                for item in z.infolist():
                    p=PurePosixPath(item.filename)
                    name=str(PurePosixPath(*p.parts[1:]))
                    low=name.lower()
                    if item.is_dir() or item.file_size>15_000_000: continue
                    wanted=(low.endswith(('.kicad_sch','.sch','.kicad_sym','.lib','.dcm','.kicad_pro')) or p.name.lower().startswith(('license','readme')) or (low.endswith('.pdf') and any(s in low for s in ('schem','board_files','hardware')) and 'dimension' not in low))
                    if wanted and '3dshapes/' not in low and len(str(base/name)) < 240:
                        selected.append(save(base,name,z.read(item),f'https://github.com/{repo}/blob/{branch}/{urllib.parse.quote(name)}'))
                rows.extend(selected)
                # Archive SHA identifies the received snapshot even if branch moves.
                result={'repository':repo,'source':f'https://github.com/{repo}','branch':branch,'archive_sha256':hashlib.sha256(archive).hexdigest(),'archive_bytes':len(archive),'files':rows,'errors':errors,'status':'acquired' if any(r['path'].endswith(('.kicad_sch','.sch')) for r in rows) else 'no_cad_found','design_independence_verified':False,'annotations_verified':False,'used_for_training':False,'used_for_model_selection':False}
                (base/'source-record.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
                return result
        except Exception as e: errors.append(f'{branch}: {type(e).__name__}: {e}')
    return {'repository':repo,'files':rows,'status':'download_failed','errors':errors,'design_independence_verified':False,'annotations_verified':False}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--retry-failed',action='store_true'); args=parser.parse_args()
    BASE.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'data/splits/freeze-a-source-intake.json'
    previous=json.loads(manifest.read_text(encoding='utf-8'))['sources'] if args.retry_failed and manifest.exists() else []
    results=[r for r in previous if r['status']=='acquired']
    repos=[r['repository'] for r in previous if r['status']!='acquired'] if previous else REPOS
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for item in pool.map(acquire,repos):
            results.append(item); print(item['repository'],item['status'],len(item['files']),flush=True)
    out={'status':'intake_only_not_ready_for_scoring','required_verified_designs':25,'repositories_attempted':len(REPOS),'acquired_repositories':sum(r['status']=='acquired' for r in results),'sources':results,'prohibited_use':'training, architecture selection and threshold tuning','missing':['design-level overlap adjudication','rendered page coordinate registration','component body bbox review','IC pin number/name/point review','annotation reviewer sign-off']}
    (ROOT/'data/splits/freeze-a-source-intake.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
