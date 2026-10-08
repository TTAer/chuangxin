"""Retry bounded manufacturer assets without huge repository archives."""
import io
import json
import re
import urllib.parse
import zipfile
from acquire_freeze_a_sources import ROOT,BASE,fetch,save,acquire

def main():
    path=ROOT/'data/splits/freeze-a-source-intake.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    for item in manifest['sources']:
        if item['status']=='acquired': continue
        repo=item['repository']; base=BASE/repo.replace('/','--')
        if 'BMV080' in repo:
            replacement=acquire(repo); item.update(replacement); print(repo,item['status'],flush=True); continue
        root=f'https://raw.githubusercontent.com/{repo}/main/'
        rows=[]; errors=[]
        for name in ('README.md','LICENSE.md','docs/resources.md'):
            try: rows.append(save(base,name,fetch(root+name,1_000_000),root+name))
            except Exception as e: errors.append(f'{name}: {e}')
        links=set()
        for row in rows:
            text=(ROOT/row['path']).read_text(encoding='utf-8',errors='replace')
            for match in re.findall(r'\]\(([^)]+)\)',text):
                if match.lower().endswith(('.zip','.pdf')):
                    url=urllib.parse.urljoin(root+row['path'].split(repo.replace('/','--')+'/')[-1],match)
                    if 'board_files' in url or 'schem' in url.lower(): links.add(url)
        for url in sorted(links):
            try:
                data=fetch(url,30_000_000)
                name=urllib.parse.unquote(url.rsplit('/',1)[-1]); rows.append(save(base,'small-assets/'+name,data,url))
                if name.endswith('.zip'):
                    with zipfile.ZipFile(io.BytesIO(data)) as z:
                        for info in z.infolist():
                            if info.filename.lower().endswith(('.kicad_sch','.sch','.kicad_sym','.lib','.kicad_pro')) and info.file_size<10_000_000:
                                rows.append(save(base,'small-assets/extracted/'+info.filename,z.read(info),url+'#'+info.filename))
            except Exception as e: errors.append(f'{url}: {e}')
        item.update(files=rows,errors=errors,status='acquired' if any(r['path'].endswith(('.kicad_sch','.sch')) for r in rows) else 'download_failed',acquisition='bounded original resource assets')
        base.mkdir(parents=True,exist_ok=True); (base/'source-record.json').write_text(json.dumps(item,ensure_ascii=False,indent=2),encoding='utf-8')
        print(repo,item['status'],len(rows),flush=True)
    manifest['acquired_repositories']=sum(i['status']=='acquired' for i in manifest['sources'])
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()
