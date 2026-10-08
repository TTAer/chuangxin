"""Check archive hashes and execute packaged competition entry on two cases."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.competition import export_competition
from pcbparse.dataset import index_dataset

def main():
    archive=ROOT/'submission/graphschematic-association-development.zip'; workspace=ROOT/'tmp/association-release-smoke'; workspace.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('BUILD_MANIFEST.json'))
        for name,r in manifest['files'].items(): assert hashlib.sha256(z.read(name)).hexdigest()==r['sha256'],name
        for name in z.namelist():
            assert (workspace/name).resolve().is_relative_to(workspace.resolve()),name
        z.extractall(workspace)
    reports=[]
    for mode in ('legacy','guarded'):
        inputs=ROOT/'tmp'/('association-release-input-'+mode); inputs.mkdir(parents=True,exist_ok=True)
        if mode=='legacy':
            case='guide-schematic'; shutil.copyfile(ROOT/'data/examples/guide-schematic.png',inputs/(case+'.png'))
            expected=load(ROOT/'output/recognition-workspace-smoke/final_results/guide-schematic_Pred.json')
        else:
            case='0196'; sample=next(s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300') if s['id']==case)
            shutil.copyfile(sample['image'],inputs/(case+'.png'))
            base=ROOT/'output/designator-guarded/pcb55'/case
            expected=export_competition(load(base/'prediction.json'),load(base/'observations.json')['component_categories'])
        assert len(list(inputs.iterdir()))==1
        out=ROOT/'output'/('association-packaged-'+mode)
        cmd=[sys.executable,'agent_system','-input',str(inputs),'-output',str(out),'--designator-mode',mode]
        if mode=='guarded': cmd+=['--ic-identity-mode','retrieval','--side-pin-model','models/candidates/side-strip-local/side-strip.onnx','--side-name-anchors']
        run=subprocess.run(cmd,cwd=workspace,capture_output=True,encoding='utf-8',errors='replace',timeout=150)
        assert run.returncode==0,(run.stdout,run.stderr)
        assert load(out/'final_results'/(case+'_Pred.json'))==expected,case
        reports.append({'mode':mode,'case':case,'exact_match':True})
    save(ROOT/'output/association-release-verification.json',{'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'verified_files':len(manifest['files']),'reports':reports,'official_evaluator_verified':False})
    print({'verified_files':len(manifest['files']),'reports':reports},flush=True)
if __name__=='__main__': main()
