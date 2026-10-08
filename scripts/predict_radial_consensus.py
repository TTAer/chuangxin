"""Research-only two-view correction; release gate failed, never emit verified nets."""
import sys,json,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
from scripts.ordered_line_rectification_v2 import register_ordered
from scripts.evaluate_radial_consensus import consensus_patch_decisions
from experiments.radial_evidence_net import RadialEvidenceNet
from experiments.contrast_edge_circuit_reader import ContrastOnlyReader


def main():
    p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if (a.output/'prediction.json').exists():raise ValueError('Preserve result')
    raw=np.asarray(Image.open(a.image).convert('L'));aligned,d=register_ordered(raw);a.output.mkdir(parents=True,exist_ok=True)
    result={'geometry':d,'program_proposal':None,'verified_net_groups':None,'topology_status':'unknown','requires_review':True,
            'automatic_release_enabled':False,'scope':'failed-gate research proposal only','input_sha256':hashlib.sha256(a.image.read_bytes()).hexdigest()}
    if d['ready']:
        torch.set_num_threads(4);base=ContrastOnlyReader().eval();base.load_state_dict(torch.load(ROOT/'output/training/contrast-edge-v1-contrast/best.pt',weights_only=True,map_location='cpu'));models={}
        for mode in ('raw','canonical'):
            models[mode]=RadialEvidenceNet().eval();models[mode].load_state_dict(torch.load(ROOT/f'output/training/radial-evidence-v1-{mode}/best.pt',weights_only=True,map_location='cpu'))
        with torch.no_grad():program=base.generate(torch.from_numpy(1-aligned.astype(np.float32)/255)[None,None])[0].numpy()
        slots,labels,accepted=consensus_patch_decisions(raw,aligned,d,models,'cpu')
        for s,v,ok in zip(slots,labels,accepted):
            if ok and program[s]!=7:program[s]=int(v)+8
        result['program_proposal']=program.tolist()
    else:result['reason']='unsupported geometry, abstain before model inference'
    (a.output/'prediction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'geometry_ready':d['ready'],'proposal':result['program_proposal'] is not None,'topology_status':'unknown','automatic_release_enabled':False}))


if __name__=='__main__':main()
