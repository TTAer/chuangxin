"""Whole256 file inference for isolated radial crossing research, always unverified."""
import sys,json,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
from scripts.ordered_line_rectification_v2 import register_ordered
from experiments.radial_evidence_net import RadialEvidenceNet,crossing_patches
from experiments.contrast_edge_circuit_reader import ContrastOnlyReader
from scripts.prepare_separable_circuit import render_separable


def main():
    p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('--mode',choices=['raw','canonical'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if (a.output/'prediction.json').exists():raise ValueError('Preserve prediction')
    raw=np.asarray(Image.open(a.image).convert('L'));aligned,d=register_ordered(raw);a.output.mkdir(parents=True,exist_ok=True)
    result={'geometry':d,'mode':a.mode,'program_proposal':None,'verified_net_groups':None,'topology_status':'unknown','requires_review':True,
            'scope':'finite orthogonal R/C line grammar, not independently validated or official scoring',
            'input_sha256':hashlib.sha256(a.image.read_bytes()).hexdigest()}
    Image.fromarray(aligned).save(a.output/'aligned.png')
    if d['ready']:
        summary=json.loads((ROOT/f'output/training/radial-evidence-v1-{a.mode}/summary.json').read_text());assert summary['full_training_executed']
        torch.set_num_threads(4);base=ContrastOnlyReader().eval();reader=RadialEvidenceNet().eval()
        paths=[ROOT/'output/training/contrast-edge-v1-contrast/best.pt',ROOT/f'output/training/radial-evidence-v1-{a.mode}/best.pt']
        base.load_state_dict(torch.load(paths[0],weights_only=True,map_location='cpu'));reader.load_state_dict(torch.load(paths[1],weights_only=True,map_location='cpu'))
        patches,slots=crossing_patches(raw,aligned,d,a.mode)
        with torch.no_grad():
            program=base.generate(torch.from_numpy(1-aligned.astype(np.float32)/255)[None,None])[0].numpy();classes=reader(torch.from_numpy(patches)).argmax(-1).numpy()
        for s,v in zip(slots,classes):
            if program[s]!=7:program[s]=int(v)+8
        result['program_proposal']=program.tolist();result['checkpoint_sha256']=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        Image.fromarray(render_separable(program,2)).save(a.output/'proposal.png')
    else:result['reason']='unsupported geometry; abstain before model inference'
    (a.output/'prediction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'geometry_ready':d['ready'],'proposal':result['program_proposal'] is not None,'topology_status':'unknown'}))


if __name__=='__main__':main()
