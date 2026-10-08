"""CPU file interface for dev-selected local model; never claim verified electrical output."""
import sys,json,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
from experiments.radial_evidence_net import RadialEvidenceNet,crossing_patches
from experiments.spatial_evidence_net import SpatialEvidenceNet
from experiments.contrast_edge_circuit_reader import ContrastOnlyReader
from scripts.ordered_line_rectification_v2 import register_ordered
from scripts.evaluate_evidence_factorial import trial_folder


def main():
    p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if (a.output/'prediction.json').exists():raise ValueError('Preserve output')
    report=json.loads((ROOT/'output/evidence-factorial-holdout/report.json').read_text());trial=report['protocol']['selected_trial_dev_only']
    raw=np.asarray(Image.open(a.image).convert('L'));aligned,d=register_ordered(raw);a.output.mkdir(parents=True,exist_ok=True)
    result={'geometry':d,'trial':trial,'program_proposal':None,'topology_status':'unknown','requires_review':True,'verified_net_groups':None,
            'automatic_release_enabled':False,'controlled_gate_passed':report['selected_gate_passed'],'scope':'finite orthogonal grammar research, no independent real accuracy',
            'input_sha256':hashlib.sha256(a.image.read_bytes()).hexdigest()}
    if d['ready']:
        torch.set_num_threads(4);base=ContrastOnlyReader().eval();local=(RadialEvidenceNet() if trial.startswith('radial') else SpatialEvidenceNet()).eval()
        base.load_state_dict(torch.load(ROOT/'output/training/contrast-edge-v1-contrast/best.pt',weights_only=True,map_location='cpu'))
        local.load_state_dict(torch.load(trial_folder(trial)/'best.pt',weights_only=True,map_location='cpu'))
        patches,slots=crossing_patches(raw,aligned,d,'raw')
        with torch.no_grad():
            program=base.generate(torch.from_numpy(1-aligned.astype(np.float32)/255)[None,None])[0].numpy();classes=local(torch.from_numpy(patches)).argmax(-1).numpy()
        for s,v in zip(slots,classes):
            if program[s]!=7:program[s]=int(v)+8
        result['program_proposal']=program.tolist()
    else:result['reason']='unsupported geometry; abstain before loading neural models'
    Image.fromarray(aligned).save(a.output/'aligned.png');(a.output/'prediction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'trial':trial,'ready':d['ready'],'proposal':result['program_proposal'] is not None,'topology_status':'unknown'}))


if __name__=='__main__':main()
