"""Research proposal interface; shared linewidth is unverified even when geometry is ready."""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
from experiments.contrast_edge_circuit_reader import ContrastOnlyReader
from experiments.page_wire_residual import page_scores
from experiments.stroke_residual import decisions
from scripts.ordered_line_rectification_v2 import register_ordered


def main():
    p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if (a.output/'prediction.json').exists():raise ValueError('Preserve result')
    report=json.loads((ROOT/'output/page-wire-residual-holdout/report.json').read_text());thresholds=report['protocol']['thresholds'];raw=np.asarray(Image.open(a.image).convert('L'));aligned,g=register_ordered(raw);a.output.mkdir(parents=True,exist_ok=True)
    result={'geometry':g,'program_proposal':None,'topology_status':'unknown','requires_review':True,'verified_net_groups':None,
            'automatic_release_enabled':False,'controlled_gate_passed':report['gate_passed'],'shared_linewidth_verified':False,
            'scope':'finite full-length orthogonal wires with shared profiles; mixed linewidth counterexample exists'}
    if g['ready']:
        torch.set_num_threads(4);model=ContrastOnlyReader().eval();model.load_state_dict(torch.load(ROOT/'output/training/contrast-edge-v1-contrast/best.pt',weights_only=True,map_location='cpu'))
        with torch.no_grad():program=model.generate(torch.from_numpy(1-aligned.astype(np.float32)/255)[None,None])[0].numpy()
        scores,slots=page_scores(raw,aligned,g);labels=decisions(scores,**thresholds)
        for s,d in zip(slots,labels):
            if d!=7 and program[s]!=7:program[s]=d
        result['program_proposal']=program.tolist()
    else:result['reason']='unsupported geometry; abstain before model loading'
    (a.output/'prediction.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'ready':g['ready'],'proposal':result['program_proposal'] is not None,'topology_status':'unknown','shared_linewidth_verified':False}))


if __name__=='__main__':main()
