"""File inference for the visible-dot research model; output is always a proposal."""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
from experiments.factorized_circuit_reader import FactorizedCircuitReader
from scripts.prepare_separable_circuit import render_separable
from scripts.prepare_circuit_program import execute


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('image',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    im=Image.open(args.image).convert('L')
    if im.size!=(256,256):raise ValueError('Native256 fixed renderer grammar only')
    checkpoint=ROOT/'output/training/separable-circuit-v1/best.pt';torch.set_num_threads(4)
    model=FactorizedCircuitReader().eval();model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'))
    x=torch.from_numpy(1-np.asarray(im,dtype=np.float32)/255)[None,None]
    with torch.no_grad():program=model.generate(x)[0].numpy()
    _,_,components,groups=execute(program)
    args.output.mkdir(parents=True,exist_ok=True)
    report={'renderer':'visible-dot native256 fixed grid','input_sha256':hashlib.sha256(args.image.read_bytes()).hexdigest(),
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'program_proposal':program.tolist(),
            'component_proposals':components,'net_group_proposals':groups,'verified_net_groups':None,
            'requires_review':True,'topology_status':'unknown','scope':'R/C fixed-grid grammar, no IC/OCR/flywire support',
            'reason':'External file provenance and geometry assumptions have not been independently verified'}
    (args.output/'prediction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    Image.fromarray(render_separable(program,2)).save(args.output/'proposal.png')
    print(json.dumps({'output':str(args.output),'topology_status':'unknown'}))


if __name__=='__main__':main()
