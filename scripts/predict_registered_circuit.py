"""Image-only registration followed by conservative research inference."""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
from experiments.factorized_circuit_reader import FactorizedCircuitReader
from scripts.circuit_registration import register_image
from scripts.prepare_separable_circuit import render_separable


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('image',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    im=np.asarray(Image.open(a.image).convert('L'))
    if im.shape!=(256,256):raise ValueError('Native256 uniformly spaced orthogonal renderer only')
    aligned,registration=register_image(im);a.output.mkdir(parents=True,exist_ok=True)
    result={'registration':registration,'requires_review':True,'topology_status':'unknown','verified_net_groups':None,
            'input_sha256':hashlib.sha256(a.image.read_bytes()).hexdigest(),'program_proposal':None,
            'scope':'global scale/translation only, no arbitrary layout or OCR'}
    Image.fromarray(aligned).save(a.output/'registered.png')
    if registration['ready']:
        summary=json.loads((ROOT/'output/training/registered-circuit-v1/summary.json').read_text())
        checkpoint=ROOT/('output/training/registered-circuit-v1/best.pt' if summary['full_training_executed'] else 'output/training/separable-circuit-v1/best.pt')
        model=FactorizedCircuitReader().eval();model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'));torch.set_num_threads(4)
        with torch.no_grad():program=model.generate(torch.from_numpy(1-aligned.astype(np.float32)/255)[None,None])[0].numpy()
        result['program_proposal']=program.tolist();result['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        Image.fromarray(render_separable(program,2)).save(a.output/'proposal.png')
    else:result['reason']='registration failed; abstain from program generation'
    (a.output/'prediction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'registration_ready':registration['ready'],'topology_status':'unknown'}))


if __name__=='__main__':main()
