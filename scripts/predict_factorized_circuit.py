"""Research visual proposal with conservative unknown topology status."""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
from experiments.factorized_circuit_reader import FactorizedCircuitReader
from scripts.prepare_factorized_circuit import render_native


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('image',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    im=Image.open(a.image).convert('L')
    if im.size!=(256,256):raise ValueError('Native256 fixed grammar only, arbitrary schematic layouts unsupported')
    torch.set_num_threads(4);checkpoint=ROOT/'output/training/factorized-circuit-v1/best.pt'
    model=FactorizedCircuitReader().eval();model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'))
    array=np.asarray(im);x=torch.from_numpy(1-array.astype(np.float32)/255)[None,None]
    with torch.no_grad():program=model.generate(x)[0].numpy()
    binary=array<float(np.median(array))*.5;stroke_thickness=int(binary[24:49,24].sum())
    a.output.mkdir(parents=True,exist_ok=True)
    # No style/provenance proof at this generic file boundary: proposals remain unknown.
    result={'scope':'native256 renderer-grid proposal only','requires_review':True,'topology_verified':False,
            'topology_status':'unknown','program_proposal':program.tolist(),
            'isolated_left_line_pixel_thickness':stroke_thickness,
            'known_renderer_width7_collision_warning':stroke_thickness>=9,
            'warning_scope':'heuristic assumes this renderer; not an external-image reliability classifier',
            'verified_net_groups':None,'input_sha256':hashlib.sha256(a.image.read_bytes()).hexdigest(),
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    Image.fromarray(render_native(program,2)).save(a.output/'proposal.png')
    (a.output/'prediction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(a.output),'topology_status':'unknown'}))


if __name__=='__main__':main()
