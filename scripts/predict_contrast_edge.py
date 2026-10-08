"""Guarded file inference for either fixed paired model; always a research proposal."""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
from scripts.circuit_geometry_guard import register_guarded
from scripts.prepare_separable_circuit import render_separable
from experiments.contrast_edge_circuit_reader import ContrastOnlyReader,ContrastEdgeReader


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('image',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',choices=['contrast','edge'],default='edge');a=p.parse_args()
    image=np.asarray(Image.open(a.image).convert('L'))
    if image.shape!=(256,256):raise ValueError('Native256 uniform orthogonal grammar only')
    aligned,geometry=register_guarded(image);a.output.mkdir(parents=True,exist_ok=True)
    result={'geometry':geometry,'model':a.model,'program_proposal':None,'topology_status':'unknown','requires_review':True,
            'verified_net_groups':None,'scope':'finite R/C grid, not general schematic parser',
            'input_sha256':hashlib.sha256(a.image.read_bytes()).hexdigest()}
    Image.fromarray(aligned).save(a.output/'aligned.png')
    summary=json.loads((ROOT/f'output/training/contrast-edge-v1-{a.model}/summary.json').read_text())
    if geometry['ready'] and summary['full_training_executed']:
        checkpoint=ROOT/f'output/training/contrast-edge-v1-{a.model}/best.pt';torch.set_num_threads(4)
        model=(ContrastOnlyReader() if a.model=='contrast' else ContrastEdgeReader()).eval();model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'))
        with torch.no_grad():program=model.generate(torch.from_numpy(1-aligned.astype(np.float32)/255)[None,None])[0].numpy()
        result['program_proposal']=program.tolist();result['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        Image.fromarray(render_separable(program,2)).save(a.output/'proposal.png')
    else:result['reason']='geometry not supported or model tiny gate failed; abstain'
    (a.output/'prediction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'geometry_ready':geometry['ready'],'generated_proposal':result['program_proposal'] is not None,'topology_status':'unknown'}))


if __name__=='__main__':main()
