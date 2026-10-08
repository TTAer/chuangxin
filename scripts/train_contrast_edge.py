"""Fixed paired contrast-only / edge model trials without editing frozen trainer."""
import sys,argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_factorized_circuit as training
from experiments.contrast_edge_circuit_reader import ContrastOnlyReader,ContrastEdgeReader


class ExperimentRoot:
    def __init__(self,model):self.model=model
    def __truediv__(self,relative):
        return ROOT/{'output/training/factorized-circuit-v1':f'output/training/contrast-edge-v1-{self.model}',
                     'data/processed/factorized-circuit-v1':'data/processed/contrast-edge-v1'}.get(relative,relative)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['contrast','edge'],required=True);args=parser.parse_args()
    assert (ROOT/'data/processed/contrast-edge-v1/manifest.json').exists()
    out=ROOT/f'output/training/contrast-edge-v1-{args.model}';out.mkdir(parents=True,exist_ok=True)
    if (out/'model-definition.json').exists():raise ValueError('Preserve model trial')
    (out/'model-definition.json').write_text(json.dumps({'model':args.model,'architecture_class':'ContrastOnlyReader' if args.model=='contrast' else 'ContrastEdgeReader',
                                                      'source_sha256':hashlib.sha256((ROOT/'experiments/contrast_edge_circuit_reader.py').read_bytes()).hexdigest(),
                                                      'fresh_random_initialization':True,'training_script_adapter':'same frozen trainer; generic trainer label superseded by this model definition'},indent=2),encoding='utf-8')
    # Model identity comes from the bank protocol and this immutable entry, not the frozen trainer's generic name.
    training.FactorizedCircuitReader=ContrastOnlyReader if args.model=='contrast' else ContrastEdgeReader
    training.ROOT=ExperimentRoot(args.model);training.main()
