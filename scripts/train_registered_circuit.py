"""One development-gated fixed training run on image-only registered pages."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_factorized_circuit as training


class ExperimentRoot:
    def __truediv__(self,relative):
        return ROOT/{'output/training/factorized-circuit-v1':'output/training/registered-circuit-v1',
                     'data/processed/factorized-circuit-v1':'data/processed/registered-circuit-v1'}.get(relative,relative)


if __name__=='__main__':
    decision=json.loads((ROOT/'output/registered-circuit-evaluation/development-decision.json').read_text())
    if not decision['new_training_required']:raise ValueError('Registered gate says do not repeat training')
    training.ROOT=ExperimentRoot();training.main()
