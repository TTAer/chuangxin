"""Versioned training entry for the separable-dot bank, same fixed learning protocol."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_factorized_circuit as training


class ExperimentRoot:
    def __truediv__(self,relative):
        remap={'output/training/factorized-circuit-v1':'output/training/separable-circuit-v1',
               'data/processed/factorized-circuit-v1':'data/processed/separable-circuit-v1'}
        return ROOT/remap.get(relative,relative)


if __name__=='__main__':
    import json
    admission=json.loads((ROOT/'data/processed/separable-circuit-v1/admission.json').read_text())
    if admission['collisions']:raise ValueError('Training blocked by input ambiguity')
    training.ROOT=ExperimentRoot()
    training.main()
