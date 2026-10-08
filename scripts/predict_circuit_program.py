"""Research inference for the fixed-grid R/C schematic grammar, not competition JSON."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from experiments.circuit_program_hierarchical import HierarchicalCircuitProgramNet
from scripts.prepare_circuit_program import execute, render_program


def load_model(checkpoint, device):
    model = HierarchicalCircuitProgramNet().to(device)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'output/training/circuit-program-v4/best.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert('L')
    if image.size != (128, 128):
        raise ValueError('This research grammar requires a 128x128 whole raster; arbitrary PCB images are unsupported')
    torch.set_num_threads(4)
    model = load_model(args.checkpoint, args.device)
    x = torch.from_numpy(1-np.asarray(image, dtype=np.float32)/255)[None, None].to(args.device)
    with torch.no_grad():
        program = model.generate(x, 6)[0].cpu().numpy()
    valid, _, components, groups = execute(program)
    Image.fromarray(render_program(program)).save(args.output/'reconstruction.png')
    result = {'scope': 'fixed-grid synthetic R/C grammar only; no OCR, IC, or flywire label support',
              'requires_review': True, 'independent_accuracy_verified': False,
              'input_sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'program': program.tolist(), 'horizontal_count': int(program[0]),
              'vertical_count': int(program[1]), 'components': components,
              'active_wire_node_ids': np.flatnonzero(valid).tolist(), 'net_wire_node_groups': groups,
              'reconstruction_style': 'canonical width1, not inferred source style'}
    (args.output/'prediction.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'requires_review': True}))


if __name__ == '__main__':
    main()
