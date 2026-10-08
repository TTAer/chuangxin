"""Fixed-checkpoint controls; no optimization or stress-based model selection."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image, ImageDraw
from experiments.circuit_program import MASK
from scripts.predict_circuit_program import load_model
from scripts.prepare_circuit_program import execute, render_program
from scripts.train_circuit_program import metrics


def main():
    torch.set_num_threads(4)
    out = ROOT/'output/circuit-program-audit'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists():
        raise ValueError('Preserve completed audit')
    checkpoint = ROOT/'output/training/circuit-program-v4/best.pt'
    protocol = {'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'selection': 'No optimization; already fixed validation-selected checkpoint',
                'controls': ['grammar-constrained steps1 vs steps6', 'blank image', 'reversed image/gold pairing',
                             'same stress programs clean render widths1/2/3 diagnostic only'],
                'preview': 'first8 validation and first8 stress, no cherry selection',
                'style_diagnostics_are_independent_validation': False}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    model = load_model(checkpoint, 'cuda')
    report = {'protocol': protocol, 'cohorts': {}}
    def infer(images, steps=6):
        result = []
        with torch.no_grad():
            for i in range(0, len(images), 16):
                x = torch.tensor(1-images[i:i+16].astype(np.float32)/255, device='cuda')
                with torch.autocast('cuda', dtype=torch.float16):
                    result.append(model.generate(x, steps).cpu().numpy())
        return np.concatenate(result)
    for name in ('validation', 'stress'):
        with np.load(ROOT/f'data/processed/circuit-program-v2/{name}.npz') as data:
            images, gold = data['images'], data['program']
        one, six = infer(images, 1), infer(images, 6)
        controls = {'steps1': metrics(one, gold), 'steps6': metrics(six, gold),
                    'blank_image': metrics(infer(np.full_like(images, 255)), gold),
                    'reversed_image_pairing': metrics(infer(images[::-1].copy()), gold)}
        if name == 'stress':
            controls['clean_rerender_diagnostic'] = {
                str(w): metrics(infer(np.stack([render_program(g, w)[None] for g in gold])), gold)
                for w in (1, 2, 3)}
        report['cohorts'][name] = controls
        canvas = Image.new('RGB', (384, 8*156), 'white')
        draw = ImageDraw.Draw(canvas)
        for i in range(8):
            for col, im in enumerate((images[i, 0], render_program(gold[i]), render_program(six[i]))):
                canvas.paste(Image.fromarray(im).convert('RGB'), (col*128, i*156+24))
            draw.text((2, i*156+3), f'{name} {i}: input | gold width1 | pred width1', fill='black')
        canvas.save(out/f'{name}-first8.png')
        if name == 'stress':
            Image.fromarray(images[0, 0]).save(out/'stress-first.png')
    # Count prefix must depend on image only, irrespective of program context.
    x = torch.tensor(1-images[:4].astype(np.float32)/255, device='cuda')
    with torch.no_grad():
        a = model(x, torch.full((4, 68), MASK, device='cuda', dtype=torch.long))[:, :2]
        b = model(x, torch.zeros((4, 68), device='cuda', dtype=torch.long))[:, :2]
        cpu_model = load_model(checkpoint, 'cpu')
        cpu_pred = cpu_model.generate(x.cpu(), 6).numpy()
        gpu_pred = model.generate(x, 6).cpu().numpy()
    report['count_context_independent'] = bool(torch.equal(a, b))
    report['cpu_cuda_float32_first4_program_equal'] = bool(np.array_equal(cpu_pred, gpu_pred))
    # Interpreter fixture: an isolated resistor cannot short its two terminals.
    fixture = np.full(68, 7, np.int64); fixture[:2] = 3
    for i in range(3):
        for j in range(3): fixture[2+i*6+j] = 8
    for i in range(2):
        for j in range(3): fixture[38+i*6+j] = 10
    fixture[38] = 11
    _, relation, components, _ = execute(fixture)
    assert relation[6, 12] == 0 and len(components) == 1
    report['resistor_does_not_short_terminals'] = True
    frozen = {}
    for path in ('data/splits/rc0-evaluation.json', 'output/port-lift-research-manifest.json',
                 'output/port-lift-uncertainty-manifest.json', 'output/probe-flow-research-manifest.json',
                 'output/netset-oracle-research-manifest.json', 'output/topology-agreement-manifest.json'):
        manifest = json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
        hashes = manifest.get('runtime_sha256', manifest.get('sha256', {}))
        if isinstance(hashes, dict):
            mismatches = [p for p, h in hashes.items() if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != h]
            frozen[path] = {'checked': len(hashes), 'mismatches': mismatches}
    report['frozen_checks'] = frozen
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
