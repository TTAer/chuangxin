"""Fixed development-only oracle-ROI OCR diagnosis; never imported by RC0."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import re
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
from pcbparse.dataset import index_dataset
from pcbparse.vision import ImageParser, read_image, iou
from scripts.prepare_pin_token_v1 import referenced_labels

VARIANTS = ('native', '2x', '3x', 'contrast_2x', 'binary_2x', 'cw_2x', 'ccw_2x')


def local_exact(tokens, text, box):
    return any(t['text'] == text and iou(t['bbox'], box) >= .25 for t in tokens)


def restore_point(x, y, variant, width, height, scale, origin):
    x, y = x / scale, y / scale
    if variant == 'cw_2x':
        x, y = y, height - 1 - x
    elif variant == 'ccw_2x':
        x, y = width - 1 - y, x
    return [x + origin[0], y + origin[1]]


def add_label_quality(final):
    for row in final['rows']:
        row['number_grammar_valid'] = bool(re.fullmatch(r'[A-Z]?[0-9]+', row['text'])) if row['role'] == 'pin_number' else None
        row['label_risks'] = []
        if row['role'] == 'pin_number' and not row['number_grammar_valid']:
            row['label_risks'].append('non_number_like_explicit_weak_reference')
        if row['role'] == 'pin_number' and re.fullmatch(r'[RCLDUQ][0-9]+', row['text']):
            row['label_risks'].append('ball_number_or_designator_ambiguity_requires_review')
    subset = [r for r in final['rows'] if r['role']=='pin_number' and r['number_grammar_valid']]
    final['number_grammar_valid_subset'] = dict(samples=len(subset), any_variant_recovered=sum(r['any_variant_exact'] for r in subset),
        variants={v:sum(r['variants'][v]['exact'] for r in subset) for v in VARIANTS},
        note='grammar validity is not proof of true pin-number role; no samples relabeled or discarded')
    final['label_risk_counts'] = dict(Counter(risk for r in final['rows'] for risk in r['label_risks']))
    for role, group_summary in final['summary'].items():
        group = [r for r in final['rows'] if r['role'] == role]
        group_summary['literal_anywhere_in_oracle_crop_union'] = sum(any(any(t['text']==r['text'] for t in z['tokens']) for z in r['variants'].values()) for r in group)
        for variant, stats in group_summary['variants'].items():
            stats['literal_anywhere_in_oracle_crop'] = sum(any(t['text']==r['text'] for t in r['variants'][variant]['tokens']) for r in group)
            stats['exact_string_but_bad_location'] = sum(r['variants'][variant]['exact_elsewhere_only'] for r in group)
    return final


def main():
    global VARIANTS
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['legacy','bounded'], default='legacy')
    args = ap.parse_args()
    if args.mode == 'bounded':
        VARIANTS = ('2x',)
    data = ROOT / 'data/processed/pin-token-v1'
    preparation = json.loads((data / 'preparation.json').read_text(encoding='utf-8'))
    lookup = {r['id']: r for r in index_dataset(ROOT / 'data/raw/PCBnet/Dataset300')}
    buckets = defaultdict(lambda: defaultdict(list))
    counts = Counter()
    for page in preparation['selection']:
        item = lookup[page['page']]
        raw = json.loads(Path(item['annotation']).read_text(encoding='utf-8'))
        roles, owners = referenced_labels(raw)
        tokens = json.loads((data / (page['page'] + '-ocr.json')).read_text(encoding='utf-8'))
        for tid in sorted(raw['texts']):
            if len(roles[tid]) != 1 or len(owners[tid]) != 1:
                continue
            role = next(iter(roles[tid]))
            if role not in ('pin_number', 'pin_name'):
                continue
            cid, pid = next(iter(owners[tid]))
            if raw['components'][cid]['category'] != 'Box':
                continue
            text = raw['texts'][tid]['content']
            if not text:
                continue
            b = raw['texts'][tid]['location']
            box = [b[k] for k in ('x1', 'y1', 'x2', 'y2')]
            counts[(page['split'], role, 'eligible')] += 1
            if local_exact(tokens, text, box):
                counts[(page['split'], role, 'full_exact')] += 1
                continue
            buckets[(page['split'], role)][page['page']].append(dict(
                page=page['page'], split=page['split'], role=role, component=cid,
                pin=pid, text_id=tid, text=text, bbox=box, image=item['image'],
                annotation=item['annotation'], image_sha256=item['sha256']))
    selected = []
    # Fixed 12 per train/validation and number/name stratum, round-robin pages.
    for key in [('train', 'pin_number'), ('train', 'pin_name'),
                ('validation', 'pin_number'), ('validation', 'pin_name')]:
        pool = buckets[key]
        chosen = 0
        for offset in range(12):
            for page in sorted(pool):
                if offset < len(pool[page]) and chosen < 12:
                    selected.append(pool[page][offset])
                    chosen += 1
    protocol = dict(selection='12 per split/role; sorted pages and text IDs, round-robin; no outcome-based selection',
                    variants=list(VARIANTS), fixed_crop_padding_pixels=8,
                    exact_match='literal string AND bbox IoU >= 0.25; no whitelist/corrected characters',
                    samples=selected)
    out = ROOT / ('output/ablation/local-pin-ocr-bounded.json' if args.mode=='bounded' else 'output/ablation/local-pin-ocr.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    protocol_path = out.with_name('local-pin-ocr-bounded-protocol.json' if args.mode=='bounded' else 'local-pin-ocr-protocol.json')
    protocol['mode'] = args.mode
    protocol['detector_config'] = {'limit_type':'max', 'limit_side_len':736, 'threads':2} if args.mode=='bounded' else {'limit_type':'min','limit_side_len':1600,'threads':'ImageParser default'}
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    parser = ImageParser()
    if args.mode == 'bounded':
        from rapidocr_onnxruntime import RapidOCR
        parser.ocr = RapidOCR(det_limit_type='max', det_limit_side_len=736, intra_op_num_threads=2, inter_op_num_threads=2)
    rows = []
    for row in selected:
        image = read_image(row['image'])
        assert hashlib.sha256(Path(row['image']).read_bytes()).hexdigest() == row['image_sha256']
        x1, y1, x2, y2 = map(round, row['bbox'])
        x1, y1 = max(0, x1-8), max(0, y1-8)
        x2, y2 = min(image.shape[1], x2+8), min(image.shape[0], y2+8)
        patch = image[y1:y2, x1:x2]
        h, w = patch.shape[:2]
        evaluations = {}
        for variant in VARIANTS:
            scale = 1 if variant == 'native' else 3 if variant == '3x' else 2
            crop = patch.copy()
            if variant == 'contrast_2x':
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
            if variant == 'binary_2x':
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            if variant == 'cw_2x':
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            elif variant == 'ccw_2x':
                crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if scale != 1:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            started = time.perf_counter()
            result, _ = parser.ocr(crop)
            elapsed = time.perf_counter() - started
            tokens = []
            for quad, text, confidence in result or []:
                points = [restore_point(p[0], p[1], variant, w, h, scale, (x1,y1)) for p in quad]
                box = [min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)]
                tokens.append(dict(text=text, confidence=float(confidence), bbox=box, target_iou=iou(box, row['bbox'])))
            localized = [t for t in tokens if t['target_iou'] >= .25]
            picked = max(localized, key=lambda t: (t['target_iou'], t['confidence']), default=None)
            evaluations[variant] = dict(seconds=elapsed, tokens=tokens,
                exact=local_exact(tokens, row['text'], row['bbox']),
                spatial_selected_text=picked['text'] if picked else None,
                wrong_spatial_selected=bool(picked and picked['text'] != row['text']),
                exact_elsewhere_only=any(t['text']==row['text'] for t in tokens) and not local_exact(tokens,row['text'],row['bbox']))
        rows.append(dict(row, variants=evaluations, any_variant_exact=any(v['exact'] for v in evaluations.values())))
        out.write_text(json.dumps(dict(status='running', protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(), rows=rows), ensure_ascii=False, indent=2),encoding='utf-8')
        print(json.dumps(dict(completed=len(rows), page=row['page'], role=row['role'], exact=rows[-1]['any_variant_exact'])), flush=True)
    summary = {}
    for role in ('pin_number','pin_name'):
        group = [r for r in rows if r['role']==role]
        summary[role] = dict(samples=len(group), any_variant_recovered=sum(r['any_variant_exact'] for r in group),
            variants={v:dict(recovered=sum(r['variants'][v]['exact'] for r in group), wrong_spatial_selected=sum(r['variants'][v]['wrong_spatial_selected'] for r in group)) for v in VARIANTS})
    final = dict(status='complete', mode=args.mode, detector_limit_type=parser.ocr.text_det.limit_type, detector_limit_side_len=parser.ocr.text_det.limit_side_len, rc0_unchanged=True, protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        selected_samples=len(rows), source_counts={'/'.join(k):v for k,v in counts.items()}, summary=summary,
        any_variant_union_is_oracle_upper_bound=True,
        limitations=['GT used for crop ROI and localized scoring: not an automatic recovery system',
            'explicit CVJson pin OCR references only; weak labels may be noisy',
            'full-page OCR misses may mean wrong text, bad localization, or fragmented token',
            'no pin binding accuracy evaluated; wrong strings and neighbor tokens can misbind',
            'validation is development, not independent Freeze-A; Freeze-A/B untouched',
            'only real RapidOCR top1 results; no synthetic top-k or character confidence'], rows=rows)
    out.write_text(json.dumps(add_label_quality(final), ensure_ascii=False, indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
