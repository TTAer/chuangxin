"""Category-constrained global association of OCR designators to symbols."""
import re
from .core import assignment
from .dataset import DESIGNATOR

# Categories whose output identity is a net label must not consume designators.
PREFIXES = {'R': ('R',), 'C': ('C',), 'L': ('L',),
            'Box': ('U', 'IC', 'J', 'P', 'CN'),
            'Block': ('U', 'IC', 'J', 'P', 'CN'),
            'D': ('D',), 'LED': ('D', 'LED', 'PWR'),
            'Crystal': ('X', 'Y'), 'Switch': ('SW', 'K'),
            'MOSFET': ('Q',), 'BJT': ('Q',), 'Fuse': ('F',),
            'PIN': ('J', 'P', 'CN'), 'Jumper': ('J', 'JP'),
            'AMP': ('U', 'IC'), 'Transformer': ('T',), 'Buzzer': ('BZ',)}


def associate(detections, texts, mode='legacy'):
    from .vision import box_distance, compatible
    if mode not in ('legacy', 'typed', 'boxed', 'guarded'):
        raise ValueError('unsupported designator association mode')
    designators = [t for t in texts if DESIGNATOR.fullmatch(t['text']) or
                   (mode in ('boxed','guarded') and re.fullmatch(r'USB\d+[A-Za-z]?',t['text'],re.I))]
    scores = []
    for d in detections:
        row = []
        for t in designators:
            prefix = re.match(r'[A-Za-z]+', t['text']).group().upper()
            prefixes=PREFIXES.get(d['category'], ())
            if mode in ('boxed','guarded') and d['category'] in ('Box','Block'):
                prefixes=prefixes+('USB','LED')
            allowed = compatible(t['text'], d['category']) if mode == 'legacy' else prefix in prefixes
            # Preserve unfamiliar detector categories rather than deleting valid
            # motor/antenna/measurement symbols through an incomplete type list.
            if mode=='guarded' and d['category'] not in PREFIXES:
                allowed=d['category'] not in ('Gnd','V','NET_input','Net_output','Net_bidirection','Testpoint') and compatible(t['text'],d['category'])
            row.append(max(0, 100-box_distance(t['bbox'], d['bbox'])) if allowed else 0)
        scores.append(row)
    assigned = {i: j for i, j in assignment(scores) if scores[i][j] > 25}
    records = [{'detection': i, 'category': d['category'], 'bbox': d['bbox'],
                'designator': designators[assigned[i]]['text'] if i in assigned else None,
                'status': 'assigned' if i in assigned else 'unassigned',
                'score': scores[i][assigned[i]] if i in assigned else None}
               for i, d in enumerate(detections)]
    return designators, assigned, {'mode': mode, 'records': records}
