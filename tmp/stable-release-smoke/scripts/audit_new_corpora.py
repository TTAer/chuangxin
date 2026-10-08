"""Count acquired originals, exact duplicates and repeated CGHD designs."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

base=ROOT/'data/raw/CGHD-v16-subset'; rows=json.loads((base/'records.json').read_text())
def design(row): return re.split(r'_D',row['sample'])[0]
train=[r for r in rows if int(r['drafter'].split('_')[-1])%5!=0]
val=[r for r in rows if int(r['drafter'].split('_')[-1])%5==0]
ams=ROOT/'data/raw/AMSNet-1.0/amsnet_1.0'
originals=[p for p in ams.rglob('*.jpg') if p.stem==p.parent.name]
old={digest(p) for source in ('PCBnet','AITEE') for p in (ROOT/'data/raw'/source).rglob('*') if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg')}
new=[base/r['image'] for r in rows]+originals
result={'cghd':{'acquired_paired_samples':len(rows),'unique_image_file_hashes':len({digest(base/r['image']) for r in rows}),'drafters':len({r['drafter'] for r in rows}),'design_prefix_counts':dict(Counter(design(r) for r in rows)),'train_validation_drafters_overlap':sorted({r['drafter'] for r in train}&{r['drafter'] for r in val}),'train_validation_design_prefix_overlap':sorted({design(r) for r in train}&{design(r) for r in val}),'split_claim':'held-out drafters; repeated circuit designs may cross splits; not a design-independent PCB test','license_metadata':'CC BY 4.0','license_archive_readme':'CC BY-SA 3.0','raw_redistribution':False},'amsnet':{'original_cases':len(originals),'unique_original_file_hashes':len({digest(p) for p in originals}),'image_files':len(list(ams.rglob('*.jpg'))),'cases_with_bbox_json':sum(p.with_name(p.stem+'_bbox.json').exists() for p in originals),'cases_with_spice':sum(p.with_suffix('.cir').exists() for p in originals),'training_status':'inspection only; third-party textbook figure provenance and pin mapping unresolved'},'new_to_existing_exact_file_hash_overlap':sum(digest(p) in old for p in new),'overlap_method':'SHA256 file bytes only; does not establish semantic or design independence','not_acquired':['Netlistify gated corpus','CircuitPile-9k unavailable repository','AMSNet 2.0','CircuitVQA image corpus']}
out=ROOT/'output/new-dataset-research/local-audit.json'; out.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)
