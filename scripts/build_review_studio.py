"""Build a new UI over the locked V2 cases, without rewriting the old UI/protocol."""
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_review_handoff import check_lock

def main():
    lockpath=ROOT/'output/review-handoff/convention-pilot-protocol-lock.json'
    lock=json.loads(lockpath.read_text(encoding='utf-8'));check_lock(ROOT,lock)
    original=ROOT/'output/pin-geometry-review-v2';dest=ROOT/'output/review-studio';dest.mkdir(exist_ok=True)
    cases=json.loads((original/'cases.json').read_text(encoding='utf-8'))
    assert [c['id'] for c in cases]==lock['case_ids']
    protocol=json.loads((original/'protocol.json').read_text(encoding='utf-8'))
    for name in ('index.html','app.js','styles.css'):shutil.copyfile(ROOT/'scripts/review_studio'/name,dest/name)
    (dest/'data.json').write_text(json.dumps({'cases':cases,'protocol':protocol,'lock_sha256':hashlib.sha256(lockpath.read_bytes()).hexdigest()},ensure_ascii=False),encoding='utf-8')
    check_lock(ROOT,lock)
    print(json.dumps({'cases':len(cases),'old_protocol_unchanged':True,'url':'http://127.0.0.1:8767/review-studio/index.html?pass=A'}))

if __name__=='__main__':main()
