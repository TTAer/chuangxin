"""Register a manufacturer PDF against original CAD, without model inference.

This is a partial source annotation and remains ineligible for blind scoring.
It intentionally records CAD property/project reference discrepancies.
"""
import hashlib
import json
from pathlib import Path
import pymupdf
from audit_freeze_a_sources import sexp, children, first

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/FreezeA-manufacturer/sparkfun--SparkFun_Qwiic_5V_Boost_AP3012K'
OUT = ROOT / 'data/processed/FreezeA-source-pixels/ap3012'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    pdf = SOURCE / 'docs/assets/board_files/schematic.pdf'
    cad = SOURCE / 'Hardware/SparkFun_Qwiic_5V_Boost_AP3012K.kicad_sch'
    tree = sexp(cad.read_text(encoding='utf-8'))
    instance = next(v for v in children(tree, 'symbol')
                    if first(v, 'lib_id') and first(v, 'lib_id')[1] == 'SparkFun-IC-Power:AP3012')
    if first(instance, 'at')[1:] != ['77.47', '44.45', '0'] or first(instance, 'mirror'):
        raise ValueError('sample transform no longer valid')
    lib = next(v for v in children(first(tree, 'lib_symbols'), 'symbol')
               if v[1] == first(instance, 'lib_id')[1])
    unit = int(first(instance, 'unit')[1])
    parts = [v for v in children(lib, 'symbol') if int(v[1].rsplit('_', 2)[-2]) in (0, unit)]
    doc = pymupdf.open(pdf)
    if len(doc) != 1 or tuple(doc[0].rect) != (0., 0., 792., 612.):
        raise ValueError('source PDF page changed')
    OUT.mkdir(parents=True, exist_ok=True)
    image = OUT / 'page.png'
    doc[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).save(image)
    scale = 72 / 25.4
    drawings = doc[0].get_drawings()
    # This sample is angle 0 with no mirror: local symbol y points upward.
    def point(x, y):
        return [(77.47 + float(x)) * scale * 2, (44.45 - float(y)) * scale * 2]
    rectangle = next(r for part in parts for r in children(part, 'rectangle'))
    a, b = point(*first(rectangle, 'start')[1:]), point(*first(rectangle, 'end')[1:])
    bbox = [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]
    rectangles = [tuple(item[1]) for d in drawings for item in d['items'] if item[0] == 're']
    rectangle_error = min(max(abs(r[i] * 2 - bbox[i]) for i in range(4)) for r in rectangles)
    lines = [tuple(item[1]) + tuple(item[2]) for d in drawings for item in d['items'] if item[0] == 'l']
    pins = []
    excluded = []
    for part in parts:
        for pin in children(part, 'pin'):
            num, name = first(pin, 'number')[1], first(pin, 'name')[1]
            if 'hide' in pin or (first(pin, 'hide') and first(pin, 'hide')[1:] != ['no']):
                excluded.append(num)
                continue
            end = point(*first(pin, 'at')[1:3])
            angle = int(first(pin, 'at')[3]); length = float(first(pin, 'length')[1])
            dx, dy = {0:(length,0),90:(0,length),180:(-length,0),270:(0,-length)}[angle]
            x, y = map(float, first(pin, 'at')[1:3])
            inner = point(x+dx, y+dy)
            expected = end + inner
            error = min(min(max(abs(l[i]*2-expected[i]) for i in range(4)),
                            max(abs((l[2:]+l[:2])[i]*2-expected[i]) for i in range(4))) for l in lines)
            pins.append({'number':num, 'name':name, 'point_xy':end,
                         'body_contact_xy':inner, 'hidden':False,
                         'cad_to_pdf_pin_segment_max_error_px':error,
                         'visible_number_and_name_reviewed':True})
    if rectangle_error > 0.02 or len(pins) != 5 or any(p['cad_to_pdf_pin_segment_max_error_px'] > .02 for p in pins):
        raise ValueError('original PDF does not agree with original CAD geometry')
    # Hand-reviewed visible strings/boxes in the 144 dpi image; no OCR output.
    words = [('U5',[431,196,448,207],None),('AP3012',[417,211,460,221],None),
             ('5',[384,226,394,236],'5'),('IN',[400,232,413,243],'5'),
             ('4',[384,255,394,265],'4'),('SHDN',[400,258,430,271],'4'),
             ('1',[484,226,494,236],'1'),('SW',[462,232,479,243],'1'),
             ('3',[484,255,494,265],'3'),('FB',[464,260,478,271],'3'),
             ('2',[429,281,440,292],'2'),('GND',[434,253,445,277],'2')]
    annotation = {'schema':'partial-source-annotation-v1','page':0,'image':str(image.relative_to(ROOT)),
        'image_sha256':sha(image),'image_size_xy':[1584,1224],
        'source_pdf':str(pdf.relative_to(ROOT)),'source_pdf_sha256':sha(pdf),
        'source_cad':str(cad.relative_to(ROOT)),'source_cad_sha256':sha(cad),
        'renderer':{'name':'PyMuPDF','version':pymupdf.VersionBind,'scale':2,'dpi':144,'rotation':0},
        'transform':{'mm_to_pixel_scale':scale*2,'offset_xy':[0,0],
                     'library_y_axis':'up; instance/page y axis down',
                     'scope':'only this unrotated, unmirrored unit; not generic KiCad transform'},
        'components':[{'key':'U5','name':'AP3012','type':'IC','body_bbox_xyxy':bbox,'pins':pins,
            'source_property_reference':'U1','source_project_references':['U1','U5'],
            'reference_resolution':'PDF visible U5 manually reviewed; CAD property differs',
            'body_rectangle_max_error_px':rectangle_error}],
        'tokens':[{'text':text,'bbox_xyxy':box,'component_key':'U5','pin_number':num,
                   'source':'visual transcription/review of source PDF by coding agent; coarse text box',
                   'overbar':text=='SHDN','vertical':text in ('2','GND')} for text,box,num in words],
        'excluded_hidden_pins':excluded,'full_page_annotation_complete':False,
        'design_independence_verified':False,'eligible_for_freeze_a_scoring':False,
        'training_use_allowed_by_this_manifest':False,'inference_runs':0,
        'license':{'hardware':'CC BY-SA 4.0','attribution':'SparkFun Electronics; Designed by P. Lewis',
                   'source_license_path':str((SOURCE/'LICENSE.md').relative_to(ROOT)),
                   'source_license_sha256':sha(SOURCE/'LICENSE.md'),
                   'license_url':'https://creativecommons.org/licenses/by-sa/4.0/',
                   'source_url':'https://github.com/sparkfun/SparkFun_Qwiic_5V_Boost_AP3012K',
                   'derivative_changes':'PDF rasterized at 144 dpi; one symbol annotated; review crops and overlays added',
                   'redistribution_scope_review':'local derivative kept with source attribution; excluded from submission ZIP'},
        'limitations':['Partial one-component annotation; other page components remain unannotated.',
            'Hidden/off-unit pins must not become visible image ground truth.',
            'No learned model has seen this rendered image in this workflow.',
            'Historical corpora design-level exclusion remains unverified; zero byte overlap is insufficient.',
            'Text boxes are coarse coding-agent visual review bounds; independent human review is pending.',
            'Source title states the board is based on Indoor Air Quality Combo and SparkX Qwiic Boost; those relatives must be excluded as one design family.']}
    (OUT/'annotation.json').write_text(json.dumps(annotation,ensure_ascii=False,indent=2),encoding='utf-8')
    # Review crop and overlay are derivative evidence, never recognizer input.
    from PIL import Image, ImageDraw
    im=Image.open(image); im.crop((365,190,520,325)).resize((620,540)).save(OUT/'u5-review.png')
    overlay=im.copy(); draw=ImageDraw.Draw(overlay)
    draw.rectangle(bbox,outline='blue',width=2)
    for pin in pins:
        x,y=pin['point_xy']; draw.ellipse((x-3,y-3,x+3,y+3),outline='blue',width=2)
    for _,box,_ in words: draw.rectangle(box,outline='magenta',width=1)
    overlay.crop((365,190,520,325)).resize((620,540)).save(OUT/'u5-overlay.png')
    audit={'status':'PARTIAL_PIXEL_SAMPLE_NOT_READY_FOR_BLIND_TEST','rendered_pages':1,
           'partially_annotated_components':1,'visible_pins':len(pins),'visible_tokens':len(words),
           'rectangle_max_error_px':rectangle_error,'pin_segment_max_error_px':max(p['cad_to_pdf_pin_segment_max_error_px'] for p in pins),
           'annotation_sha256':sha(OUT/'annotation.json'),'verified_independent_designs':0,
           'complete_pixel_gt_pages':0,'independent_human_review_complete':False,'inference_runs':0}
    (ROOT/'data/splits/freeze-a-pixel-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__ == '__main__': main()
