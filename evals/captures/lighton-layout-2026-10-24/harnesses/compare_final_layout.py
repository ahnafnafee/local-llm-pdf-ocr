"""Compare fixed native predictions with conservative PP region anchoring."""
import json
from pathlib import Path

from pdf_ocr.core.native_layout import anchor_native_blocks
from pdf_ocr.evaluation import compute_report, load_ground_truth

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
rows = []
for stem in ['hybrid', 'notes', 'handwritten', 'dense']:
    regions = json.loads((OUT / f'{stem}-layout.json').read_text())
    width, height = regions['size']
    boxes = [[v / (width if i % 2 == 0 else height) for i, v in enumerate(r['bbox'])]
             for r in regions['regions'] if r['score'] >= .5 and r['label'] != 'image']
    response = json.loads((OUT / f'lighton-final-grounded-{stem}-blocks.json').read_text())
    native = [(b['bbox'], b['text']) for b in response['blocks'] if b['page_index'] == 0]
    gt, _ = load_ground_truth(ROOT / f'tests/fixtures/ground_truth_{stem}.json')
    gt = [b for b in gt if b.page_index == 0]
    variants = {'native_blocks': native, 'native_split_lines': anchor_native_blocks(native, []),
                'pp_anchored_lines': anchor_native_blocks(native, boxes)}
    rows.append({'document': stem, 'source_page': 1, 'pp_regions': len(boxes),
                 'variants': {name: compute_report(stem, gt, blocks).to_metrics_dict()
                              for name, blocks in variants.items()}})
(OUT / 'layout-final-comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(rows, indent=2))
