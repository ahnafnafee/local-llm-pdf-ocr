"""Rescore frozen old and new predictions with identical current references."""
from dataclasses import replace
import argparse
import json
from pathlib import Path

from pdf_ocr.core._layout import searchable_text
from pdf_ocr.evaluation import compute_report, load_ground_truth

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BEFORE = ROOT / 'evals/captures/lighton-layout-2026-10-24'
parser = argparse.ArgumentParser()
parser.add_argument('--available', action='store_true')
args = parser.parse_args()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


rows = []
for mode in ['grounded', 'hybrid']:
    for stem in ['dense', 'notes']:
        if args.available and not (OUT / f'final-{mode}-{stem}-output.json').exists():
            continue
        gt, _ = load_ground_truth(ROOT / f'tests/fixtures/ground_truth_{stem}.json')
        variants = {}
        for label, directory, prefix in [('before', BEFORE, f'lighton-final-{mode}'),
                                          ('after', OUT, f'final-{mode}')]:
            pairs = load(directory / f'{prefix}-{stem}-output.json')
            indices = load(directory / f'{prefix}-{stem}-page-indices.json')
            raw = compute_report(stem, gt, pairs, pipeline_page_indices=indices)
            visible = compute_report(stem, [replace(b, text=searchable_text(b.text)) for b in gt],
                                     [(box, searchable_text(text)) for box, text in pairs],
                                     pipeline_page_indices=indices)
            per_page = []
            for page in sorted(set(indices)):
                selected = [pair for pair, index in zip(pairs, indices) if index == page]
                score = compute_report(stem, [replace(b, text=searchable_text(b.text))
                                              for b in gt if b.page_index == page],
                                       [(box, searchable_text(text)) for box, text in selected],
                                       pipeline_page_indices=[page]*len(selected))
                per_page.append({'source_page': page+1, 'metrics': score.to_metrics_dict()})
            variants[label] = {'raw': raw.to_metrics_dict(), 'visible': visible.to_metrics_dict(),
                               'pages': per_page, 'blocks': len(pairs)}
        rows.append({'document': stem, 'mode': mode, 'variants': variants})

(OUT / ('partial-before-after-comparison.json' if args.available else 'before-after-comparison.json')).write_text(json.dumps({
    'method': 'Both captures use the current source-audited references (dense v3, notes v2), '
              'page-aware Hungarian matching, and identical normalization. Visible text uses '
              'the current HTML/LaTeX decoder for both old and new predictions; no segmentation '
              'is changed. Character recall measures box coverage, not recognition accuracy.',
    'rows': rows,
}, indent=2)+'\n', encoding='utf-8')
for row in rows:
    print(row['document'], row['mode'], {
        label: {key: value['visible'][key] for key in ['bow_f1', 'hmean', 'char_recall']}
        for label, value in row['variants'].items()})
