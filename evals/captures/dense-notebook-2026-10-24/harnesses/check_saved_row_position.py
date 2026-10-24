from pathlib import Path
import fitz

out = Path(__file__).resolve().parent
path = next((out/'live-contrast-hybrid-v16-files').rglob('dense-hybrid.pdf'))
with fitz.open(path) as pdf:
    page = pdf[1]
    hits = [[r.x0/page.rect.width, r.y0/page.rect.height,
             r.x1/page.rect.width, r.y1/page.rect.height] for r in page.search_for('reachable')]
    print({'saved_pdf': str(path), 'reachable_rectangles': hits})
    assert any(.223 <= (r[1]+r[3])/2 <= .235 for r in hits), 'Search box is above the cyan source line'
