"""Replay strengthened checks against the saved pre-zoom searchable PDFs."""
from pathlib import Path
import fitz

OUT=Path(__file__).resolve().parent
failures=[]
for mode in ['hybrid','grounded']:
    path=next((OUT/f'live-layout-{mode}-final-files').rglob(f'dense-{mode}.pdf'))
    with fitz.open(path) as doc:
        page=doc[2]
        clip=fitz.Rect(.70*page.rect.width,.01*page.rect.height,.98*page.rect.width,.14*page.rect.height)
        for phrase in ['three tuples','best normal form']:
            if not page.search_for(phrase,clip=clip):
                failures.append({'mode':mode,'missing_printed_phrase':phrase})
assert not failures, failures
