"""Visualize actual written PDF search rectangles on the source pixels."""
from pathlib import Path
import argparse
import json
import fitz
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--dense-pdf')
parser.add_argument('--notes-pdf')
args = parser.parse_args()
if not args.dense_pdf:
    args.dense_pdf = str(next((OUT/'live-complete-hybrid-final-files').rglob('dense-hybrid.pdf')))
if not args.notes_pdf:
    args.notes_pdf = str(next((OUT/'live-complete-hybrid-final-files').rglob('notes-hybrid.pdf')))
CASES = [
    ('dense-page2', args.dense_pdf, 1,
     ['Snowball', 'reachable'], [0, 0, .36, .32]),
    ('dense-printed-panel', args.dense_pdf, 2,
     ['candidate keys', 'three tuples', 'best normal form'], [.68, .01, .98, .14]),
    ('notebook-schema', args.notes_pdf, 5,
     'EMPLOYEE FNAME MINIT LNAME SSN BDATE ADDRESS SEX SALARY SUPERSSN DNO '
     'DEPARTMENT DNAME DNUMBER MGRSSN MGRSTARTDATE DEPT_LOCATIONS DLOCATION '
     'PROJECT PNAME PNUMBER PLOCATION DNUM WORKS_ON ESSN PNO HOURS '
     'DEPENDENT DEPENDENT_NAME RELATIONSHIP', [.1, .05, .95, .46]),
    ('notebook-photo', args.notes_pdf, 16,
     ['Sailors', 'Reserves', 'Boats'], [.15, .19, .87, .44]),
]
observations = []
for label, name, index, terms, bounds in CASES:
    if isinstance(terms, str):
        terms = terms.split()
    with fitz.open(name) as doc:
        page = doc[index]
        clip = fitz.Rect(*(v * (page.rect.width if i % 2 == 0 else page.rect.height)
                           for i, v in enumerate(bounds)))
        pixmap = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip)
        image = Image.frombytes('RGB', [pixmap.width, pixmap.height], pixmap.samples)
        draw = ImageDraw.Draw(image)
        hits = {}
        for term in terms:
            hits[term] = []
            for rect in page.search_for(term):
                if not clip.intersects(rect):
                    continue
                hits[term].append(list(rect))
                draw.rectangle([(rect.x0-clip.x0)*3, (rect.y0-clip.y0)*3,
                                (rect.x1-clip.x0)*3, (rect.y1-clip.y0)*3],
                               outline=(20, 90, 185), width=2)
        image.save(OUT / f'{label}-search-rectangles.png')
        observations.append({'source_page': index+1, 'pdf': name,
                             'clip_normalized': bounds, 'matches_in_pdf_points': hits})
(OUT / 'search-rectangle-observations.json').write_text(json.dumps(observations, indent=2)+'\n')
