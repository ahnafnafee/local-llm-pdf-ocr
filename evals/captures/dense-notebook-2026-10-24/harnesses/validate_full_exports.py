"""Inspect PDF exports made from actual full-corpus inference captures."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import fitz
from pdf_ocr.core.pdf import PDFHandler

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--mode',required=True)
parser.add_argument('--document',choices=['dense','notes'],required=True)
args=parser.parse_args()
prefix=f'final-{args.mode}-{args.document}'
pairs=json.loads((OUT/f'{prefix}-output.json').read_text())
indices=json.loads((OUT/f'{prefix}-page-indices.json').read_text())
pages=defaultdict(list)
for pair,index in zip(pairs,indices):
    pages[index].append(pair)
output=OUT/f'{prefix}-export.pdf'
PDFHandler().embed_structured_text(ROOT/f'examples/{args.document}.pdf',output,pages)
observations=[]
with fitz.open(output) as pdf:
    if args.document=='dense':
        checks=[(1,'Snowball',[.15,.06,.33,.13]),(1,'reachable',[.07,.223,.32,.235]),
                (2,'BCNF',[.02,.28,.32,.34]),(2,'candidate keys',[.70,.039,.98,.045]),
                (2,'three tuples',[.70,.060,.98,.067]),(2,'best normal form',[.70,.123,.98,.130])]
    else:
        checks=[(5,'FNAME',[.15,.10,.24,.145]),(5,'MINIT',[.22,.10,.30,.145]),
                (5,'LNAME',[.28,.10,.36,.145]),(5,'DLOCATION',[.42,.22,.55,.27]),
                (5,'SEX',[.52,.39,.56,.43]),(12,'Customer',[.13,.13,.5,.18]),
                (12,'Fairfax',[.17,.49,.85,.56]),(16,'Sailors',[.17,.20,.85,.43]),
                (16,'Reserves',[.17,.20,.85,.43]),(16,'Boats',[.17,.20,.85,.43])]
    for index,term,region in checks:
        page=pdf[index]
        hits=[list(r) for r in page.search_for(term) if
              region[0]<=(r.x0+r.x1)/2/page.rect.width<=region[2] and
              region[1]<=(r.y0+r.y1)/2/page.rect.height<=region[3]]
        observations.append({'source_page':index+1,'term':term,'matches':hits,'passed':bool(hits)})
    if args.document=='notes':
        observations.append({'assertion':'Exactly two SEX labels','passed':len(pdf[5].search_for('SEX'))==2})
        observations.append({'assertion':'Last page text','passed':pdf[20].get_text().strip()=='Project 1-3'})
    else:
        page=pdf[2]
        clip=fitz.Rect(.70*page.rect.width,.01*page.rect.height,.98*page.rect.width,.14*page.rect.height)
        observations.append({'assertion':'No invented Value table','passed':not page.search_for('Value',clip=clip)})
(OUT/f'{prefix}-export-checks.json').write_text(json.dumps(observations,indent=2)+'\n')
print(json.dumps(observations,indent=2))
assert all(row['passed'] for row in observations)
