from pathlib import Path
import json
import fitz

out = Path(__file__).resolve().parent
path = next((out/'live-rows-hybrid-v17-files').rglob('dense-hybrid.pdf'))
observations = []
with fitz.open(path) as pdf:
    for word, y0, y1 in [('candidate keys',.039,.045), ('three tuples',.060,.067),
                        ('best normal form',.123,.130)]:
        hits = [[r.x0/pdf[2].rect.width, r.y0/pdf[2].rect.height,
                 r.x1/pdf[2].rect.width, r.y1/pdf[2].rect.height] for r in pdf[2].search_for(word)]
        observations.append({'term':word, 'rectangles':hits,
                             'passed':any(y0 <= (r[1]+r[3])/2 <= y1 for r in hits)})
(out/'paragraph-position-red.json').write_text(json.dumps(observations,indent=2)+'\n')
print(json.dumps(observations,indent=2))
assert all(row['passed'] for row in observations), 'Printed paragraph words occupy the wrong source rows'
