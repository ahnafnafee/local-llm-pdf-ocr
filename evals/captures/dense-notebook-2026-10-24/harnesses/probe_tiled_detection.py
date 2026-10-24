"""Check zoomed source detection for content missed by page thumbnails."""
import io
import json
import os
from pathlib import Path
os.environ['TORCH_DEVICE'] = 'cpu'
os.environ['TQDM_DISABLE'] = '1'
from pdf_ocr.core.aligner import HybridAligner
from pdf_ocr.core.pdf import PDFHandler
from pdf_ocr.core.native_regions import intersection, area

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
source = PDFHandler().render_page_source(ROOT/'examples/dense.pdf', 2)
bounds = [[0,0,.55,.55], [.45,0,1,.55], [0,.45,.55,1], [.45,.45,1,1]]
encoded=[]
for box in bounds:
    crop = source.crop(tuple(round(v*source.size[i%2]) for i,v in enumerate(box)))
    buf=io.BytesIO()
    crop.save(buf,format='PNG')
    encoded.append(buf.getvalue())
detector=HybridAligner()
original=json.loads((OUT/'dense-3-source-boxes.json').read_text())
result=[]
for region, boxes in zip(bounds, detector.get_detected_boxes_batch(encoded)):
    mapped=[[region[i%2]+b[i]*(region[i%2+2]-region[i%2]) for i in range(4)] for b in boxes]
    result.extend(b for b in mapped if not any(intersection(b,old)/max(area(b),1e-8)>.5 for old in original))
(OUT/'dense3-zoomed-boxes.json').write_text(json.dumps(result,indent=2)+'\n')
print('New boxes',len(result),'printed panel',sum(b[0]>.65 and b[1]<.14 for b in result))
print([b for b in result if b[0]>.65 and b[1]<.14])
