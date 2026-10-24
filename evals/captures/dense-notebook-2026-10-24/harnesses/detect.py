import os
os.environ['TORCH_DEVICE']='cpu'
os.environ['TQDM_DISABLE']='1'
import json
from pathlib import Path
from pdf_ocr.core.aligner import HybridAligner

out=Path(__file__).resolve().parent
aligner=HybridAligner()
paths=[out/f'{name}-source.png' for name in ['dense-1','dense-2','dense-3','notes-1','notes-6','notes-17']]
for path in paths:
    boxes=aligner.get_detected_boxes_batch([path.read_bytes()])[0]
    (out/f'{path.stem}-boxes.json').write_text(json.dumps([list(b) for b in boxes],indent=2),encoding='utf-8')
    print(path.stem,len(boxes),flush=True)
