import sys, json, time
from pathlib import Path
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT / 'layout-deps'))
import torch, fitz
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection
torch.set_num_threads(4)
path = 'D:/llm/models/PP-DocLayoutV3_safetensors'
model = AutoModelForObjectDetection.from_pretrained(path).eval()
processor = AutoImageProcessor.from_pretrained(path)
for stem in ['hybrid', 'notes', 'handwritten', 'dense']:
    with fitz.open(OUT.parents[1] / 'examples' / f'{stem}.pdf') as doc:
        pix = doc[0].get_pixmap(dpi=200)
    image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    image.thumbnail((2048,2048))
    image.save(OUT/f'{stem}.png')
    started = time.monotonic()
    with torch.inference_mode():
        result = processor.post_process_object_detection(model(**processor(images=image,return_tensors='pt')),threshold=0.3,target_sizes=[image.size[::-1]])[0]
    rows=[]
    draw=ImageDraw.Draw(image)
    for i,(score,label,box,poly) in enumerate(zip(result['scores'],result['labels'],result['boxes'],result['polygon_points'])):
        box=box.tolist(); label=model.config.id2label[int(label)]
        rows.append(dict(order=i,label=label,score=float(score),bbox=box,polygon=poly.tolist()))
        draw.rectangle(box,outline='red',width=2)
        draw.text((box[0],box[1]),f'{i}: {label}',fill='red',stroke_width=1,stroke_fill='white')
    (OUT/f'{stem}-layout.json').write_text(json.dumps(dict(size=image.size,seconds=time.monotonic()-started,regions=rows),indent=2),encoding='utf-8')
    image.save(OUT/f'{stem}-layout.png')
    print(stem,len(rows),round(time.monotonic()-started,2),flush=True)
