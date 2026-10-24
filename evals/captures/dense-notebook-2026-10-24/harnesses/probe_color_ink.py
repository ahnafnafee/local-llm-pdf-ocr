"""Compare exact source-crop color and minimum-channel ink contrast."""
import asyncio
import base64
import io
import json
import os
from pathlib import Path
os.environ['TORCH_DEVICE']='cpu'
os.environ['NO_PROXY']='127.0.0.1,localhost,::1'
os.environ['no_proxy']=os.environ['NO_PROXY']
from PIL import ImageChops
from pdf_ocr.core.aligner import HybridAligner
from pdf_ocr.core.grounded import PromptedGroundedOCR
from pdf_ocr.core.pdf import PDFHandler
from pdf_ocr.core.native_regions import detect_missing_lines,plan_regions,region_bounds,repair_targets

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def encoded(image):
    b=io.BytesIO(); image.save(b,format='PNG'); return b.getvalue()

async def main():
    handler=PDFHandler()
    page=handler.convert_to_images(ROOT/'examples/dense.pdf',dpi=200,max_image_dim=2048,pages='2',lossless=True)[1]
    aligner=HybridAligner()
    boxes=aligner.get_detected_boxes_batch([base64.b64decode(page)])[0]
    source=handler.render_page_source(ROOT/'examples/dense.pdf',1)
    async def detect(images):
        return aligner.get_detected_boxes_batch(images)
    boxes=await detect_missing_lines(source,boxes,detect)
    targets=repair_targets([],boxes,source,[[0,0,1,1]])[1]
    groups=plan_regions(targets)
    group=next(g for g in groups if (lambda b:b[0]<.19<b[2] and b[1]<.22<b[3])(region_bounds(g)))
    bounds=region_bounds(group,neighbors=boxes)
    crop=source.crop(tuple(round(v*source.size[i%2]) for i,v in enumerate(bounds)))
    crop.thumbnail((2048,2048))
    r,g,b=crop.split()
    dark=ImageChops.darker(ImageChops.darker(r,g),b).convert('RGB')
    backend=PromptedGroundedOCR(api_base='http://127.0.0.1:28127/v1',model='lightonai/LightOnOCR-3-4B')
    backend.client.generation_options['seed']=0
    results={}
    try:
        for name,image in [('original',crop),('minimum-channel',dark)]:
            image.save(OUT/f'colored-ink-{name}.png')
            result=await backend.client.complete('grounding',base64.b64encode(encoded(image)).decode(),
                   model=backend.model,timeout=240,max_tokens=max(512,min(2048,len(group)*128)),temperature=0)
            results[name]=result
            print(name,result,flush=True)
    finally:
        await backend.aclose()
    (OUT/'colored-ink-comparison.json').write_text(json.dumps({'bounds':bounds,'group':group,'results':results},indent=2)+'\n')
asyncio.run(main())
