import asyncio, base64, io, json, os, time
from pathlib import Path
from PIL import Image
from pdf_ocr import OCRProcessor
from pdf_ocr.utils.image import crop_for_ocr
os.environ['NO_PROXY']='127.0.0.1,localhost'
OUT=Path(__file__).resolve().parent
async def main():
    ocr=OCRProcessor(api_base='http://127.0.0.1:28127/v1',model='lightonai/LightOnOCR-3-4B')
    rows=[]
    try:
        for stem,indices in [('hybrid',[11]),('notes',[0]),('handwritten',[0,16])]:
            data=json.loads((OUT/f'{stem}-layout.json').read_text());w,h=data['size']
            image=Image.open(OUT/f'{stem}.png');buf=io.BytesIO();image.save(buf,format='PNG');b64=base64.b64encode(buf.getvalue()).decode()
            for idx in indices:
                region=data['regions'][idx];box=[v/(w if i%2==0 else h) for i,v in enumerate(region['bbox'])]
                for canvas in [None,1024]:
                    crop=crop_for_ocr(b64,box,canvas_size=canvas)
                    for seed in [0,1,2]:
                        ocr.client.generation_options['seed']=seed
                        t=time.monotonic();text=await ocr.perform_ocr_on_crop(crop)
                        row=dict(document=stem,region=idx,canvas=canvas,seed=seed,text=text,seconds=time.monotonic()-t)
                        rows.append(row);print(row,flush=True)
                        (OUT/'region-results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    finally: await ocr.aclose()
asyncio.run(main())
