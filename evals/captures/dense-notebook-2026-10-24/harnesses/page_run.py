import os
os.environ['TORCH_DEVICE']='cpu'
os.environ['TQDM_DISABLE']='1'
import asyncio,base64,hashlib,io,json,sys,time
from pathlib import Path
from dataclasses import asdict
from PIL import Image
from openai.resources.chat.completions import AsyncCompletions
from pdf_ocr.core.grounded import PromptedGroundedOCR
from pdf_ocr.core._layout import searchable_text,split_multi_line_bbox
from pdf_ocr.evaluation import load_ground_truth,compute_report

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
stem,pages,label=sys.argv[1:]
original=AsyncCompletions.create
count=0
async def capture(resource,**kwargs):
    global count
    count+=1; n=count;t=time.monotonic()
    kwargs['seed']=0
    image_part=next(x for x in kwargs['messages'][0]['content'] if x['type']=='image_url')
    data=base64.b64decode(image_part['image_url']['url'].split(',',1)[1])
    with Image.open(io.BytesIO(data)) as img: size=img.size
    result=await original(resource,**kwargs)
    payload={'elapsed_s':time.monotonic()-t,'size':size,'image_sha256':hashlib.sha256(data).hexdigest(),
             'temperature':kwargs.get('temperature'),'max_tokens':kwargs.get('max_tokens'),'seed':0,'response':result.model_dump()}
    (OUT/f'{label}-completion-{n:03}.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'{label} request {n}: {payload["elapsed_s"]:.2f}s {result.choices[0].finish_reason}',flush=True)
    return result
AsyncCompletions.create=capture

async def main():
    backend=PromptedGroundedOCR(api_base='http://127.0.0.1:28127/v1',model='lightonai/LightOnOCR-3-4B')
    t=time.monotonic()
    try: result=await backend.ocr_document(str(ROOT/'examples'/f'{stem}.pdf'),pages=pages)
    finally: await backend.aclose()
    (OUT/f'{label}-blocks.json').write_text(json.dumps(asdict(result),indent=2,ensure_ascii=False),encoding='utf-8')
    gt,_=load_ground_truth(ROOT/'tests/fixtures'/f'ground_truth_{stem}.json')
    metrics={}
    for page in result.page_indices:
        subset=[b for b in gt if b.page_index==page]
        pairs=[pair for b in result.blocks if b.page_index==page for pair in split_multi_line_bbox(b.bbox,searchable_text(b.text))]
        report=compute_report(stem,subset,pairs,pipeline_page_indices=[page]*len(pairs))
        metrics[str(page+1)]=report.to_metrics_dict()
    (OUT/f'{label}-metrics.json').write_text(json.dumps({'elapsed_s':time.monotonic()-t,'page_errors':result.page_errors,'metrics':metrics},indent=2),encoding='utf-8')
    print(metrics, result.page_errors,flush=True)
    if result.page_errors: raise SystemExit(2)

asyncio.run(main())
