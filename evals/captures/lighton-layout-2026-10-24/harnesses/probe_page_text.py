import asyncio,json,os,time,sys
from pathlib import Path
from pdf_ocr import PDFHandler,OCRProcessor
from pdf_ocr.evaluation import load_doc_checks,run_doc_checks
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
os.environ['NO_PROXY']='127.0.0.1,localhost'
async def main():
    rows=[];ocr=OCRProcessor(api_base='http://127.0.0.1:28127/v1',model='lightonai/LightOnOCR-3-4B')
    try:
        for dim,lossless in [(1024,False),(2048,True)]:
            for stem in ['notes','hybrid']:
                image=PDFHandler().convert_to_images(ROOT/'examples'/f'{stem}.pdf',dpi=200,max_image_dim=dim,pages='1',lossless=lossless)[0]
                for temp in [0,.2]:
                    ocr.temperature=temp
                    for seed in ([0] if temp==0 else [0,1,2]):
                        ocr.client.generation_options['seed']=seed;t=time.monotonic();lines=await ocr.perform_ocr(image)
                        row=dict(document=stem,dim=dim,lossless=lossless,seed=seed,temp=temp,seconds=time.monotonic()-t,text='\n'.join(lines))
                        row['checks']=run_doc_checks(row['text'],load_doc_checks(ROOT/f'evals/checks/{stem}.jsonl'))
                        rows.append(row);print({k:v for k,v in row.items() if k!='text'},flush=True)
                        (OUT/'page-text-results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    finally:await ocr.aclose()
asyncio.run(main())
