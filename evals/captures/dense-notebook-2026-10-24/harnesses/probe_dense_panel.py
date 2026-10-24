"""Exercise tiny printed questions through the actual selected-page pipeline."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
os.environ['TORCH_DEVICE']='cpu'
os.environ['NO_PROXY']='127.0.0.1,localhost,::1'
os.environ['no_proxy']=os.environ['NO_PROXY']
os.environ['OCR_CAPTURE_LABEL']='printed-panel-final-probe'
import e2e_capture
import fitz
from pdf_ocr import HybridAligner, OCRProcessor, OCRPipeline, PDFHandler
from pdf_ocr.core._layout import searchable_text
from pdf_ocr.evaluation import compute_report,load_ground_truth

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
captured={}
handler=PDFHandler()
def writer(source,output,data,dpi):
    captured.update(data)
    handler.embed_structured_text(source,output,data,dpi)

async def main():
    processor=OCRProcessor(api_base='http://127.0.0.1:28127/v1',model='lightonai/LightOnOCR-3-4B')
    processor.client.generation_options['seed']=0
    output=OUT/'printed-panel-final-probe.pdf'
    try:
        pipe=OCRPipeline(aligner=HybridAligner(),ocr_processor=processor,pdf_handler=handler,output_writer=writer)
        await pipe.run(str(ROOT/'examples/dense.pdf'),str(output),pages='3',max_image_dim=2048,concurrency=1)
    finally:
        await processor.aclose()
    pairs=captured[2]
    gt,_=load_ground_truth(ROOT/'tests/fixtures/ground_truth_dense.json')
    report=compute_report('dense-page3',[replace(b,text=searchable_text(b.text)) for b in gt if b.page_index==2],
                          [(b,searchable_text(t)) for b,t in pairs],pipeline_page_indices=[2]*len(pairs))
    with fitz.open(output) as doc:
        page=doc[2]
        text=page.get_text(clip=fitz.Rect(.70*page.rect.width,.01*page.rect.height,.98*page.rect.width,.14*page.rect.height))
    print(text,flush=True)
    print(report.to_metrics_dict(),flush=True)
    result={'metrics':report.to_metrics_dict(),'blocks':pairs,'printed_text':text}
    (OUT/'printed-panel-final-probe.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    assert 'three tuples' in text and 'best normal form' in text
    assert 'Attribute' not in text and 'Value' not in text

asyncio.run(main())
