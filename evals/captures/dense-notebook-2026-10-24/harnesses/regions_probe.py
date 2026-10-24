import asyncio, base64, io, json, sys, time
from dataclasses import asdict
from pathlib import Path
import fitz
from PIL import Image
from openai import AsyncOpenAI
from pdf_ocr.core.grounded import _parse_lighton_grounding

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]

def partition(boxes, limit=12):
    if len(boxes)<=limit:
        return [boxes]
    candidates=[]
    for axis in [0,1]:
        ordered=sorted(boxes,key=lambda b:b[axis])
        end=ordered[0][axis+2]
        for i in range(1,len(ordered)):
            gap=ordered[i][axis]-end
            if gap>0.001:
                balance=min(i,len(boxes)-i)/len(boxes)
                candidates.append((gap*(1.4 if axis==0 else 1)*(.5+balance),ordered[:i],ordered[i:]))
            end=max(end,ordered[i][axis+2])
    if candidates:
        _,a,b=max(candidates,key=lambda c:c[0])
    else:
        ordered=sorted(boxes,key=lambda b:(b[1]+b[3])/2)
        a,b=ordered[:len(boxes)//2],ordered[len(boxes)//2:]
    return partition(a,limit)+partition(b,limit)

async def main():
    stem,page=sys.argv[1],int(sys.argv[2])
    name=f'{stem}-{page}'
    boxes=json.loads((OUT/f'{name}-source-boxes.json').read_text())
    boxes=[b for b in boxes if b[2]-b[0]<.97]
    groups=partition(boxes)
    with fitz.open(ROOT/'examples'/f'{stem}.pdf') as doc:
        pix=doc[page-1].get_pixmap(dpi=400,colorspace=fitz.csRGB,alpha=False)
    source=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
    result=[]
    records=[]
    async with AsyncOpenAI(base_url='http://127.0.0.1:28127/v1',api_key='not-required',max_retries=0) as client:
        for i,group in enumerate(groups):
            bounds=[max(0,min(b[a] for b in group)-.008) if a<2 else min(1,max(b[a] for b in group)+.008) for a in range(4)]
            crop=source.crop(tuple(round(v*source.size[a%2]) for a,v in enumerate(bounds)))
            crop.thumbnail((2048,2048))
            buf=io.BytesIO();crop.save(buf,format='PNG')
            content=[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}},{'type':'text','text':'grounding'}]
            t=time.monotonic()
            response=await client.chat.completions.create(model='lightonai/LightOnOCR-3-4B',messages=[{'role':'user','content':content}],temperature=0,top_p=.9,seed=0,max_tokens=3072,timeout=120)
            record={'bounds':bounds,'detected':group,'elapsed_s':time.monotonic()-t,'response':response.model_dump()}
            try:
                if response.choices[0].finish_reason=='length': raise ValueError('length')
                blocks=_parse_lighton_grounding(response.choices[0].message.content,page-1)
                for b in blocks:
                    b.bbox=[bounds[a%2]+b.bbox[a]*(bounds[2+a%2]-bounds[a%2]) for a in range(4)]
                result.extend(blocks)
            except Exception as exc: record['error']=str(exc)
            records.append(record)
            print(name,i+1,'/',len(groups),'boxes',len(group),'seconds',round(record['elapsed_s'],2),record.get('error',''),flush=True)
            (OUT/f'{name}-regions.json').write_text(json.dumps({'records':records,'blocks':[asdict(b) for b in result]},indent=2,ensure_ascii=False),encoding='utf-8')

asyncio.run(main())
