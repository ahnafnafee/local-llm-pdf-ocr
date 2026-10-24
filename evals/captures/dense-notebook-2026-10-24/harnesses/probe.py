import asyncio
import base64
import io
import json
import sys
import time
from pathlib import Path

import fitz
from PIL import Image
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from pdf_ocr.core.grounded import _parse_lighton_grounding
from dataclasses import asdict

async def main():
    name, stem, page, dpi, prompt, *bounds = sys.argv[1:]
    page, dpi = int(page), int(dpi)
    bounds = list(map(float, bounds)) or [0, 0, 1, 1]
    with fitz.open(ROOT / 'examples' / f'{stem}.pdf') as doc:
        pix = doc[page - 1].get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    source = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    crop = source.crop(tuple(round(v * source.size[i % 2]) for i, v in enumerate(bounds)))
    crop.thumbnail((2048, 2048))
    crop.save(OUT / f'{name}.png')
    buf = io.BytesIO()
    crop.save(buf, format='PNG')
    content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()}}]
    if prompt != 'plain':
        content.append({'type': 'text', 'text': prompt})
    started = time.monotonic()
    async with AsyncOpenAI(base_url='http://127.0.0.1:28127/v1', api_key='not-required', max_retries=0) as client:
        result = await client.chat.completions.create(model='lightonai/LightOnOCR-3-4B', messages=[{'role':'user', 'content':content}], temperature=0, top_p=.9, seed=0, max_tokens=4096, timeout=180)
    raw = result.choices[0].message.content or ''
    record = {'document':stem, 'page':page, 'dpi':dpi, 'bounds':bounds, 'size':crop.size, 'prompt':prompt, 'elapsed_s':time.monotonic()-started, 'response':result.model_dump()}
    if prompt == 'grounding' and result.choices[0].finish_reason != 'length':
        try:
            blocks = _parse_lighton_grounding(raw, page-1)
            for b in blocks:
                b.bbox = [bounds[i % 2] + b.bbox[i] * (bounds[2 + i % 2]-bounds[i % 2]) for i in range(4)]
            record['blocks'] = [asdict(b) for b in blocks]
        except Exception as exc:
            record['parse_error'] = str(exc)
    (OUT / f'{name}.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUT / f'{name}.txt').write_text(raw, encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k not in {'response','blocks'}}, ensure_ascii=False), flush=True)
    print(raw, flush=True)

asyncio.run(main())
