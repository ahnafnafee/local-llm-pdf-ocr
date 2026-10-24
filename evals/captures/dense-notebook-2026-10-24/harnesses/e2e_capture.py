import hashlib, json, os, time
from pathlib import Path
from openai.resources.chat.completions import AsyncCompletions

OUTPUT=Path(__file__).resolve().parent
LABEL=os.environ.get('OCR_CAPTURE_LABEL','live')
original=AsyncCompletions.create
number=0

async def capture(resource, **kwargs):
    global number
    number+=1;n=number;t=time.monotonic()
    result=await original(resource,**kwargs)
    record={'elapsed_s':time.monotonic()-t,'max_tokens':kwargs.get('max_tokens'),
            'temperature':kwargs.get('temperature'),'seed':kwargs.get('seed'),
            'response':result.model_dump()}
    (OUTPUT/f'{LABEL}-completion-{n:04}.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    return result

AsyncCompletions.create=capture

def pytest_runtest_logreport(report):
    if report.when=='call':
        with (OUTPUT/f'{LABEL}-tests.jsonl').open('a',encoding='utf-8') as out:
            out.write(json.dumps({'test':report.nodeid,'outcome':report.outcome,'duration_s':report.duration})+'\n')
