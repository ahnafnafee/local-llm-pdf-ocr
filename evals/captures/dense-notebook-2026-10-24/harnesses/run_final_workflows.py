"""Run real workflows serially and retain each command, exit code and log."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PYTHON = str(ROOT / '.venv/Scripts/python.exe')
MODEL = 'lightonai/LightOnOCR-3-4B'
ENV = {**os.environ, 'TORCH_DEVICE': 'cpu', 'NO_PROXY': '127.0.0.1,localhost,::1',
       'no_proxy': '127.0.0.1,localhost,::1', 'TQDM_DISABLE': '1'}
ENV.pop('CONFIDENCE_DOCUMENTS', None)
RECORD = OUT / 'final-workflow-status.json'
records = json.loads(RECORD.read_text()) if RECORD.exists() else {}


def run(label, args, environment=None):
    assert label not in records, f'Preserve previous run evidence: {label}'
    command = [PYTHON, '-X', 'utf8', *args]
    print('Starting', label, flush=True)
    started = time.monotonic()
    with (OUT / f'{label}.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=ROOT, env={**ENV, **(environment or {})},
                                stdout=log, stderr=subprocess.STDOUT)
    records[label] = {'command': command, 'exit_code': result.returncode,
                      'elapsed_s': round(time.monotonic()-started, 2)}
    RECORD.write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
    print(label, records[label], flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


parser = argparse.ArgumentParser()
parser.add_argument('stages', nargs='+')
args = parser.parse_args()
for stage in args.stages:
    if stage in {'full-grounded', 'full-hybrid'}:
        mode = stage.removeprefix('full-')
        label = f'final-{mode}'
        run(label, [str(OUT / 'run_confidence.py'), '--path', mode,
                    '--api-base', 'http://127.0.0.1:28127/v1', f'--{mode}-model', MODEL,
                    '--max-image-dim', '2048', '--baseline-dir', str(OUT / 'no-baselines')],
            {'CONFIDENCE_RUN_LABEL': label})
    elif stage == 'cli':
        for mode in ['hybrid', 'text', 'grounded']:
            run(f'final-selection-{mode}', [str(OUT / 'page_selection_e2e.py'),
                                            '--mode', mode, '--model', MODEL, '--use-default'])
    elif stage in {'export-grounded', 'export-hybrid'}:
        mode = stage.removeprefix('export-')
        for document in ['dense', 'notes']:
            run(f'final-export-{mode}-{document}', [str(OUT/'validate_full_exports.py'),
                                                  '--mode', mode, '--document', document])
    elif stage == 'image':
        for mode in ['hybrid', 'grounded']:
            run(f'final-image-{mode}', [str(OUT / 'image_confidence.py'), '--path', mode, '--model', MODEL])
    elif stage == 'web':
        import httpx

        with (OUT / 'web-server.log').open('w', encoding='utf-8') as log:
            server = subprocess.Popen([PYTHON, '-X', 'utf8', '-m', 'uvicorn',
                                       'pdf_ocr.server:app', '--host', '127.0.0.1', '--port', '28129'],
                                       cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                with httpx.Client(timeout=3, trust_env=False) as client:
                    for attempt in range(90):
                        if server.poll() is not None:
                            raise RuntimeError('Web process stopped before readiness')
                        try:
                            if client.get('http://127.0.0.1:28129/').status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        time.sleep(1)
                    else:
                        raise RuntimeError('Web readiness timed out')
                for mode in ['hybrid', 'grounded']:
                    run(f'final-web-{mode}', [str(OUT / 'web_e2e.py'), '--mode', mode,
                                             '--model', MODEL, '--use-default'])
            finally:
                # Windows venv launchers can leave a child interpreter behind.
                # Stop only the process tree created and still owned here.
                if server.poll() is None:
                    subprocess.run(['taskkill', '/PID', str(server.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                server.wait(timeout=30)
    elif stage == 'offline':
        run('offline-final-v18', ['-m', 'pytest', '-m', 'not live_llm', '-q'])
    elif stage == 'live-paragraphs':
        for mode in ['hybrid', 'grounded']:
            label = f'live-paragraphs-{mode}-v18'
            run(label, ['-m', 'pytest', 'tests/test_live_quality.py', '-k', 'complex and dense',
                        '-m', 'live_llm', '-p', 'e2e_capture', '-x', '-q',
                        '--basetemp', str(OUT / f'{label}-files')],
                {'OCR_LIVE_API_BASE': 'http://127.0.0.1:28127/v1', 'OCR_LIVE_MODE': mode,
                 'OCR_CAPTURE_LABEL': label, 'PYTHONPATH': str(OUT)})
    elif stage == 'live-rows':
        for mode in ['hybrid', 'grounded']:
            label = f'live-rows-{mode}-v17'
            run(label, ['-m', 'pytest', 'tests/test_live_quality.py', '-k', 'complex and dense',
                        '-m', 'live_llm', '-p', 'e2e_capture', '-x', '-q',
                        '--basetemp', str(OUT / f'{label}-files')],
                {'OCR_LIVE_API_BASE': 'http://127.0.0.1:28127/v1', 'OCR_LIVE_MODE': mode,
                 'OCR_CAPTURE_LABEL': label, 'PYTHONPATH': str(OUT)})
    elif stage == 'live-contrast':
        for mode in ['hybrid', 'grounded']:
            label = f'live-contrast-{mode}-v16'
            run(label, ['-m', 'pytest', 'tests/test_live_quality.py', '-k', 'complex and dense',
                        '-m', 'live_llm', '-p', 'e2e_capture', '-x', '-q',
                        '--basetemp', str(OUT / f'{label}-files')],
                {'OCR_LIVE_API_BASE': 'http://127.0.0.1:28127/v1', 'OCR_LIVE_MODE': mode,
                 'OCR_CAPTURE_LABEL': label, 'PYTHONPATH': str(OUT)})
    elif stage == 'live-overlap':
        for mode in ['grounded', 'hybrid']:
            label = f'live-overlap-{mode}-v15'
            run(label, ['-m', 'pytest', 'tests/test_live_quality.py', '-k', 'complex and dense',
                        '-m', 'live_llm', '-p', 'e2e_capture', '-x', '-q',
                        '--basetemp', str(OUT / f'{label}-files')],
                {'OCR_LIVE_API_BASE': 'http://127.0.0.1:28127/v1', 'OCR_LIVE_MODE': mode,
                 'OCR_CAPTURE_LABEL': label, 'PYTHONPATH': str(OUT)})
    elif stage == 'live-empty-tag':
        for mode in ['hybrid', 'grounded']:
            label = f'live-empty-tag-{mode}-v14'
            run(label, ['-m', 'pytest', 'tests/test_live_quality.py', '-k', 'complex and notes',
                        '-m', 'live_llm', '-p', 'e2e_capture', '-x', '-q',
                        '--basetemp', str(OUT / f'{label}-files')],
                {'OCR_LIVE_API_BASE': 'http://127.0.0.1:28127/v1', 'OCR_LIVE_MODE': mode,
                 'OCR_CAPTURE_LABEL': label, 'PYTHONPATH': str(OUT)})
    elif stage == 'live-complete':
        for mode in ['hybrid', 'grounded']:
            label = f'live-complete-{mode}-final'
            run(label, ['-m', 'pytest', 'tests/test_live_quality.py',
                        '-m', 'live_llm', '-p', 'e2e_capture', '-x', '-q',
                        '--basetemp', str(OUT / f'{label}-files')],
                {'OCR_LIVE_API_BASE': 'http://127.0.0.1:28127/v1', 'OCR_LIVE_MODE': mode,
                 'OCR_CAPTURE_LABEL': label, 'PYTHONPATH': str(OUT)})
    elif stage == 'live-notebook':
        for mode in ['hybrid', 'grounded']:
            label = f'live-notebook-{mode}-final-v12'
            run(label, ['-m', 'pytest', 'tests/test_live_quality.py', '-k', 'complex and notes',
                        '-m', 'live_llm', '-p', 'e2e_capture', '-x', '-q',
                        '--basetemp', str(OUT / f'{label}-files')],
                {'OCR_LIVE_API_BASE': 'http://127.0.0.1:28127/v1', 'OCR_LIVE_MODE': mode,
                 'OCR_CAPTURE_LABEL': label, 'PYTHONPATH': str(OUT)})
    elif stage == 'live-layout':
        for mode in ['hybrid', 'grounded']:
            label = f'live-layout-{mode}-final'
            run(label, ['-m', 'pytest', 'tests/test_live_quality.py', '-k', 'complex',
                        '-m', 'live_llm', '-p', 'e2e_capture', '-x', '-q',
                        '--basetemp', str(OUT / f'{label}-files')],
                {'OCR_LIVE_API_BASE': 'http://127.0.0.1:28127/v1', 'OCR_LIVE_MODE': mode,
                 'OCR_CAPTURE_LABEL': label, 'PYTHONPATH': str(OUT)})
    else:
        raise ValueError(stage)
