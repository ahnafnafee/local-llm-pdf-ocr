"""Verify final results and the exact archived source/capture hashes."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import fitz

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CAPTURE = ROOT / 'evals/captures/dense-notebook-2026-10-24'
manifest = json.loads((CAPTURE / 'manifest.json').read_text(encoding='utf-8'))


def normalized(path):
    data = path.read_bytes()
    return data if path.suffix in {'.pdf', '.png', '.avif'} else data.replace(b'\r\n', b'\n')


archive_info = manifest.get('raw_archive')
if archive_info:
    raw = (CAPTURE/archive_info['path']).read_bytes()
    assert len(raw) == archive_info['bytes']
    assert hashlib.sha256(raw).hexdigest() == archive_info['sha256']
    archive = zipfile.ZipFile(CAPTURE/archive_info['path'])
    assert archive.testzip() is None
    expected = {name for name, entry in manifest['files'].items()
                if entry.get('storage') == archive_info['path']}
    assert set(archive.namelist()) == expected
    assert len(expected) == archive_info['members']
else:
    archive = None
try:
    for name, entry in manifest['files'].items():
        data = (archive.read(name) if entry.get('storage') == 'raw-evidence.zip'
                else normalized(CAPTURE/name))
        assert len(data) == entry['bytes'], name
        assert hashlib.sha256(data).hexdigest() == entry['sha256'], name
finally:
    if archive:
        archive.close()
for collection in ['sources', 'runtime_sources']:
    for name, digest in manifest[collection].items():
        assert hashlib.sha256(normalized(ROOT / name)).hexdigest() == digest, name
for name, digest in manifest['runtime_sources'].items():
    data = subprocess.check_output(['git', 'show', manifest['code_commit']+':'+name], cwd=ROOT)
    assert hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest() == digest, name
for mode in ['grounded', 'hybrid']:
    results = json.loads((CAPTURE / f'final-{mode}-results.json').read_text())
    assert len(results) == 5
    assert sum(row['checks'][0] for row in results) == 46
    assert all(row['checks'][0] == row['checks'][1] for row in results)
    pages = 0
    for row in results:
        stem = Path(row['document']).stem
        indices = json.loads((CAPTURE / f'final-{mode}-{stem}-page-indices.json').read_text())
        with fitz.open(ROOT / 'examples' / row['document']) as document:
            assert set(indices) == set(range(len(document)))
        pages += len(set(indices))
        if mode == 'grounded':
            assert not json.loads((CAPTURE / f'final-{mode}-{stem}-blocks.json').read_text())['page_errors']
    assert pages == 27
    for document in ['dense', 'notes']:
        checks = json.loads((CAPTURE / f'final-{mode}-{document}-export-checks.json').read_text())
        assert all(row['passed'] for row in checks), (mode, document)
tests = json.loads((CAPTURE / 'final-live-outcomes.json').read_text())
assert len(tests) == 28 and all(row['outcome'] == 'passed' for row in tests)
assert all(row['run'].startswith('live-complete-') for row in tests)
for mode in ['hybrid', 'grounded']:
    cases = [json.loads(line) for line in
             (CAPTURE / f'live-complete-{mode}-final-tests.jsonl').read_text().splitlines()]
    assert len(cases) == 28
    assert sum(row['outcome'] == 'passed' for row in cases) == 14
    assert sum(row['outcome'] == 'skipped' for row in cases) == 14
for name in ['selection-hybrid', 'selection-grounded', 'selection-text', 'web-hybrid', 'web-grounded']:
    assert json.loads((CAPTURE / f'{name}.json').read_text())['passed'], name
assert all(value['exit_code'] == 0 for value in
           json.loads((CAPTURE / 'final-workflow-status.json').read_text()).values())
print(f'Verified {len(manifest["files"])} archived files, committed runtime sources, '
      '54 recognized pages, 92 content checks and 28 distinct passing live cases')
