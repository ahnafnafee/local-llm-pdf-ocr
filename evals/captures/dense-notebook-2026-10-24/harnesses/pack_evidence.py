"""Package already audited captures without losing per-file verification."""
import hashlib
import json
from pathlib import Path
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DEST = (ROOT/'evals/captures/dense-notebook-2026-10-24').resolve()
assert DEST.is_relative_to(ROOT.resolve()) and DEST.name == 'dense-notebook-2026-10-24'
manifest_path = DEST/'manifest.json'
manifest = json.loads(manifest_path.read_text())
keep = {
    'before-after-comparison.json', 'final-workflow-status.json', 'final-live-outcomes.json',
    'final-suite-start.json', 'search-rectangle-observations.json',
    'publication-audit.log', 'expanded-publication-audit.log', 'model-server-stall.log',
    'offline-final-v18.log', 'build.log', 'package-verification.log',
}
for mode in ['grounded','hybrid']:
    keep.update([f'final-{mode}-results.json', f'final-{mode}.log',
                 f'live-complete-{mode}-final-tests.jsonl', f'live-complete-{mode}-final.log',
                 f'web-{mode}.json', f'image-{mode}.json'])
    for doc in ['dense','notes']:
        keep.add(f'final-{mode}-{doc}-export-checks.json')
    for doc in ['digital','hybrid','handwritten','dense','notes']:
        keep.add(f'final-{mode}-{doc}-page-indices.json')
        if mode == 'grounded':
            keep.add(f'final-{mode}-{doc}-blocks.json')
for mode in ['grounded','hybrid','text']:
    keep.add(f'selection-{mode}.json')
archive_path = DEST/'raw-evidence.zip'
packed = []
with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for name, entry in manifest['files'].items():
        path = (DEST/name).resolve()
        assert path.is_relative_to(DEST) and path.is_file(), name
        data = path.read_bytes()
        if path.suffix not in {'.pdf','.png','.avif'}:
            data = data.replace(b'\r\n', b'\n')
        assert hashlib.sha256(data).hexdigest() == entry['sha256'], name
        if name in keep or name.startswith('harnesses/') or path.suffix == '.png':
            entry['storage'] = 'loose'
            continue
        info = zipfile.ZipInfo(name, date_time=(2026,10,24,0,0,0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, data, compresslevel=9)
        entry['storage'] = 'raw-evidence.zip'
        packed.append(name)
# Verify every member before removing any redundant repository copy.
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    assert set(archive.namelist()) == set(packed)
    for name in packed:
        assert hashlib.sha256(archive.read(name)).hexdigest() == manifest['files'][name]['sha256']
for name in packed:
    path = (DEST/name).resolve()
    assert path.is_relative_to(DEST) and path.is_file()
    path.unlink()
manifest['raw_archive'] = {'path': archive_path.name, 'members': len(packed),
    'bytes': archive_path.stat().st_size, 'sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    'note': 'Expanded contents were audited before packing; every member retains its own hash in files.'}
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Packed {len(packed)} captures into {archive_path.stat().st_size} bytes; '
      f'{len(manifest["files"])-len(packed)} evidence files remain directly readable')
