from pathlib import Path
import json

out = Path(__file__).resolve().parent
for mode in ['grounded', 'hybrid']:
    for path in out.glob(f'final-{mode}*'):
        target = path.with_name(path.name.replace('final-', 'before-row-anchors-', 1))
        assert target.parent.resolve() == out and not target.exists()
        path.rename(target)
for path in out.glob('final-export-*'):
    target = path.with_name(path.name.replace('final-', 'before-row-anchors-', 1))
    assert target.parent.resolve() == out and not target.exists()
    path.rename(target)
record = out/'final-workflow-status.json'
status = json.loads(record.read_text())
(out/'before-row-anchors-status.json').write_text(json.dumps({
    'reason': 'The seeded hybrid PDF placed reachable above its source line. Grounded corpus completed; hybrid corpus was intentionally interrupted for the ordered-row correction. It is not a timeout or a final validation run.',
    'runtime_code_before_correction': 'c5ba20355eba3da2e92aa8673d48ad5a2322fa71',
    'status': status,
}, indent=2)+'\n')
record.write_text(json.dumps({k:v for k,v in status.items()
    if k not in ['final-grounded','final-hybrid'] and not k.startswith('final-export-')}, indent=2)+'\n')
(out/'before-row-anchors-suite-start.json').write_bytes((out/'final-suite-start.json').read_bytes())
print('Retained the completed grounded run and interrupted hybrid candidate separately')
