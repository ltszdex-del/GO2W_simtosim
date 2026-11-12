"""Verify published policy weights without loading pickle or requiring Torch."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'docs/POLICY_MANIFEST.json').read_text(encoding='utf-8'))
for mode, spec in manifest['policies'].items():
    path = root / spec['checkpoint']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec['sha256']:
        raise SystemExit('SHA256 mismatch: ' + mode)
    print('OK', mode, path.name)
print('All published policy checksums match.')
