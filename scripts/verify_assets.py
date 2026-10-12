"""Verify local immutable resources against their migration SHA256 inventory."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

if __name__=='__main__':
    manifest=json.loads((ROOT/'reference/assets-manifest.json').read_text())
    for name,sha in manifest['sha256'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or digest(path)!=sha:
            raise ValueError(f'asset missing or changed: {name}')
    print(f"Verified {len(manifest['sha256'])} local asset/archive files.")
