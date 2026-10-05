"""Create and verify the final submission without running any model."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / 'submissions/02599_LuuQuangKhai'
OUT = SUB.parent / '02599_LuuQuangKhai_submission.zip'

def digest(data):
    return hashlib.sha256(data).hexdigest()

files = []
for p in sorted(SUB.rglob('*')):
    if not p.is_file():
        continue
    rel = p.relative_to(SUB)
    if any(part in {'__pycache__', 'node_modules', '.git', 'stage2_cpu_smoke'} for part in rel.parts):
        continue
    if p.suffix.lower() in {'.pyc', '.pt', '.pth', '.tmp', '.zip'} or p.name == 'submission_manifest.json':
        continue
    if rel.parts[:2] == ('eda', 'local'):
        continue
    files.append(p)
records = [{'path': p.relative_to(SUB).as_posix(), 'bytes': p.stat().st_size,
            'sha256': digest(p.read_bytes())} for p in files]
manifest = SUB / 'submission_manifest.json'
manifest.write_text(json.dumps({'schema_version': 1, 'submission': SUB.name,
    'hash': 'SHA256', 'manifest_excludes_itself': True, 'files': records},
    ensure_ascii=False, indent=2), encoding='utf-8')
with zipfile.ZipFile(OUT, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in files + [manifest]:
        z.write(p, f'{SUB.name}/{p.relative_to(SUB).as_posix()}')
with zipfile.ZipFile(OUT) as z:
    assert z.testzip() is None, 'CRC error'
    expected = {f'{SUB.name}/{r["path"]}' for r in records} | {f'{SUB.name}/submission_manifest.json'}
    assert set(z.namelist()) == expected
    for r in records:
        data = z.read(f'{SUB.name}/{r["path"]}')
        assert len(data) == r['bytes'] and digest(data) == r['sha256'], r['path']
    assert z.read(f'{SUB.name}/submission_manifest.json') == manifest.read_bytes()
result = {'status': 'PASS', 'zip': OUT.name, 'files_including_manifest': len(records)+1,
          'bytes': OUT.stat().st_size, 'sha256': digest(OUT.read_bytes()),
          'crc_verified': True, 'all_packaged_files_sha256_verified': True}
OUT.with_suffix('.audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
OUT.with_suffix('.sha256.txt').write_text(f'{result["sha256"]}  {OUT.name}\n', encoding='utf-8')
print(json.dumps(result, indent=2))
