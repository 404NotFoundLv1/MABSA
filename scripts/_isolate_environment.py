"""Offline copy of CPython runtime and effective distributions; never edits sources."""
import hashlib,importlib.metadata as im,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'.runtime'
BASE=Path(sys.base_prefix)
if TARGET.exists():raise FileExistsError(TARGET)
TARGET.mkdir();(TARGET/'bin').mkdir()
shutil.copy2(sys._base_executable,TARGET/'bin/python3.11')
(TARGET/'bin/python').symlink_to('python3.11');(TARGET/'bin/python3').symlink_to('python3.11')
shutil.copytree(BASE/'lib',TARGET/'lib',ignore=shutil.ignore_patterns('site-packages','__pycache__'),symlinks=False)
site=TARGET/'lib/python3.11/site-packages';site.mkdir()
seen={}
for d in im.distributions():
 name=d.metadata['Name'].lower().replace('_','-')
 seen.setdefault(name,d)
report=[];owned={}
for name,d in sorted(seen.items()):
 if d.files is None:raise ValueError(f'missing package RECORD: {name}')
 copied=0
 for f in d.files:
  rel=Path(str(f))
  if '..' in rel.parts or rel.is_absolute() or rel.suffix=='.pyc':continue
  src=Path(d.locate_file(f));dst=site/rel
  if not src.is_file():continue
  if dst.exists():
   if src.read_bytes()!=dst.read_bytes():raise ValueError(f'conflicting package files: {rel}')
   continue
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);copied+=1
 report.append(dict(name=name,version=d.version,source=str(d.locate_file('')),copied_files=copied))
 print(name,d.version,copied,flush=True)
(ROOT/'requirements.lock.txt').write_text('\n'.join(f'{n}=={d.version}' for n,d in sorted(seen.items()))+'\n')
(ROOT/'reference/environment-import.json').write_text(json.dumps(dict(python=sys.version,base_source=str(BASE),source_prefix=sys.prefix,packages=report),indent=2)+'\n')
print('Isolated runtime copied.',flush=True)
