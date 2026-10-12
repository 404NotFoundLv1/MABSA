"""One-time, read-only import from historical experiments; outputs only inside MACT."""
import hashlib,json,shutil,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent
EXPS=OLD/'mabsa_evidence/experiments'
SOURCE=EXPS/'20261012_sequence_crf_35ep_s20_31/group2'
records=[]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def copy_file(src,dst):
 src=Path(src);dst=Path(dst)
 assert dst.is_relative_to(ROOT)
 if dst.exists():raise FileExistsError(dst)
 dst.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(src,dst)
 a,b=sha(src),sha(dst)
 if a!=b:raise ValueError(f'copy mismatch {src}')
 if os.stat(src).st_ino==os.stat(dst).st_ino:raise ValueError('hard link forbidden')
 records.append(dict(source=str(src),destination=str(dst.relative_to(ROOT)),sha256=a))
def tree(src,dst,exclude=()):
 for p in sorted(Path(src).rglob('*')):
  if p.is_file() and not any(x in p.parts for x in exclude):copy_file(p,Path(dst)/p.relative_to(src))
tree(SOURCE/'source/mabsa_sequence',ROOT/'src/mact',('__pycache__',))
for p in (SOURCE/'tests').glob('*.py'):
 (ROOT/'tests'/p.name).write_text(p.read_text().replace('mabsa_sequence','mact'))
for name,src in [('deberta-v3-base',OLD/'deqa_reproduction/models/deberta-v3-base'),('siglip2',EXPS/'20261004_backbone_2x2/models/siglip2')]:
 tree(src,ROOT/'assets/models'/name,('.cache','.git'))
 print('copied model',name,flush=True)
for ds in ('twitter2015','twitter2017'):
 tree(OLD/'IJCAI2019_data'/ds,ROOT/'assets/data'/ds)
 tree(OLD/f'{ds}_images',ROOT/'assets/images'/ds)
 for split,src in [('train_dev',EXPS/'20261004_backbone_2x2/cache'/f'{ds}-siglip2'),('test',EXPS/'20261011_sequence_crf/cache'/f'{ds}-test-siglip2')]:
  target=ROOT/'assets/cache'/ds/split
  tree(src,target)
  # Retain original manifests as provenance, change only path strings in active copy.
  original=target/'manifest.json';meta=json.loads(original.read_text())
  (target/'manifest.original.json').write_bytes(original.read_bytes())
  meta['vision_model']=os.path.relpath(ROOT/'assets/models/siglip2',target)
  meta['image_root']=os.path.relpath(ROOT/'assets/images'/ds,target)
  original.write_text(json.dumps(meta,indent=2)+'\n')
 print('copied official data, images, caches',ds,flush=True)
allrows=json.loads((EXPS/'20261012_sequence_crf_35ep_s20_31/all_21_seed_results.json').read_text())
copy_file(EXPS/'20261012_sequence_crf_35ep_s20_31/all_21_seed_results.json',ROOT/'reference/all_21_seed_results.original.json')
for row in allrows['rows']:
 src=Path(row['run']);target=ROOT/'reference/runs'/row['dataset']/f"seed{row['seed']}"
 for p in sorted(src.rglob('*')):
  if p.is_file() and (p.suffix!='.pt' or (p.name=='best.pt' and row['seed'] in (24,25))):copy_file(p,target/p.relative_to(src))
 row['original_run']=row['run'];row['run']=str(target.relative_to(ROOT))
(ROOT/'reference/all_21_seed_results.json').write_text(json.dumps(allrows,indent=2)+'\n')
fig=OLD/'mabsa_evidence/figures/current_sequence_fusion/architecture_paper_no_title.png'
if fig.exists():copy_file(fig,ROOT/'docs/architecture.png')
(ROOT/'reference/import_manifest.json').write_text(json.dumps(dict(source=str(SOURCE),files=records,note='Original files are unchanged. Active cache manifests relocate identity paths only; original manifests retained.'),indent=2)+'\n')
print('Import verified:',len(records),'files',flush=True)
