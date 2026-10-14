"""Verified, label-blind OCR text assets. Recognition runs offline, outside training."""
import json
import math
import re
from pathlib import Path

from .common.cache import file_hash

POLICY = dict(min_confidence=.8, require_ascii_letters=2,
              whitespace='collapse', order='engine', manual_corrections=False)


def filter_lines(lines):
    kept=[]
    for line in lines:
        text,score=line['text'],line['confidence']
        if (not isinstance(text,str) or type(score) not in (float,int)
                or not math.isfinite(score) or not 0<=score<=1):
            raise ValueError('invalid OCR line')
        text=' '.join(text.split())
        if score>=POLICY['min_confidence'] and re.search(r'[A-Za-z]{2}',text):
            kept.append(text)
    return ' '.join(kept)


def open_ocr(cache, image_root, examples, expected_manifest_sha256=None):
    cache=Path(cache); image_root=Path(image_root).resolve()
    digest=file_hash(cache/'manifest.json')
    if expected_manifest_sha256 is not None and digest!=expected_manifest_sha256:
        raise ValueError('OCR manifest changed since training/config resolution')
    meta=json.loads((cache/'manifest.json').read_text())
    if meta.get('format')!='mact_ocr_v1' or file_hash(cache/'texts.json')!=meta.get('texts_sha256'):
        raise ValueError('invalid OCR asset format or text hash')
    rows=json.loads((cache/'texts.json').read_text());texts={}
    for image_id in sorted({ex.image_id for ex in examples}):
        if image_id not in rows:
            raise ValueError(f'missing OCR image: {image_id}; generate the correct split cache first')
        image=(image_root/image_id).resolve();row=rows[image_id]
        if (not image.is_relative_to(image_root) or not image.is_file()
                or file_hash(image)!=row.get('image_sha256') or not isinstance(row.get('text'),str)):
            raise ValueError(f'OCR image identity/text mismatch: {image_id}')
        texts[image_id]=row['text']
    return texts,dict(manifest_sha256=digest,metadata=meta,images=len(texts),
                      nonempty=sum(bool(t) for t in texts.values()))


def export_probe(probe, dataset, output):
    """Promote frozen automatic OCR only; never read quality notes or annotations."""
    probe=Path(probe);output=Path(output)
    if dataset not in ('twitter2015','twitter2017'):raise ValueError('unknown dataset')
    if output.exists():raise FileExistsError(output)
    status=json.loads((probe/'status.json').read_text())
    manifest=json.loads((probe/'manifest.json').read_text())
    if status['state']!='completed' or status['failed'] or status['manifest_sha256']!=file_hash(probe/'manifest.json'):
        raise ValueError('incomplete or changed OCR probe')
    for name,digest in manifest['models_sha256'].items():
        if file_hash(probe/'models'/name)!=digest:raise ValueError('OCR model changed')
    images={r['image_sha256']:r['image_path'] for r in manifest['images']}
    rows={};sources={}
    for r in manifest['records']:
        if r['dataset']!=dataset:continue
        if r['split'] not in ('train','dev'):raise ValueError('unexpected split in probe')
        key=r['image_sha256'];path=probe/'cache'/f'{key}.json'
        result=json.loads(path.read_text())
        if (result['state']!='completed' or result['image_sha256']!=key
                or result['config_sha256']!=status['config_sha256'] or file_hash(images[key])!=key):
            raise ValueError('OCR cache/image provenance mismatch')
        row=dict(image_sha256=key,text=filter_lines(result['lines']))
        if r['image_id'] in rows and rows[r['image_id']]!=row:raise ValueError('conflicting OCR image ID')
        rows[r['image_id']]=row;sources[key]=file_hash(path)
    if not rows:raise ValueError('empty OCR export')
    output.mkdir(parents=True)
    def save(name,value):(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    save('texts.json',rows)
    save('manifest.json',dict(format='mact_ocr_v1',dataset=dataset,policy=POLICY,
        texts_sha256=file_hash(output/'texts.json'),source_manifest_sha256=status['manifest_sha256'],
        recognition_config_sha256=status['config_sha256'],engine_versions=status['engine_versions'],
        models_sha256=manifest['models_sha256'],source_cache_sha256=sources,
        splits=['train','dev'],label_fields_used=False))
