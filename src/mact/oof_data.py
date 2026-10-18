"""Label-independent grouped outer/inner splits for OOF correction features."""
import hashlib
import json
import math
from pathlib import Path

from .common.cache import file_hash
from .data import load_labeled
from .experiment import ROOT
from .runtime import write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def make_partition(examples, image_hashes):
    ids = sorted(e.id for e in examples)
    if len(set(ids)) != len(ids): raise ValueError('duplicate sentence IDs')
    parent = {sid: sid for sid in ids}
    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid
    seen = {}
    for e in sorted(examples, key=lambda e: e.id):
        for key in [('text', tuple(w.casefold() for w in e.words)), ('image', image_hashes[e.image_id])]:
            if key in seen:
                a, b = find(e.id), find(seen[key])
                parent[max(a, b)] = min(a, b)
            seen[key] = e.id
    groups = {}
    for sid in ids: groups.setdefault(find(sid), []).append(sid)
    members = {digest(v): v for v in groups.values()}
    if len(members) < 6: raise ValueError('too few independent groups for three outer and inner splits')
    assignment, sizes = {}, [0, 0, 0]
    for g in sorted(members, key=lambda g: (-len(members[g]), digest(['outer42', g]))):
        fold = min(range(3), key=lambda f: (sizes[f], f))
        assignment[g] = fold
        sizes[fold] += len(members[g])
    mapping = {sid: g for g, values in members.items() for sid in values}
    folds = []
    for fold in range(3):
        available = sorted((g for g in members if assignment[g] != fold), key=lambda g: digest(['inner42', fold, g]))
        count = max(1, math.ceil(.1*len(available)))
        if count >= len(available): raise ValueError('empty inner fit split')
        inner = set(available[:count])
        folds.append(dict(fold=fold,
            fit_ids=[s for s in ids if assignment[mapping[s]] != fold and mapping[s] not in inner],
            inner_ids=[s for s in ids if mapping[s] in inner],
            held_ids=[s for s in ids if assignment[mapping[s]] == fold]))
    return dict(format='grouped_oof_partition_v1', seed=42, n_folds=3, groups=mapping, folds=folds)


def verify_partition(examples, partition, image_hashes):
    if partition != make_partition(examples, image_hashes):
        raise ValueError('partition differs from deterministic grouped membership')


def prepare_dataset(dataset, root):
    if dataset not in ('twitter2015', 'twitter2017'): raise ValueError('unsupported dataset')
    directory = Path(root)/dataset/'splits'
    if directory.exists(): raise FileExistsError('split destination must be fresh')
    source = ROOT/'assets/data'/dataset/'train.json'
    examples, _ = load_labeled(source)
    raw = json.loads(source.read_text())
    records = {e.id: dict(r, id=e.id) for e, r in zip(examples, raw)}
    manifest = ROOT/'assets/cache'/dataset/'train_dev/manifest.json'
    images = json.loads(manifest.read_text())['image_sha256']
    for e in examples:
        if file_hash(ROOT/'assets/images'/dataset/e.image_id) != images[e.image_id]:
            raise ValueError('stale image identity')
    partition = make_partition(examples, images)
    directory.mkdir(parents=True)
    write_json(directory/'partition.json', partition)
    for fold in partition['folds']:
        target = directory/f'fold{fold["fold"]}'
        target.mkdir()
        for role in ('fit', 'inner', 'held'):
            path = target/(role+'.json')
            write_json(path, [records[sid] for sid in fold[role+'_ids']])
            loaded, _ = load_labeled(path)
            if [e.id for e in loaded] != fold[role+'_ids']: raise ValueError('materialized IDs changed')
    hashes = {str(p.resolve()): file_hash(p) for p in directory.rglob('*.json')}
    hashes.update({str(source): file_hash(source), str(manifest): file_hash(manifest)})
    write_json(directory/'manifest.json', dict(dataset=dataset, seed=42, input_sha256=hashes,
        sentences=len(examples), groups=len(set(partition['groups'].values())),
        counts=[{role: len(f[role+'_ids']) for role in ('fit', 'inner', 'held')} for f in partition['folds']],
        test_used=False))
    return directory


def verify_splits(directory):
    directory = Path(directory)
    meta = json.loads((directory/'manifest.json').read_text())
    for p, sha in meta['input_sha256'].items():
        if file_hash(p) != sha: raise ValueError('stale split manifest: '+p)
    dataset = meta['dataset']
    examples, _ = load_labeled(ROOT/'assets/data'/dataset/'train.json')
    images = json.loads((ROOT/'assets/cache'/dataset/'train_dev/manifest.json').read_text())['image_sha256']
    partition = json.loads((directory/'partition.json').read_text())
    verify_partition(examples, partition, images)
    original = {e.id: e for e in examples}
    for fold in partition['folds']:
        for role in ('fit', 'inner', 'held'):
            values, _ = load_labeled(directory/f'fold{fold["fold"]}'/(role+'.json'))
            if [e.id for e in values] != fold[role+'_ids'] or any(e != original[e.id] for e in values):
                raise ValueError('materialized fold changed original examples')
    return partition, meta
