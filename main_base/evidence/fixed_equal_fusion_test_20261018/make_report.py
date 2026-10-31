"""Render the completed fixed-policy Test comparison, without selecting anything."""
import csv
import json
from collections import Counter
from pathlib import Path
from mact.common.cache import file_hash
from mact.runtime import write_json

root=Path(__file__).resolve().parent
report=json.loads((root/'comparison.json').read_text())
lock=json.loads((root/'decision.json').read_text())
assert report['decision_sha256']==file_hash(root/'decision.json')
assert report['parameters']=={'temperature_base':1.,'temperature_target':1.,'alpha':.5}
assert report['completed_pairs']==30 and not report['test_used_for_selection']
lines=['固定 1:1 概率融合：Test 配对结果','',
       '两个温度均为1，目标分类器权重0.5；所有模型和参数在本轮 Test 前锁定。',
       '均值、样本标准差和差值均以 F1 百分点表示。',
       '历史 seed 和此前 Test 结果已知，因此不能声称来自未接触过的基准。','']
csv_rows=[]
for dataset,cohorts in report['datasets'].items():
    for name,c in cohorts.items():
        lines += [dataset+' / '+name,'方案 | 平均 F1 | 样本标准差']
        for method,label in [('baseline','原 MACT'),('target','目标上下文候选'),('fusion','固定等权融合')]:
            s=c[method];lines.append(f"{label} | {s['mean']:.4f} | {s['sample_std']:.4f}")
        for control,label in [('baseline','原 MACT'),('target','当前候选')]:
            s=c['delta_vs_'+control]
            lines.append(f"相对{label}: {s['mean']:+.4f}; {s['wins']}升/{s['ties']}平/{s['losses']}降")
        lines.append('')
    lines += ['逐 seed：','seed | 原 MACT | 当前候选 | 固定融合 | 相对原 MACT | 相对当前候选']
    for r in report['rows']:
        if r['dataset'] != dataset:continue
        db,dt=r['fusion']-r['baseline'],r['fusion']-r['target']
        csv_rows.append(dict(**r,delta_vs_baseline=db,delta_vs_target=dt))
        lines.append(f"{r['seed']} | {r['baseline']:.4f} | {r['target']:.4f} | {r['fusion']:.4f} | {db:+.4f} | {dt:+.4f}")
    lines.append('')
(root/'SUMMARY.txt').write_text('\n'.join(lines)+'\n')
with (root/'paired_results.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
verified=[]
for selected in lock['models']:
    ds,s=selected['dataset'],selected['seed']
    path=root/ds/f'seed{s}'/'result.json'
    assert file_hash(path)==path.with_suffix('.sha256').read_text().strip()
    r=json.loads(path.read_text());p=r['provenance'];hashes=p['input_sha256']
    assert hashes[selected['baseline_checkpoint']]==selected['baseline_sha256']
    assert hashes[selected['target_checkpoint']]==selected['target_sha256']
    assert p['decision_sha256']==report['decision_sha256']
    assert p['baseline_replay_exact'] and p['target_replay_exact'] and p['gold_removed_from_inference']
    assert not p['test_used_for_selection'] and p['full_test'] and p['boundaries_unchanged']
    for filename in ('manifest.json','features.pt'):
        assert any(str(Path('assets/cache')/ds/'test'/filename) in name for name in hashes)
    assert all(x['mate']==r['metrics']['baseline']['mate'] for x in r['metrics'].values())
    verified.append(dict(dataset=ds,seed=s,controls_exact=True,boundaries_unchanged=True,
                         effective_visual_cache_fingerprinted=True))
write_json(root/'postrun-audit.json',dict(verified_pairs=verified,count=len(verified),
    summary_sha256=file_hash(root/'comparison.json'),decision_sha256=file_hash(root/'decision.json'),
    parameters_unchanged=True,test_used_for_selection=False))
write_json(root/'status.json',dict(state='completed',verified_pairs=len(verified),
    test_evaluated=True,parameters_unchanged=True,test_used_for_selection=False,
    report='SUMMARY.txt',audit='postrun-audit.json'))
print('\n'.join(lines))
