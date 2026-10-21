"""Independent exploratory stress stage; never writes primary results or core code."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import numpy as np
import cv2
cv2.setNumThreads(1)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from imaging_core import extract_feature
from matching_metrics import similarity_maps, summarize_map, NUMBA_AVAILABLE
from stress_model import perturb

STRATEGIES = ['raw', 'full_pc', 'down_pc', 'mfpc', 'mfpc_scale_matched']
K_LEVELS = [1, 4, 16]
SEEDS = list(range(30))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def fingerprint():
    paths = ['results/protocol.json', 'results/main_rows.jsonl', 'results/main_rows.signature.json',
             'results/controls_rows.jsonl', 'results/controls_rows.signature.json',
             'results/timing_rows.json', 'results/timing_rows.signature.json',
             'code/imaging_core.py', 'code/matching_metrics.py', 'code/run_benchmark.py']
    return {p: sha(ROOT / p) for p in paths}


def freeze():
    destination = HERE / 'protocol.json'
    if destination.exists():
        raise RuntimeError('Stress protocol already exists; do not overwrite it')
    source = json.loads((ROOT / 'results/protocol.json').read_text(encoding='utf-8'))
    for name, digest in source['source_code_sha256'].items():
        if sha(ROOT / 'code' / name) != digest:
            raise RuntimeError(f'Frozen core changed: {name}')
    pairs = [p for p in source['pairs'] if p['dataset'] == 'SAR' and p['historical']]
    assert len(pairs) == 10
    crop_signatures = {}
    for p in pairs:
        crop_signatures[p['pair_id']] = {suffix: sha(ROOT / 'results/crops' / (p['pair_id'] + suffix))
                                         for suffix in ('.npz', '.json')}
    protocol = dict(name='normalized-image multiplicative stress test',
                    interpretation='exploratory extra perturbation of observed normalized imagery; not physically calibrated speckle',
                    frozen_utc=datetime.now(timezone.utc).isoformat(), original_run_signature=source['run_signature'],
                    frozen_primary_files=fingerprint(), crop_signatures=crop_signatures,
                    supplemental_sources={n: sha(HERE / n) for n in ('stress_model.py', 'run_stress.py')},
                    pairs=pairs, offsets=source['offsets'], strategies=STRATEGIES,
                    gamma_shape_k=K_LEVELS, realizations=30, master_seed=2026101506,
                    perturbation='target_only: float32(clip(b*Gamma(k,scale=1/k),0,1)); no realization min-max or empirical mean correction',
                    sharing='same pair/k/seed image used across all strategies and offsets; independent between pair/k/seed',
                    grid_step=source['grid_step'], histogram_bins=source['histogram_bins'],
                    PC_parameters=source['PC_parameters'], expected_metric_rows=54600,
                    expected_perturbation_rows=910, max_workers=2, inferential_n=10,
                    bootstrap_replicates=10000, bootstrap_seed=2026101507,
                    python=sys.version, libraries={n: importlib.metadata.version(n) for n in ('numpy','scipy','numba','phasepack','PyWavelets')})
    protocol['run_signature'] = hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    dump(destination, protocol)
    print(json.dumps({'frozen': str(destination), 'run_signature': protocol['run_signature']}), flush=True)


def load_protocol():
    p = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
    if fingerprint() != p['frozen_primary_files']:
        raise RuntimeError('Frozen primary files changed')
    for name, digest in p['supplemental_sources'].items():
        if sha(HERE / name) != digest:
            raise RuntimeError(f'Stress source changed after freeze: {name}')
    if not NUMBA_AVAILABLE:
        raise RuntimeError('This experiment requires the compiled matching kernel')
    return p


def load_crop(pair, protocol):
    name = pair['pair_id']; cache = ROOT / 'results/crops'
    for suffix, digest in protocol['crop_signatures'][name].items():
        if sha(cache / (name + suffix)) != digest:
            raise RuntimeError('Frozen crop changed')
    with np.load(cache / (name + '.npz'), allow_pickle=False) as z:
        return z['a'].copy(), z['b'].copy()


def compare_feature_maps(pair, fa, fb, strategy, k, seed, protocol):
    W, T, S = pair['geometry']; ts = (W - T) // 2
    records = []
    for dr, dc in protocol['offsets']:
        sr, sc = (W - S) // 2 + dr, (W - S) // 2 + dc
        template = fa[ts:ts+T, ts:ts+T]; search = fb[sr:sr+S, sc:sc+S]
        maps = similarity_maps(template, search, step=protocol['grid_step'], bins=protocol['histogram_bins'],
                               template_range=(0, 1), search_range=(0, 1))
        for measure in ('MI','CCRE'):
            row = summarize_map(maps[measure], maps['rows'], maps['cols'], (ts-sr, ts-sc), (T,T))
            row.update(dataset='SAR', pair_id=pair['pair_id'], strategy=strategy, measure=measure,
                       condition='unperturbed' if k is None else f'gamma_k{k}', gamma_shape_k=k,
                       seed=None if k is None else seed, offset=[dr,dc], bins=protocol['histogram_bins'],
                       step=protocol['grid_step'], run_signature=protocol['run_signature'])
            records.append(row)
    return records


def evaluate_pair(payload):
    pair, protocol, pilot = payload
    t0 = time.perf_counter()
    a,b = load_crop(pair, protocol)
    features = {s: extract_feature(a, s, pc_kwargs=protocol['PC_parameters']) for s in protocol['strategies']}
    rows, diagnostics = [], []
    configurations = [(None, 0)] + ([(4,0)] if pilot else [(k,seed) for k in protocol['gamma_shape_k'] for seed in SEEDS])
    for k, seed in configurations:
        condition_start=time.perf_counter()
        stressed, info = perturb(b, k, pair['pair_id'], seed, protocol['master_seed'])
        info.update(pair_id=pair['pair_id'], condition='unperturbed' if k is None else f'gamma_k{k}',
                    gamma_shape_k=k, seed=None if k is None else seed)
        diagnostics.append(info)
        for strategy in protocol['strategies']:
            target = extract_feature(stressed, strategy, pc_kwargs=protocol['PC_parameters'])
            rows.extend(compare_feature_maps(pair, features[strategy], target, strategy, k, seed, protocol))
        info['condition_wall_s_diagnostic']=time.perf_counter()-condition_start
    result = dict(pair_id=pair['pair_id'], run_signature=protocol['run_signature'], pilot=pilot,
                  metric_rows=rows, perturbations=diagnostics, elapsed_s=time.perf_counter()-t0)
    destination = HERE / ('pilot.json' if pilot else f'parts/{pair["pair_id"]}.json')
    dump(destination, result)
    return dict(pair_id=pair['pair_id'], rows=len(rows), elapsed_s=result['elapsed_s'])


def baseline_check(rows):
    original = {}
    ids = {r['pair_id'] for r in rows}
    with (ROOT / 'results/main_rows.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['pair_id'] in ids and r['strategy'] in STRATEGIES:
                original[(r['pair_id'],r['strategy'],r['measure'],tuple(r['offset']))]=r
    mismatches=[]; checked=0
    fields=('err','success3','estimated_top_left','degenerate','peak_tie_count','map_range','baseline_corrected_peak_drop','PSR')
    for r in rows:
        if r['condition']!='unperturbed': continue
        o=original[(r['pair_id'],r['strategy'],r['measure'],tuple(r['offset']))]
        for field in fields:
            x,y=r.get(field),o.get(field)
            same = x==y if x is None or y is None or isinstance(x,bool) else np.allclose(x,y,rtol=1e-10,atol=1e-12)
            if not same: mismatches.append(dict(pair_id=r['pair_id'],strategy=r['strategy'],measure=r['measure'],offset=r['offset'],field=field,new=x,original=y))
        checked+=1
    return dict(checked_rows=checked, checked_fields=list(fields), mismatch_count=len(mismatches), mismatches=mismatches)


def run(pilot=False, workers=2):
    protocol=load_protocol()
    if pilot:
        info=evaluate_pair((protocol['pairs'][0],protocol,True))
        result=json.loads((HERE/'pilot.json').read_text(encoding='utf-8'))
        check=baseline_check(result['metric_rows']); dump(HERE/'pilot_baseline_check.json',check)
        if check['mismatch_count']: raise RuntimeError('Pilot baseline does not reproduce frozen main')
        info['pilot_based_serial_estimate_s']=result['perturbations'][-1]['condition_wall_s_diagnostic']*900+result['perturbations'][0]['condition_wall_s_diagnostic']*10
        info['timing_note']='diagnostic estimate only; one pair/one severity, excluding later I/O variation and contention'
        print(json.dumps(info),flush=True); return
    if not 1<=workers<=2: raise ValueError('At most two workers')
    pending=[]
    for pair in protocol['pairs']:
        f=HERE/'parts'/f'{pair["pair_id"]}.json'
        if f.exists():
            part=json.loads(f.read_text(encoding='utf-8'))
            if part['run_signature']!=protocol['run_signature'] or len(part['metric_rows'])!=5460 or len(part['perturbations'])!=91:
                raise RuntimeError('Invalid existing checkpoint')
        else: pending.append(pair)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(evaluate_pair,(p,protocol,False)) for p in pending]
        for f in as_completed(futures):print(json.dumps(f.result()),flush=True)
    all_rows=[]; diagnostics=[]
    for pair in protocol['pairs']:
        part=json.loads((HERE/'parts'/f'{pair["pair_id"]}.json').read_text(encoding='utf-8'))
        all_rows.extend(part['metric_rows']); diagnostics.extend(part['perturbations'])
    assert len(all_rows)==protocol['expected_metric_rows'] and len(diagnostics)==protocol['expected_perturbation_rows']
    check=baseline_check(all_rows); dump(HERE/'baseline_consistency.json',check)
    if check['mismatch_count']:raise RuntimeError('Unperturbed baseline differs from frozen results')
    load_protocol() # ensure primary inputs remained unchanged throughout the run
    out=HERE/'stress_rows.jsonl'; temp=out.with_name(out.name+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        for r in all_rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    os.replace(temp,out)
    dump(HERE/'perturbation_diagnostics.json',diagnostics)
    dump(HERE/'completion.json',dict(status='complete',run_signature=protocol['run_signature'],
         row_count=len(all_rows),result_sha256=sha(out),perturbation_row_count=len(diagnostics),
         perturbation_sha256=sha(HERE/'perturbation_diagnostics.json'),
         baseline_mismatches=check['mismatch_count'],frozen_primary_unchanged=True,
         completed_utc=datetime.now(timezone.utc).isoformat(),workers=workers))
    print(json.dumps({'status':'complete','rows':len(all_rows)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('freeze','pilot','run'))
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.mode=='freeze': freeze()
    else:run(pilot=args.mode=='pilot',workers=args.workers)
