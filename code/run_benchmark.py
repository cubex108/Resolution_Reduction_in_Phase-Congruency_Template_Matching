"""Resumable v3 experiment with an explicit freeze boundary.

prepare -> freeze (in the production Python environment) -> main/controls/timing.
Only the separate serial timing run is a runtime benchmark. Main-run wall times
are diagnostics under concurrent execution. The evaluation is a new experiment,
not a numerical patch to the old manuscript tables.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(1)
from imaging_core import read_image, extract_feature, PC_DEFAULTS
from matching_metrics import similarity_maps, summarize_map, NUMBA_AVAILABLE

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'Code_论文' / 'func' / 'data'
RESULT = ROOT / 'results'
CACHE = RESULT / 'crops'
OFFSETS = [(0, 0), (5, 0), (0, 5), (10, 10), (-20, 15), (40, -30)]
STRATEGIES = ['raw', 'full_pc', 'down_pc', 'mfpc', 'mfpc_scale_matched', 'no_detail', 'gated']
SEED = 20261015
PROTOCOL_SCHEMA = 2
EXECUTION_SOURCES = ('run_benchmark.py', 'imaging_core.py', 'matching_metrics.py')
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def dump(path, obj):
    """A completed output must replace its destination atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def source_hashes():
    return {name: sha(Path(__file__).parent / name) for name in EXECUTION_SOURCES}


def runtime_fingerprint():
    distributions = {'numpy': 'numpy', 'scipy': 'scipy', 'pywt': 'PyWavelets',
                     'tifffile': 'tifffile', 'phasepack': 'phasepack',
                     'numba': 'numba', 'llvmlite': 'llvmlite', 'Pillow': 'Pillow'}
    versions = {}
    for key, distribution in distributions.items():
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    versions['opencv'] = cv2.__version__
    result = {'python_executable': str(Path(sys.executable).resolve()), 'python': sys.version,
              'platform': platform.platform(), 'machine': platform.machine(),
              'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
              'libraries': versions, 'thread_environment': {k: os.environ.get(k) for k in THREAD_KEYS},
              'opencv_threads': cv2.getNumThreads(), 'numba_available': bool(NUMBA_AVAILABLE)}
    try:
        import psutil
        result['total_memory_bytes'] = int(psutil.virtual_memory().total)
    except ImportError:
        result['total_memory_bytes'] = None
    return result


def build_manifest():
    original = [1, 5, 10, 15, 20, 25, 30, 35, 40, 41]
    opt = DATA / 'QXSLAB_SAROPT' / 'QXSLAB_SAROPT' / 'opt_256_oc_0.2'
    sar = opt.parent / 'sar_256_oc_0.2'
    candidates = sorted(int(p.stem) for p in opt.glob('*.png') if (sar / p.name).is_file())
    if not set(original).issubset(candidates):
        raise RuntimeError('At least one historical SAR pair is missing')
    pool = [i for i in candidates if i not in original]
    if len(pool) < 90:
        raise RuntimeError('Fewer than 90 additional matched SAR patches are available')
    selected = original + sorted(np.random.default_rng(SEED).choice(pool, 90, replace=False).tolist())
    pairs = []
    for i in selected:
        pairs.append(dict(dataset='SAR', pair_id=f'SAR_{i:05d}', a=str((opt / f'{i}.png').relative_to(DATA)),
                          b=str((sar / f'{i}.png').relative_to(DATA)), a_mode='gray', b_mode='gray',
                          geometry=[240, 48, 160], historical=i in original))
    for ds, bd, ad, suffix, bmode in [('Label', 'Lable_RGB', 'RGB_Lable', 'label', 'label'),
                                    ('NIR', 'IR_RGB', 'RGB_IR', 'RGBIR', 'nir')]:
        for p in sorted((DATA / bd).glob('*.tif')):
            stem = p.stem.removesuffix('_' + suffix)
            a = DATA / ad / (stem + '_RGB.tif')
            if not a.is_file():
                raise FileNotFoundError(a)
            pairs.append(dict(dataset=ds, pair_id=ds + '_' + stem.replace('top_potsdam_', ''),
                              a=str(a.relative_to(DATA)), b=str(p.relative_to(DATA)), a_mode='rgb',
                              b_mode=bmode, geometry=[400, 64, 240], historical=True))
    if len({p['pair_id'] for p in pairs}) != len(pairs):
        raise RuntimeError('Duplicate pair_id')
    counts = {d: sum(p['dataset'] == d for p in pairs) for d in ('SAR', 'Label', 'NIR')}
    if counts != {'SAR': 100, 'Label': 12, 'NIR': 10}:
        raise RuntimeError(f'Expected the declared 100+12+10 pair selection, got {counts}')
    return pairs, len(candidates)


def protocol_definition(pairs, available):
    return dict(schema_version=PROTOCOL_SCHEMA, selection_seed=SEED,
                sar_available_matched_patches=available,
                sar_selection='10 historical + 90 uniform without replacement from remainder',
                potsdam_selection='All locally available paired tiles; not randomly sampled from full archive',
                inferential_unit='pair_id; offsets and random realizations are repeated measures; SAR parent-scene provenance unavailable',
                offsets=[list(x) for x in OFFSETS], strategies=list(STRATEGIES), grid_step=2, histogram_bins=32,
                quantization='fixed [0,1] for both work-image features; no per-candidate normalization',
                raw_scaling='read_image normalizes each complete source image; raw work crop is not rescaled again',
                feature_context='once on complete work crop; central optical template; shifted target search windows',
                truth='dataset co-registration; no independently surveyed point truth', PC_parameters=PC_DEFAULTS,
                random_controls=dict(pairs='historical 10 SAR and all 22 Potsdam pairs', offset=[10, 10],
                                     strategies=['noise', 'surrogate'], realizations=30, detail_scales=[0.5, 1.0, 2.0],
                                     independence='SHA256 streams separate modalities, pairs, subbands and seeds; scales intentionally reuse paired random draws'),
                deterministic_controls=dict(offset=[10, 10], pairs='same 32 original pairs',
                                            wavelets=['haar', 'db2', 'db3'], levels=[1, 2],
                                            wavelet_strategies=['mfpc', 'mfpc_scale_matched'],
                                            detail_scales=[0, 0.25, 0.5, 1, 2], histogram_bins=[16, 32, 64],
                                            extra_strategies=['lowfreq', 'all_subbands', 'denoise']),
                timing=dict(repeats=30, warmup=3, threads=1,
                            estimand='serial in-memory two-image feature extraction + joint MI/CCRE map + joint peak/metric decoding, one offset',
                            excludes=['source image reading/decoding', 'disk I/O', 'crop loading', 'JIT compilation'],
                            order='randomized strategy order per repeat',
                            representative_pairs='first manifest pair per dataset; technical repeats are not additional image pairs',
                            main_wall_times='diagnostic only under concurrent execution, not a publication runtime benchmark',
                            feature_accounting='features are shared across offsets; each main row charges the full feature cost for a hypothetical isolated one-offset query; never sum those row costs across measures/offsets'),
                external_baselines=dict(HOPC_original='not_run', RIFT_author_implementation='not_run',
                                        pc_orientation_channels='available exploratory feature; not original HOPC and not in primary strategies'),
                pairs=pairs)


def _validate_crop(p, verify_signature=None):
    target = CACHE / (p['pair_id'] + '.npz')
    metadata = CACHE / (p['pair_id'] + '.json')
    if not target.is_file() or not metadata.is_file():
        raise RuntimeError(f'Incomplete crop cache for {p["pair_id"]}; finish prepare before freeze')
    infos = json.loads(metadata.read_text(encoding='utf-8'))
    if not isinstance(infos, list) or len(infos) != 2:
        raise RuntimeError(f'Invalid crop metadata for {p["pair_id"]}')
    with np.load(target, allow_pickle=False) as cache:
        for index, key in enumerate(('a', 'b')):
            arr = cache[key]
            if arr.shape != (p['geometry'][0],) * 2 or arr.dtype != np.float32:
                raise RuntimeError(f'Wrong crop shape/dtype for {p["pair_id"]}/{key}')
            if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
                raise RuntimeError(f'Invalid crop values for {p["pair_id"]}/{key}')
            if hashlib.sha256(arr.tobytes()).hexdigest() != infos[index]['crop_sha256']:
                raise RuntimeError(f'Crop array checksum mismatch for {p["pair_id"]}/{key}')
            if infos[index]['source_relative_path'] != p[key]:
                raise RuntimeError(f'Crop source path mismatch for {p["pair_id"]}/{key}')
    record = dict(pair_id=p['pair_id'], npz_sha256=sha(target), metadata_sha256=sha(metadata),
                  source_sha256=[x['source_sha256'] for x in infos])
    if verify_signature is not None and record != verify_signature:
        raise RuntimeError(f'Frozen input signature mismatch for {p["pair_id"]}')
    return record


def prepare():
    RESULT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    pairs, available = build_manifest()
    pfile = RESULT / 'protocol.json'
    if pfile.exists():
        old = json.loads(pfile.read_text(encoding='utf-8'))
        if old['pairs'] != pairs:
            raise RuntimeError('Pair selection changed; use a separate experiment directory')
        if old.get('run_signature'):
            load_frozen_protocol()
            print('Already frozen and validated; prepare made no changes', flush=True)
            return
    else:
        prepared = protocol_definition(pairs, available)
        prepared.update(status='Prepared input selection only; not frozen and not preregistered',
                        prepared_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        dump(pfile, prepared)
    for index, p in enumerate(pairs):
        target = CACHE / (p['pair_id'] + '.npz')
        metadata = CACHE / (p['pair_id'] + '.json')
        if target.exists() and metadata.exists():
            _validate_crop(p)
            continue
        crops, infos = [], []
        W = p['geometry'][0]
        for key, mode in [('a', p['a_mode']), ('b', p['b_mode'])]:
            image, info = read_image(DATA / p[key], mode, return_info=True)
            cy, cx = np.array(image.shape) // 2
            crop = image[cy - W // 2:cy + W // 2, cx - W // 2:cx + W // 2].copy()
            if crop.shape != (W, W):
                raise ValueError(f'Source too small for work crop: {p["pair_id"]}')
            info.pop('source_path', None)
            info['source_relative_path'] = p[key]
            info['source_sha256'] = sha(DATA / p[key])
            info['crop_top_left'] = [int(cy - W // 2), int(cx - W // 2)]
            info['crop_sha256'] = hashlib.sha256(crop.tobytes()).hexdigest()
            crops.append(crop)
            infos.append(info)
            del image
        temporary = target.with_name(target.name + f'.tmp.{os.getpid()}')
        with temporary.open('wb') as file:
            np.savez_compressed(file, a=crops[0], b=crops[1])
        os.replace(temporary, target)
        dump(metadata, infos)
        print(f'cached {index + 1}/{len(pairs)} {p["pair_id"]}', flush=True)


def freeze():
    """Call only after prepare completes, using the intended production runtime."""
    pfile = RESULT / 'protocol.json'
    if not pfile.exists():
        raise RuntimeError('Run prepare before freeze')
    old = json.loads(pfile.read_text(encoding='utf-8'))
    if old.get('run_signature'):
        protocol = load_frozen_protocol()
        print('Already frozen and validated: ' + protocol['run_signature'], flush=True)
        return
    pairs, available = build_manifest()
    if old['pairs'] != pairs:
        raise RuntimeError('Prepared pair selection differs from current selection')
    if not NUMBA_AVAILABLE:
        raise RuntimeError('Freeze must use the production runtime with Numba available')
    warmup()  # Validate that the runtime can actually compile and execute the kernel.
    protocol = protocol_definition(pairs, available)
    protocol.update(status='Frozen before v3 result inspection; not preregistered',
                    frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    runtime=runtime_fingerprint(), source_code_sha256=source_hashes(),
                    input_signatures=[_validate_crop(p) for p in pairs])
    protocol['run_signature'] = canonical_hash(protocol)
    dump(pfile, protocol)
    print('FROZEN ' + protocol['run_signature'], flush=True)


def load_frozen_protocol(validate_inputs=True):
    protocol = json.loads((RESULT / 'protocol.json').read_text(encoding='utf-8'))
    signature = protocol.get('run_signature')
    if not signature:
        raise RuntimeError('Protocol is not frozen; run freeze in the production runtime first')
    payload = {k: v for k, v in protocol.items() if k != 'run_signature'}
    if canonical_hash(payload) != signature:
        raise RuntimeError('Frozen protocol contents have been modified')
    if protocol.get('schema_version') != PROTOCOL_SCHEMA:
        raise RuntimeError('Frozen protocol schema is not supported')
    if protocol['source_code_sha256'] != source_hashes():
        raise RuntimeError('Execution source files changed after freeze; do not mix old and new results')
    if protocol['runtime'] != runtime_fingerprint():
        raise RuntimeError('Runtime differs from frozen production runtime')
    declared = protocol_definition(protocol['pairs'], protocol['sar_available_matched_patches'])
    for key, value in declared.items():
        if protocol.get(key) != value:
            raise RuntimeError(f'Frozen execution parameter changed: {key}')
    if validate_inputs:
        records = {x['pair_id']: x for x in protocol['input_signatures']}
        for p in protocol['pairs']:
            _validate_crop(p, records[p['pair_id']])
    return protocol


def evaluate(p, a, b, strategy, offsets=OFFSETS, seed=0, detail_scale=1., wavelet='db2',
             level=1, bins=32, save_maps=False, step=2, pc_kwargs=None, run_signature=None):
    W, T, S = p['geometry']
    ts = (W - T) // 2
    start = time.perf_counter()
    fa = extract_feature(a, strategy, seed=seed, image_key=p['pair_id'] + ':optical',
                         detail_scale=detail_scale, wavelet=wavelet, level=level, pc_kwargs=pc_kwargs)
    fb = extract_feature(b, strategy, seed=seed, image_key=p['pair_id'] + ':target',
                         detail_scale=detail_scale, wavelet=wavelet, level=level, pc_kwargs=pc_kwargs)
    ft = time.perf_counter() - start
    rows = []
    for dr, dc in offsets:
        sr, sc = (W - S) // 2 + dr, (W - S) // 2 + dc
        if sr < 0 or sc < 0 or sr + S > W or sc + S > W:
            raise ValueError('Search outside work crop')
        template = fa[ts:ts + T, ts:ts + T]
        search = fb[sr:sr + S, sc:sc + S]
        if search.shape != (S, S):
            raise ValueError('Search has unexpected shape')
        start_map = time.perf_counter()
        maps = similarity_maps(template, search, step=step, bins=bins,
                               template_range=(0, 1), search_range=(0, 1))
        mt = time.perf_counter() - start_map
        truth = (ts - sr, ts - sc)
        start_decode = time.perf_counter()
        current = []
        for measure in ['MI', 'CCRE']:
            record = summarize_map(maps[measure], maps['rows'], maps['cols'], truth, (T, T))
            record.update(dataset=p['dataset'], pair_id=p['pair_id'], historical=bool(p['historical']),
                          strategy=strategy, measure=measure, offset=[dr, dc], seed=seed,
                          detail_scale=detail_scale, wavelet=wavelet, level=level, bins=bins, step=step)
            current.append(record)
        dt = time.perf_counter() - start_decode
        for record in current:
            record.update(feature_time_s=ft, map_time_s=mt, peak_decode_time_s=dt,
                          single_offset_compute_s=ft + mt + dt,
                          timing_scope='joint_MI_CCRE_one_offset_full_feature_charge_diagnostic')
        rows.extend(current)
        if save_maps and (dr, dc) == (10, 10):
            path = RESULT / 'maps' / (p['pair_id'] + '_' + strategy + '.npz')
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
            with temporary.open('wb') as file:
                np.savez_compressed(file, MI=maps['MI'], CCRE=maps['CCRE'], rows=maps['rows'],
                                    cols=maps['cols'], truth=truth, run_signature=run_signature)
            os.replace(temporary, path)
    return rows


def load_pairs():
    return load_frozen_protocol()['pairs']


def warmup():
    if not NUMBA_AVAILABLE:
        raise RuntimeError('Numba required for production run')
    a = np.arange(64, dtype=float).reshape(8, 8) / 64
    similarity_maps(a[:4, :4], a, bins=32)


def one_pair(arg):
    p, mode, expected_signature = arg
    protocol = load_frozen_protocol(validate_inputs=False)
    if protocol['run_signature'] != expected_signature:
        raise RuntimeError('Protocol changed during execution')
    expected_input = next(x for x in protocol['input_signatures'] if x['pair_id'] == p['pair_id'])
    _validate_crop(p, expected_input)
    warmup()
    outdir = RESULT / (mode + '_pairs')
    outdir.mkdir(exist_ok=True)
    target = outdir / (p['pair_id'] + '.json')
    signature_path = outdir / (p['pair_id'] + '.signature.json')
    if target.exists() and signature_path.exists():
        cached = json.loads(signature_path.read_text(encoding='utf-8'))
        if cached.get('run_signature') != expected_signature or cached.get('mode') != mode:
            raise RuntimeError(f'Stale result signature: {target}')
        if cached.get('result_sha256') != sha(target):
            raise RuntimeError(f'Result checksum mismatch: {target}')
        return p['pair_id'], 'cached'
    with np.load(CACHE / (p['pair_id'] + '.npz'), allow_pickle=False) as cache:
        a, b = cache['a'], cache['b']
    rows = []
    start = time.perf_counter()
    common = dict(step=protocol['grid_step'], bins=protocol['histogram_bins'],
                  pc_kwargs=protocol['PC_parameters'], run_signature=expected_signature)
    if mode == 'main':
        for strategy in protocol['strategies']:
            rows.extend(evaluate(p, a, b, strategy, protocol['offsets'], save_maps=True, **common))
    elif mode == 'controls':
        random = protocol['random_controls']
        deterministic = protocol['deterministic_controls']
        for strategy in random['strategies']:
            for scale in random['detail_scales']:
                for seed in range(random['realizations']):
                    rows.extend(evaluate(p, a, b, strategy, [random['offset']], seed, scale, **common))
        for wl in deterministic['wavelets']:
            for level in deterministic['levels']:
                for strategy in deterministic['wavelet_strategies']:
                    rows.extend(evaluate(p, a, b, strategy, [deterministic['offset']],
                                         wavelet=wl, level=level, **common))
        for scale in deterministic['detail_scales']:
            rows.extend(evaluate(p, a, b, 'mfpc', [deterministic['offset']], detail_scale=scale, **common))
        for bins in deterministic['histogram_bins']:
            if bins == protocol['histogram_bins']:
                continue  # The corresponding main-run rows provide this baseline.
            for strategy in protocol['strategies']:
                options = {**common, 'bins': bins}
                rows.extend(evaluate(p, a, b, strategy, [deterministic['offset']], **options))
        for strategy in deterministic['extra_strategies']:
            rows.extend(evaluate(p, a, b, strategy, [deterministic['offset']], **common))
    else:
        raise ValueError(mode)
    dump(target, rows)
    dump(signature_path, dict(run_signature=expected_signature, pair_id=p['pair_id'], mode=mode,
                              row_count=len(rows), result_sha256=sha(target)))
    return p['pair_id'], round(time.perf_counter() - start, 2)


def execute(mode, workers, limit):
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError('workers and optional limit must be positive')
    protocol = load_frozen_protocol()
    all_pairs = protocol['pairs']
    if mode == 'controls':
        all_pairs = [p for p in all_pairs if p['historical']]
    pairs = all_pairs[:limit] if limit else all_pairs
    signature = protocol['run_signature']
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_pair, (p, mode, signature)) for p in pairs]
        for index, future in enumerate(concurrent.futures.as_completed(futures)):
            print(f'{mode} {index + 1}/{len(pairs)} {future.result()}', flush=True)
    partial = len(pairs) != len(all_pairs)
    suffix = f'.partial_{len(pairs)}' if partial else ''
    path = RESULT / (mode + '_rows' + suffix + '.jsonl')
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    count = 0
    with temporary.open('w', encoding='utf-8') as file:
        for p in pairs:
            source = RESULT / (mode + '_pairs') / (p['pair_id'] + '.json')
            saved = json.loads((source.with_name(p['pair_id'] + '.signature.json')).read_text(encoding='utf-8'))
            if saved['run_signature'] != signature or saved['result_sha256'] != sha(source):
                raise RuntimeError(f'Output changed before aggregation: {source}')
            rows = json.loads(source.read_text(encoding='utf-8'))
            if len(rows) != saved['row_count']:
                raise RuntimeError(f'Output row count mismatch: {source}')
            for record in rows:
                file.write(json.dumps(record, allow_nan=False) + '\n')
                count += 1
    os.replace(temporary, path)
    dump(RESULT / (mode + '_rows' + suffix + '.signature.json'),
         dict(run_signature=signature, mode=mode, partial=partial, expected_pairs=len(all_pairs),
              completed_pairs=len(pairs), pair_ids=[p['pair_id'] for p in pairs], row_count=count,
              result_sha256=sha(path)))


def timing():
    protocol = load_frozen_protocol()
    warmup()
    records = []
    rng = np.random.default_rng(protocol['selection_seed'])
    specification = protocol['timing']
    for dataset in ['SAR', 'Label', 'NIR']:
        p = next(p for p in protocol['pairs'] if p['dataset'] == dataset)
        with np.load(CACHE / (p['pair_id'] + '.npz'), allow_pickle=False) as cache:
            a, b = cache['a'], cache['b']
        for repeat in range(-specification['warmup'], specification['repeats']):
            for strategy in rng.permutation(protocol['strategies']):
                start = time.perf_counter()
                rows = evaluate(p, a, b, str(strategy), [(10, 10)], step=protocol['grid_step'],
                                bins=protocol['histogram_bins'], pc_kwargs=protocol['PC_parameters'])
                elapsed = time.perf_counter() - start
                if repeat >= 0:
                    row = rows[0]
                    records.append(dict(dataset=dataset, pair_id=p['pair_id'], strategy=str(strategy),
                                        repeat=repeat, feature_time_s=row['feature_time_s'],
                                        map_time_s=row['map_time_s'], peak_decode_time_s=row['peak_decode_time_s'],
                                        serial_compute_s=elapsed,
                                        timing_scope='in_memory_joint_MI_CCRE_single_offset_including_peak_decoding'))
            print(f'timing {dataset} repeat {repeat}', flush=True)
    path = RESULT / 'timing_rows.json'
    dump(path, records)
    dump(RESULT / 'timing_rows.signature.json', dict(run_signature=protocol['run_signature'],
                                                   row_count=len(records), result_sha256=sha(path),
                                                   estimand=specification['estimand']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'freeze', 'main', 'controls', 'timing'])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'freeze':
        freeze()
    elif args.mode == 'timing':
        timing()
    else:
        execute(args.mode, args.workers, args.limit)
