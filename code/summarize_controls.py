"""Summarize stochastic/parameter controls and serial technical timings.

Controls use only the historical 32-pair cohort and offset [10,10]. Their
baseline is extracted from the SAME main-run pairs and offset; the six-offset
main RMSE is never a control baseline. Seeds are nested in image pairs and
never counted as independent images. All control comparisons are exploratory.
Unavailable stages/settings remain explicitly pending in the output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from summarize_results import _finite, _mean, _seed, bootstrap_mean_ci, read_records


SETTING_FIELDS = ('dataset', 'strategy', 'measure', 'wavelet', 'level', 'detail_scale', 'bins')
CONTROL_OFFSET = (10.0, 10.0)
SHAPES = ('map_range', 'baseline_corrected_peak_drop', 'PSR')
TIMING_FIELDS = ('feature_time_s', 'map_time_s', 'peak_decode_time_s', 'serial_compute_s')


def setting_key(row):
    return (str(row['dataset']), str(row['strategy']), str(row['measure']), str(row['wavelet']),
            int(row['level']), float(row['detail_scale']), int(row['bins']))


def setting_dict(key):
    return dict(zip(SETTING_FIELDS, key))


def _config(strategy, wavelet='db2', level=1, detail_scale=1., bins=32):
    return strategy, wavelet, level, float(detail_scale), bins


def expected_control_settings(protocol):
    random, deterministic = protocol['random_controls'], protocol['deterministic_controls']
    bins_default = protocol['histogram_bins']
    configs = set()
    for strategy in random['strategies']:
        for scale in random['detail_scales']:
            configs.add(_config(strategy, detail_scale=scale, bins=bins_default))
    for wavelet in deterministic['wavelets']:
        for level in deterministic['levels']:
            for strategy in deterministic['wavelet_strategies']:
                configs.add(_config(strategy, wavelet, level, bins=bins_default))
    for scale in deterministic['detail_scales']:
        configs.add(_config('mfpc', detail_scale=scale, bins=bins_default))
    for bins in deterministic['histogram_bins']:
        if bins != bins_default:
            for strategy in protocol['strategies']:
                configs.add(_config(strategy, bins=bins))
    for strategy in deterministic['extra_strategies']:
        configs.add(_config(strategy, bins=bins_default))
    datasets = sorted({p['dataset'] for p in protocol['pairs'] if p['historical']})
    return {(dataset, strategy, measure, wavelet, level, scale, bins)
            for dataset in datasets for strategy, wavelet, level, scale, bins in configs
            for measure in ('MI', 'CCRE')}


def expected_pairs(protocol):
    output = defaultdict(set)
    for pair in protocol['pairs']:
        if pair['historical']:
            output[pair['dataset']].add(str(pair['pair_id']))
    return output


def select_main_baselines(main_rows, protocol):
    cohort = expected_pairs(protocol)
    selected = []
    for row in main_rows:
        if str(row['pair_id']) not in cohort.get(row['dataset'], set()):
            continue
        if tuple(map(float, row['offset'])) != CONTROL_OFFSET:
            continue
        if row['bins'] != protocol['histogram_bins'] or row['wavelet'] != 'db2' or row['level'] != 1 or row['detail_scale'] != 1:
            raise ValueError('main baseline does not have frozen default parameters')
        selected.append(row)
    return selected


def _same_result(a, b):
    """Ignore technical runtime when checking duplicate deterministic calls."""
    fields = ('err', 'success3', 'map_range', 'baseline_corrected_peak_drop', 'PSR',
              'estimated_top_left', 'degenerate', 'peak_tie_count')
    for field in fields:
        x, y = a.get(field), b.get(field)
        if x is None or y is None:
            if x != y:
                return False
        elif isinstance(x, (list, tuple)):
            if not np.allclose(x, y, rtol=1e-10, atol=1e-12):
                return False
        elif _finite(x) and _finite(y):
            if not math.isclose(float(x), float(y), rel_tol=1e-10, abs_tol=1e-12):
                return False
        elif x != y:
            return False
    return True


def aggregate_control_pairs(rows, protocol, *, is_main=False):
    cohort = expected_pairs(protocol)
    groups, duplicates = defaultdict(dict), defaultdict(int)
    for row in rows:
        key = setting_key(row)
        pair = str(row['pair_id'])
        if pair not in cohort.get(row['dataset'], set()):
            raise ValueError(f'control contains a nonhistorical pair {pair}')
        if tuple(map(float, row['offset'])) != CONTROL_OFFSET:
            raise ValueError('controls must use the matched [10,10] offset')
        if not _finite(row.get('err')) or row['err'] < 0:
            raise ValueError('control error must be finite and nonnegative')
        seed = int(row.get('seed', 0))
        full_key = key + (pair,)
        if seed in groups[full_key]:
            if not _same_result(groups[full_key][seed], row):
                raise ValueError(f'duplicate seed/configuration gives different output: {full_key}, seed={seed}')
            duplicates[full_key] += 1
        else:
            groups[full_key][seed] = row
    pairs, seed_rows = [], []
    random_strategies = set(protocol['random_controls']['strategies'])
    for full_key, seeds in sorted(groups.items()):
        key, pair = full_key[:-1], full_key[-1]
        expected_seed_n = protocol['random_controls']['realizations'] if key[1] in random_strategies and not is_main else 1
        required_seeds = set(range(expected_seed_n))
        values = list(seeds.values())
        errors = np.array([float(row['err']) for row in values])
        output = dict(setting_dict(key), pair_id=pair, n_seeds=len(seeds), expected_n_seeds=expected_seed_n,
                      complete_seeds=set(seeds) == required_seeds,
                      missing_seeds=sorted(required_seeds - set(seeds)), duplicate_deterministic_calls=duplicates[full_key],
                      mse=float(np.mean(errors ** 2)), rmse=float(np.sqrt(np.mean(errors ** 2))),
                      mean_error=float(np.mean(errors)), success_rate=_mean([r['success3'] for r in values]),
                      seed_error_min=float(errors.min()), seed_error_q1=float(np.percentile(errors, 25)),
                      seed_error_median=float(np.median(errors)), seed_error_q3=float(np.percentile(errors, 75)),
                      seed_error_max=float(errors.max()), seed_error_sd=float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.)
        for metric in SHAPES:
            output[metric + '_mean'] = _mean([r.get(metric) for r in values])
            output[metric + '_n_seeds'] = sum(_finite(r.get(metric)) for r in values)
        pairs.append(output)
        for seed, row in sorted(seeds.items()):
            seed_rows.append(dict(setting_dict(key), pair_id=pair, seed=seed, err=float(row['err']),
                                  success3=bool(row['success3']), complete_pair_seeds=output['complete_seeds']))
    return pairs, seed_rows


def setting_summaries(pair_rows, protocol, *, n_boot=10000, seed=12345, is_main=False):
    groups = defaultdict(list)
    for row in pair_rows:
        groups[setting_key(row)].append(row)
    keys = set(groups) if is_main else expected_control_settings(protocol) | set(groups)
    cohort = expected_pairs(protocol)
    summaries = []
    for key in sorted(keys):
        all_pairs = groups[key]
        complete = [r for r in all_pairs if r['complete_seeds']]
        observed_ids = {r['pair_id'] for r in complete}
        planned_ids = cohort[key[0]]
        status = 'complete' if observed_ids == planned_ids else ('pending' if not all_pairs else 'pending_incomplete')
        out = dict(setting_dict(key), status=status, n_pairs=len(complete), expected_n_pairs=len(planned_ids),
                   n_pairs_observed=len(all_pairs), missing_pair_ids=sorted(planned_ids - observed_ids),
                   pair_ids=sorted(observed_ids), n_seed_observations=sum(r['n_seeds'] for r in all_pairs),
                   duplicate_deterministic_calls=sum(r['duplicate_deterministic_calls'] for r in all_pairs),
                   baseline_offset=[10, 10], source_kind='main_matched_baseline' if is_main else 'controls')
        if complete:
            mse = np.array([r['mse'] for r in complete])
            ci = bootstrap_mean_ci(mse, n_boot=n_boot, seed=_seed(seed, (key, 'control_mse')))
            out.update(pooled_rmse_px=float(np.sqrt(mse.mean())),
                       pooled_rmse_ci95=[math.sqrt(v) for v in ci] if ci else None,
                       mean_pair_rmse_px=_mean([r['rmse'] for r in complete]),
                       success_rate=_mean([r['success_rate'] for r in complete]),
                       success_rate_ci95=bootstrap_mean_ci([r['success_rate'] for r in complete], n_boot=n_boot,
                                                          seed=_seed(seed, (key, 'control_success'))),
                       median_pair_seed_error_sd=_mean([r['seed_error_sd'] for r in complete]))
            out['median_pair_seed_error_sd'] = float(np.median([r['seed_error_sd'] for r in complete]))
            for metric in SHAPES:
                out[metric + '_mean_across_pair_means'] = _mean([r[metric + '_mean'] for r in complete])
                out[metric + '_n_pairs'] = sum(_finite(r[metric + '_mean']) for r in complete)
        else:
            out.update(pooled_rmse_px=None, pooled_rmse_ci95=None, mean_pair_rmse_px=None, success_rate=None,
                       success_rate_ci95=None, median_pair_seed_error_sd=None)
        summaries.append(out)
    return summaries


def seed_variation(seed_rows, protocol):
    """Describe variation across seed scenarios; these are NOT inferential n."""
    groups = defaultdict(list)
    for row in seed_rows:
        if row['complete_pair_seeds']:
            groups[(setting_key(row), row['seed'])].append(row)
    cohort = expected_pairs(protocol)
    scenarios, by_setting = [], defaultdict(list)
    for (key, seed), rows in sorted(groups.items()):
        complete = {r['pair_id'] for r in rows} == cohort[key[0]]
        out = dict(setting_dict(key), seed=seed, n_pairs=len(rows), complete_pair_cohort=complete,
                   pooled_rmse_px=math.sqrt(np.mean([r['err'] ** 2 for r in rows])),
                   success_rate=_mean([r['success3'] for r in rows]))
        scenarios.append(out)
        if complete:
            by_setting[key].append(out)
    descriptions = []
    for key, rows in sorted(by_setting.items()):
        values = np.array([r['pooled_rmse_px'] for r in rows])
        descriptions.append(dict(setting_dict(key), n_seed_scenarios=len(rows), independent_unit='image pair, not seed',
                                 seed_rmse_min=float(values.min()), seed_rmse_q1=float(np.percentile(values, 25)),
                                 seed_rmse_median=float(np.median(values)), seed_rmse_q3=float(np.percentile(values, 75)),
                                 seed_rmse_max=float(values.max()),
                                 seed_rmse_sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0.))
    return scenarios, descriptions


def _reference_strategy(key, protocol):
    strategy, bins = key[1], key[-1]
    if bins != protocol['histogram_bins'] and strategy in protocol['strategies']:
        return strategy
    if strategy == 'mfpc_scale_matched':
        return strategy
    if strategy in ('noise', 'surrogate', 'mfpc'):
        return 'mfpc'
    return 'full_pc'


def matched_baseline_comparisons(control_pairs, baseline_pairs, protocol, *, n_boot=10000, seed=12345):
    controls, baselines = defaultdict(dict), defaultdict(dict)
    for row in control_pairs:
        if row['complete_seeds']:
            controls[setting_key(row)][row['pair_id']] = row
    for row in baseline_pairs:
        if row['complete_seeds']:
            baselines[setting_key(row)][row['pair_id']] = row
    results = []
    for key, a in sorted(controls.items()):
        reference = _reference_strategy(key, protocol)
        baseline_key = (key[0], reference, key[2], 'db2', 1, 1., protocol['histogram_bins'])
        b = baselines.get(baseline_key, {})
        common = sorted(set(a) & set(b))
        out = dict(setting_dict(key), baseline_strategy=reference, baseline_wavelet='db2', baseline_level=1,
                   baseline_detail_scale=1., baseline_bins=protocol['histogram_bins'], offset=[10, 10],
                   comparison_type='exploratory parameter/control contrast on matched pairs and offset',
                   baseline_source='main at offset [10,10], historical cohort only', n_pairs=len(common), pair_ids=common,
                   missing_baseline_pairs=sorted(set(a) - set(b)),
                   status='complete' if set(common) == expected_pairs(protocol)[key[0]] else 'pending_incomplete')
        if common:
            amse = np.array([a[p]['mse'] for p in common]); bmse = np.array([b[p]['mse'] for p in common])
            pair_difference = np.sqrt(amse) - np.sqrt(bmse)
            out.update(delta_pooled_rmse_px=float(np.sqrt(amse.mean()) - np.sqrt(bmse.mean())),
                       delta_mean_pair_rmse_px=float(pair_difference.mean()),
                       delta_mean_pair_rmse_ci95=bootstrap_mean_ci(pair_difference, n_boot=n_boot,
                                                                 seed=_seed(seed, (key, 'control_baseline_pair'))))
            ci = None
            if len(common) >= 2:
                rng = np.random.default_rng(_seed(seed, (key, 'control_baseline_pooled')))
                indices = rng.integers(0, len(common), (n_boot, len(common)))
                difference = np.sqrt(amse[indices].mean(axis=1)) - np.sqrt(bmse[indices].mean(axis=1))
                ci = [float(x) for x in np.percentile(difference, [2.5, 97.5])]
            out['delta_pooled_rmse_ci95'] = ci
        results.append(out)
    return results


def timing_summaries(records, protocol):
    groups = defaultdict(list)
    seen = set()
    for row in records:
        key = (str(row['dataset']), str(row['pair_id']), str(row['strategy']))
        repeat = int(row['repeat'])
        if key + (repeat,) in seen:
            raise ValueError('duplicate technical timing repeat')
        seen.add(key + (repeat,)); groups[key].append(row)
    summaries, lookup = [], {}
    expected_repeats = protocol['timing']['repeats']
    for key, rows in sorted(groups.items()):
        out = dict(zip(('dataset', 'pair_id', 'strategy'), key))
        out.update(n_technical_repeats=len(rows), n_independent_pairs=1,
                   expected_repeats=expected_repeats,
                   status='complete' if {r['repeat'] for r in rows} == set(range(expected_repeats)) else 'pending_incomplete',
                   timing_scope=protocol['timing']['estimand'])
        for metric in TIMING_FIELDS:
            values = [float(r[metric]) for r in rows if _finite(r.get(metric))]
            out[metric + '_n'] = len(values)
            for label, percentile in (('median', 50), ('q1', 25), ('q3', 75)):
                out[metric + '_' + label] = float(np.percentile(values, percentile)) if values else None
        summaries.append(out); lookup[key] = out
    for out in summaries:
        baseline = lookup.get((out['dataset'], out['pair_id'], 'full_pc'))
        for metric in TIMING_FIELDS:
            denominator = out[metric + '_median']
            numerator = baseline[metric + '_median'] if baseline else None
            out[metric + '_full_pc_over_strategy_ratio_point'] = numerator / denominator if numerator is not None and denominator and denominator > 0 else None
        out['ratio_inference'] = 'ratio of technical-repeat medians only; no paired image-level significance or CI'
    return summaries


def _load_verified(path, protocol, *, allow_missing=False):
    path = Path(path)
    if not path.exists():
        if allow_missing:
            return [], {'status': 'pending', 'path': str(path)}
        raise FileNotFoundError(path)
    signature = path.with_suffix('.signature.json')
    if not signature.exists():
        raise ValueError(f'missing completion signature {signature}')
    meta = json.loads(signature.read_text(encoding='utf-8'))
    if meta.get('run_signature') != protocol['run_signature']:
        raise ValueError(f'protocol signature mismatch {path}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if meta.get('result_sha256') != digest:
        raise ValueError(f'result checksum mismatch {path}')
    if meta.get('partial') or ('completed_pairs' in meta and meta['completed_pairs'] != meta['expected_pairs']):
        raise ValueError(f'partial stage cannot be used as a completed result {path}')
    rows = read_records(path)
    if meta.get('row_count') != len(rows):
        raise ValueError(f'row count mismatch {path}')
    return rows, dict(status='complete', path=str(path.resolve()), sha256=digest, row_count=len(rows))


def _csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, tuple)) else v for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()
    protocol = json.loads((args.results_dir / 'protocol.json').read_text(encoding='utf-8'))
    main_rows, main_meta = _load_verified(args.results_dir / 'main_rows.jsonl', protocol)
    controls, controls_meta = _load_verified(args.results_dir / 'controls_rows.jsonl', protocol, allow_missing=True)
    timings, timing_meta = _load_verified(args.results_dir / 'timing_rows.json', protocol, allow_missing=True)
    baselines = select_main_baselines(main_rows, protocol)
    cp, seeds = aggregate_control_pairs(controls, protocol)
    bp, _ = aggregate_control_pairs(baselines, protocol, is_main=True)
    scenarios, variation = seed_variation(seeds, protocol)
    report = dict(
        analysis_contract=dict(independent_unit='historical image pair', offset=[10, 10],
                               seed_aggregation='equal seed weight within pair; pair bootstrap only',
                               baseline='same main-run historical pairs, same [10,10] offset; not six-offset main RMSE',
                               parameter_comparisons='exploratory; explicit default-parameter reference, no p values',
                               bootstrap_replicates=args.bootstrap, bootstrap_seed=args.seed,
                               timing='one independent pair per dataset, 30 technical repetitions; median and quartiles, point ratios only'),
        input_stages=dict(main=main_meta, controls=controls_meta, timing=timing_meta),
        expected_cohort_counts={ds: len(pairs) for ds, pairs in expected_pairs(protocol).items()},
        control_pair_metrics=cp,
        control_summaries=setting_summaries(cp, protocol, n_boot=args.bootstrap, seed=args.seed),
        matched_main_baselines=setting_summaries(bp, protocol, n_boot=args.bootstrap, seed=args.seed, is_main=True),
        seed_scenarios=scenarios, seed_variation=variation,
        baseline_comparisons=matched_baseline_comparisons(cp, bp, protocol, n_boot=args.bootstrap, seed=args.seed),
        timing_summaries=timing_summaries(timings, protocol),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'controls_and_timing_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    _csv(args.out_dir / 'controls_source_rows.csv', controls)
    _csv(args.out_dir / 'matched_main_source_rows.csv', baselines)
    _csv(args.out_dir / 'timing_source_rows.csv', timings)
    for name in ('control_pair_metrics', 'control_summaries', 'matched_main_baselines', 'seed_scenarios',
                 'seed_variation', 'baseline_comparisons', 'timing_summaries'):
        _csv(args.out_dir / f'{name}.csv', report[name])
    print(json.dumps({'output': str(args.out_dir), 'stages': report['input_stages'],
                      'complete_control_settings': sum(x['status'] == 'complete' for x in report['control_summaries']),
                      'pending_control_settings': sum(x['status'] != 'complete' for x in report['control_summaries'])}))


if __name__ == '__main__':
    main()
