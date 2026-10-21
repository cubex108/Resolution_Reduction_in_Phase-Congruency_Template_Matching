"""Pair-level analysis of the v3 matching benchmark.

Input is JSONL with dataset, pair_id, strategy, measure, offset=[row,col],
seed, and fields returned by matching_metrics.summarize_map.  Timing fields
are feature_time_s, map_time_s (shared MI+CCRE computation), and
single_offset_compute_s. Main-run timings are diagnostics only; formal
serial timing uses serial_compute_s in timing_rows.json.

Random realisations are first averaged WITHIN pair/configuration/offset;
offsets then receive equal weight within the image pair.  Independent units
for confidence intervals, signed-rank tests and resampling are pair_id.
An exact conditional signed-rank distribution handles tied absolute ranks,
discards exactly zero differences, and requires exchangeable signs under H0.
It is not a test of the arithmetic-mean null; CIs describe mean differences.

The new localization families replace, and never inherit p values from, the
old 21 surface-shape comparisons.  Family 1 contains seven CCRE pair-RMSE
contrasts per dataset. Family 2 contains CCRE-minus-MI pair-RMSE contrasts
for all seven strategies per dataset. Each planned family is Holm-adjusted
separately, including unestimable planned contrasts in its multiplicity count.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


STRATEGIES = ('raw', 'full_pc', 'down_pc', 'mfpc', 'mfpc_scale_matched', 'no_detail', 'gated')
FEATURE_CONTRASTS = (
    ('full_pc', 'raw'), ('down_pc', 'full_pc'), ('mfpc', 'full_pc'),
    ('mfpc', 'down_pc'), ('mfpc_scale_matched', 'mfpc'),
    ('no_detail', 'mfpc'), ('gated', 'mfpc'),
)
SHAPES = ('baseline_corrected_peak_drop', 'PSR')
TIMINGS = ('feature_time_s', 'map_time_s', 'end_to_end_s')
METRICS = ('err', 'success3', 'map_range') + SHAPES + TIMINGS


def _finite(value):
    return value is not None and isinstance(value, (int, float, bool, np.number)) and math.isfinite(float(value))


def _mean(values):
    finite = [float(v) for v in values if _finite(v)]
    return float(np.mean(finite)) if finite else None


def _seed(seed, key):
    digest = hashlib.sha256(str(key).encode('utf-8')).digest()
    return (int(seed) + int.from_bytes(digest[:8], 'little')) % (2 ** 63 - 1)


def _offset_key(record):
    offset = record.get('offset')
    if not isinstance(offset, (list, tuple)) or len(offset) != 2 or not all(_finite(x) for x in offset):
        raise ValueError('each row requires a finite offset=[row,column]')
    return tuple(float(x) for x in offset)


def aggregate_pairs(records, *, expected_offsets=None):
    """Return pair summaries and equal-realisation-averaged offset records.

    Configurations with missing offsets are retained and flagged.  They are
    excluded from the inferential analysis, never silently completed or given
    disproportionate weight. Repeated identical seed/offset keys are errors.
    """
    groups = defaultdict(list)
    offsets_by_dataset = defaultdict(set)
    seen = set()
    for record in records:
        for field in ('dataset', 'pair_id', 'strategy', 'measure'):
            if field not in record:
                raise ValueError(f'missing required field {field}')
        key = tuple(str(record[field]) for field in ('dataset', 'pair_id', 'strategy', 'measure'))
        offset = _offset_key(record)
        seed = record.get('seed')
        unique = key + (offset, str(seed))
        if unique in seen:
            raise ValueError(f'duplicate pair/configuration/offset/seed record: {unique}')
        seen.add(unique)
        if not _finite(record.get('err')) or float(record['err']) < 0:
            raise ValueError(f'err must be finite and nonnegative: {unique}')
        if record.get('success3') not in (True, False, 0, 1):
            raise ValueError(f'success3 must be a binary result: {unique}')
        groups[key + (offset,)].append(record)
        offsets_by_dataset[key[0]].add(offset)
    if expected_offsets is not None:
        common = {tuple(map(float, x)) for x in expected_offsets}
        for dataset, observed in offsets_by_dataset.items():
            if not observed <= common:
                raise ValueError(f'unexpected offsets in {dataset}: {observed - common}')
            offsets_by_dataset[dataset] = common

    offset_rows = []
    pair_groups = defaultdict(list)
    for key, rows in sorted(groups.items()):
        dataset, pair_id, strategy, measure, offset = key
        output = dict(dataset=dataset, pair_id=pair_id, strategy=strategy,
                      measure=measure, offset=list(offset), n_realisations=len(rows))
        output.update({metric: _mean([r.get(metric) for r in rows]) for metric in METRICS})
        output['mse'] = float(np.mean([float(r['err']) ** 2 for r in rows]))
        output['degenerate_rate'] = _mean([r.get('degenerate', False) for r in rows])
        for metric in SHAPES:
            output[metric + '_n_realisations'] = sum(_finite(r.get(metric)) for r in rows)
        offset_rows.append(output)
        pair_groups[key[:4]].append(output)

    pair_rows = []
    for key, offsets in sorted(pair_groups.items()):
        dataset, pair_id, strategy, measure = key
        observed = {tuple(x['offset']) for x in offsets}
        expected = offsets_by_dataset[dataset]
        output = dict(dataset=dataset, pair_id=pair_id, strategy=strategy, measure=measure,
                      n_offsets=len(offsets), expected_n_offsets=len(expected),
                      complete_offsets=observed == expected,
                      missing_offsets=[list(x) for x in sorted(expected - observed)],
                      n_realisations=sum(x['n_realisations'] for x in offsets))
        output.update({metric: _mean([r.get(metric) for r in offsets]) for metric in METRICS})
        output['mse'] = float(np.mean([r['mse'] for r in offsets]))
        output['rmse'] = math.sqrt(output['mse'])
        output['degenerate_rate'] = _mean([r['degenerate_rate'] for r in offsets])
        for metric in SHAPES:
            valid = sum(_finite(r.get(metric)) for r in offsets)
            output[metric + '_n_offsets'] = valid
            # Primary pair correlations require every planned offset and all
            # random realisations to have an observable shape measurement.
            output[metric + '_complete'] = bool(
                observed == expected and valid == len(expected)
                and all(r[metric + '_n_realisations'] == r['n_realisations'] for r in offsets))
        pair_rows.append(output)
    return pair_rows, offset_rows


def bootstrap_mean_ci(values, *, n_boot=10000, seed=12345):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return None
    if not np.isfinite(values).all():
        raise ValueError('bootstrap input must be finite')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    replicates = values[indices].mean(axis=1)
    return [float(x) for x in np.percentile(replicates, [2.5, 97.5])]


def _ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='mergesort')
    ranked = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranked[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranked


def exact_signed_rank(differences):
    """Two-sided exact conditional sign distribution with average tied ranks."""
    difference = np.asarray(differences, dtype=float)
    if not np.isfinite(difference).all():
        raise ValueError('signed-rank input must be finite')
    difference = difference[difference != 0]
    n = len(difference)
    if n == 0:
        return {'p': 1.0, 'n_nonzero': 0, 'statistic': 0.0}
    weights = np.rint(2 * _ranks(np.abs(difference))).astype(int)
    positive = int(weights[difference > 0].sum())
    total = int(weights.sum())
    observed = min(positive, total - positive)
    counts = [0] * (total + 1)
    counts[0] = 1
    reachable = 0
    for weight in weights:
        weight = int(weight)
        for value in range(reachable, -1, -1):
            counts[value + weight] += counts[value]
        reachable += weight
    p = min(1.0, 2 * sum(counts[:observed + 1]) / (2 ** n))
    return {'p': float(p), 'n_nonzero': n, 'statistic': observed / 2.0}


def holm_adjust(p_values):
    """Keep all planned hypotheses in m; unavailable p values are not selected out."""
    valid = np.array([_finite(p) for p in p_values], dtype=bool)
    p = np.array([float(p) if ok else 1.0 for p, ok in zip(p_values, valid)])
    order = np.argsort(p, kind='mergesort')
    adjusted = np.empty(len(p))
    adjusted[order] = np.maximum.accumulate(np.minimum(1.0, p[order] * np.arange(len(p), 0, -1)))
    return [float(value) if ok else None for value, ok in zip(adjusted, valid)]


def configuration_summaries(pair_rows, *, n_boot=10000, seed=12345):
    groups = defaultdict(list)
    for row in pair_rows:
        groups[(row['dataset'], row['strategy'], row['measure'])].append(row)
    summaries = []
    for key, all_rows in sorted(groups.items()):
        rows = [r for r in all_rows if r['complete_offsets']]
        out = dict(zip(('dataset', 'strategy', 'measure'), key))
        out.update(n_pairs=len(rows), incomplete_pairs=[r['pair_id'] for r in all_rows if not r['complete_offsets']],
                   pair_ids=[r['pair_id'] for r in rows], n_offsets_per_pair=sorted({r['n_offsets'] for r in rows}))
        if rows:
            mse = np.array([r['mse'] for r in rows])
            ci = bootstrap_mean_ci(mse, n_boot=n_boot, seed=_seed(seed, (key, 'mse')))
            out['pooled_rmse_px'] = float(np.sqrt(mse.mean()))
            out['pooled_rmse_ci95'] = [math.sqrt(x) for x in ci] if ci else None
            out['mean_pair_rmse_px'] = _mean([r['rmse'] for r in rows])
            for metric in ('err', 'success3', 'map_range') + TIMINGS:
                values = [r[metric] for r in rows if _finite(r.get(metric))]
                out[metric + '_mean'] = _mean(values)
                out[metric + '_ci95'] = bootstrap_mean_ci(values, n_boot=n_boot, seed=_seed(seed, (key, metric)))
                out[metric + '_n_pairs'] = len(values)
                out[metric + '_empirical_distribution_constant'] = bool(len(values) > 0 and len(set(values)) == 1)
            out['success_rate_ci_boundary_warning'] = bool(
                out['success3_empirical_distribution_constant'] and out['success3_mean'] in (0.0, 1.0))
        summaries.append(out)
    return summaries


def planned_contrasts(pair_rows, *, n_boot=10000, seed=12345):
    available = defaultdict(dict)
    datasets = sorted({row['dataset'] for row in pair_rows})
    for row in pair_rows:
        if row['complete_offsets']:
            available[(row['dataset'], row['strategy'], row['measure'])][row['pair_id']] = row
    plans = []
    for dataset in datasets:
        for strategy_a, strategy_b in FEATURE_CONTRASTS:
            plans.append(('feature_strategies_CCRE', dataset, strategy_a, 'CCRE', strategy_b, 'CCRE'))
        for strategy in STRATEGIES:
            plans.append(('similarity_measure', dataset, strategy, 'CCRE', strategy, 'MI'))
    contrasts = []
    for family, dataset, sa, ma, sb, mb in plans:
        a, b = available[(dataset, sa, ma)], available[(dataset, sb, mb)]
        common = sorted(set(a) & set(b))
        difference = np.array([a[p]['rmse'] - b[p]['rmse'] for p in common], dtype=float)
        out = dict(family=family, dataset=dataset, strategy_a=sa, measure_a=ma,
                   strategy_b=sb, measure_b=mb, n_pairs=len(common), pair_ids=common,
                   missing_a=sorted(set(b) - set(a)), missing_b=sorted(set(a) - set(b)),
                   estimand='mean paired difference of within-pair RMSE over offsets, A minus B',
                   delta_pair_rmse_px=_mean(difference),
                   ci95=bootstrap_mean_ci(difference, n_boot=n_boot, seed=_seed(seed, (family, dataset, sa, ma, sb, mb))))
        if len(common) >= 2:
            out.update(exact_signed_rank(difference))
        else:
            out.update(p=None, n_nonzero=int(np.sum(difference != 0)), statistic=None)
        contrasts.append(out)
    for family in ('feature_strategies_CCRE', 'similarity_measure'):
        rows = [r for r in contrasts if r['family'] == family]
        adjusted = holm_adjust([r['p'] for r in rows])
        for row, p in zip(rows, adjusted):
            row.update(holm_p=p, family_size=len(rows), significant_holm05=p is not None and p < .05)
    return contrasts


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return None
    x, y = _ranks(x), _ranks(y)
    x -= x.mean(); y -= y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator > 0 else None


def _correlation_ci(replicates, n_boot):
    valid = [v for v in replicates if _finite(v)]
    # A mostly constant/tied bootstrap distribution is not a usable CI.
    return {
        'ci95': [float(x) for x in np.percentile(valid, [2.5, 97.5])] if len(valid) >= .8 * n_boot else None,
        'n_valid_bootstraps': len(valid), 'n_bootstraps': n_boot,
        'ci_available': bool(len(valid) >= .8 * n_boot),
    }


def shape_error_correlations(pair_rows, offset_rows, *, n_boot=2000, seed=12345):
    """Pair-level, pooled-offset cluster, and average-offset Spearman analyses.

    Primary pair analysis uses mean shape and pair RMSE only when shape is
    observed in every planned offset and every realisation. Pooled and
    per-offset results preserve the observable subset and disclose its size.
    Cluster bootstrap resamples complete pair blocks, never individual offsets.
    The average-offset analysis averages six separate across-pair Spearman rhos;
    it is unavailable if any planned offset has undefined correlation.
    All correlation analyses are exploratory; no p-value selection is applied.
    """
    pairs, offsets = defaultdict(list), defaultdict(list)
    for row in pair_rows:
        pairs[(row['dataset'], row['strategy'], row['measure'])].append(row)
    for row in offset_rows:
        offsets[(row['dataset'], row['strategy'], row['measure'])].append(row)
    results = []
    for key, all_pairs in sorted(pairs.items()):
        complete = [p for p in all_pairs if p['complete_offsets']]
        pair_ids = sorted(p['pair_id'] for p in complete)
        id_to_row = {p: i for i, p in enumerate(pair_ids)}
        selected_offsets = [r for r in offsets[key] if r['pair_id'] in id_to_row]
        offset_values = sorted({tuple(r['offset']) for r in selected_offsets})
        offset_to_column = {off: i for i, off in enumerate(offset_values)}
        for shape in SHAPES:
            base = dict(zip(('dataset', 'strategy', 'measure'), key))
            base['shape_metric'] = shape
            eligible = [p for p in complete if p[shape + '_complete']]
            x = np.array([p[shape] for p in eligible], dtype=float)
            y = np.array([p['rmse'] for p in eligible], dtype=float)
            rho = spearman(x, y)
            rng = np.random.default_rng(_seed(seed, (key, shape, 'pair')))
            boot = []
            if rho is not None:
                for _ in range(n_boot):
                    ii = rng.integers(0, len(x), len(x))
                    boot.append(spearman(x[ii], y[ii]))
            results.append(dict(base, analysis='pair_mean_shape_vs_pair_rmse', rho=rho,
                                n_pairs=len(eligible), n_pairs_total=len(all_pairs),
                                excluded_pair_ids=[p['pair_id'] for p in all_pairs if p not in eligible],
                                validity_rule='shape present in every offset and every realisation',
                                **_correlation_ci(boot, n_boot)))

            xs = np.full((len(pair_ids), len(offset_values)), np.nan)
            ys = np.full_like(xs, np.nan)
            for row in selected_offsets:
                i, j = id_to_row[row['pair_id']], offset_to_column[tuple(row['offset'])]
                if _finite(row.get(shape)):
                    xs[i, j] = float(row[shape]); ys[i, j] = float(row['err'])
            by_offset = []
            for j, off in enumerate(offset_values):
                valid = np.isfinite(xs[:, j]) & np.isfinite(ys[:, j])
                by_offset.append(dict(offset=list(off), n_pairs=int(valid.sum()),
                                      rho=spearman(xs[:, j], ys[:, j])))
            rho_pooled = spearman(xs.ravel(), ys.ravel())
            mean_rho = _mean([r['rho'] for r in by_offset]) if by_offset and all(r['rho'] is not None for r in by_offset) else None
            pooled_boot, averaged_boot = [], []
            rng = np.random.default_rng(_seed(seed, (key, shape, 'cluster')))
            if rho_pooled is not None or mean_rho is not None:
                for _ in range(n_boot):
                    ii = rng.integers(0, len(pair_ids), len(pair_ids))
                    xx, yy = xs[ii], ys[ii]
                    if rho_pooled is not None:
                        pooled_boot.append(spearman(xx.ravel(), yy.ravel()))
                    if mean_rho is not None:
                        rhos = [spearman(xx[:, j], yy[:, j]) for j in range(xx.shape[1])]
                        averaged_boot.append(_mean(rhos) if all(r is not None for r in rhos) else None)
            results.append(dict(base, analysis='pooled_offsets_pair_cluster_bootstrap', rho=rho_pooled,
                                n_pairs=int(np.any(np.isfinite(xs), axis=1).sum()),
                                n_observations=int(np.isfinite(xs).sum()),
                                **_correlation_ci(pooled_boot, n_boot)))
            results.append(dict(base, analysis='mean_of_offset_specific_correlations', rho=mean_rho,
                                n_pairs=len(pair_ids), n_offsets=len(offset_values), by_offset=by_offset,
                                **_correlation_ci(averaged_boot, n_boot)))
    return results


def summarize_records(records, *, n_boot=10000, correlation_boot=2000, seed=12345,
                      expected_offsets=None, include_correlations=True):
    records = list(records)
    if not records:
        raise ValueError('no observations')
    pairs, offsets = aggregate_pairs(records, expected_offsets=expected_offsets)
    return {
        'analysis_contract': {
            'independent_unit': 'dataset + pair_id',
            'aggregation': 'average realisations within offset, then equal weight offsets within pair',
            'bootstrap': 'pair-level percentile 95% CI; PCG64; numpy default_rng',
            'n_bootstrap': n_boot, 'correlation_bootstrap': correlation_boot, 'seed': seed,
            'test': 'two-sided exact conditional signed-rank with average tied ranks and exact-zero removal',
            'test_null': 'symmetric paired difference distribution about zero; exchangeable signs',
            'mean_ci_and_rank_test_have_different_estimands': True,
            'families': ['feature_strategies_CCRE', 'similarity_measure'],
            'old_shape_family_p_values_are_not_reused': True,
            'rmse_summary': 'sqrt(mean across pairs of within-pair mean squared error)',
            'contrast_estimand': 'mean across pairs of paired within-pair RMSE difference',
            'map_time_s_is_shared_between_MI_and_CCRE': True,
            'bootstrap_boundary_caution': 'all-zero/all-one empirical success gives degenerate percentile CI; this does not prove zero population uncertainty',
            'correlations': 'exploratory; missing ring measurements not set to zero',
        },
        'n_raw_records': len(records),
        'n_independent_pairs': len({(r['dataset'], r['pair_id']) for r in pairs}),
        'dataset_pair_counts': {ds: len({r['pair_id'] for r in pairs if r['dataset'] == ds})
                                for ds in sorted({r['dataset'] for r in pairs})},
        'pair_metrics': pairs,
        'offset_metrics': offsets,
        'configuration_summaries': configuration_summaries(pairs, n_boot=n_boot, seed=seed),
        'contrasts': planned_contrasts(pairs, n_boot=n_boot, seed=seed),
        'shape_error_correlations': shape_error_correlations(pairs, offsets, n_boot=correlation_boot, seed=seed)
                                    if include_correlations else [],
    }


def read_records(path):
    text = Path(path).read_text(encoding='utf-8-sig').strip()
    if not text:
        return []
    if text.startswith('['):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--correlation-bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--skip-correlations', action='store_true')
    args = parser.parse_args()
    if args.bootstrap < 100 or args.correlation_bootstrap < 100:
        parser.error('use at least 100 bootstrap draws (publication default 10000/2000)')
    expected = [(0, 0), (5, 0), (0, 5), (10, 10), (-20, 15), (40, -30)]
    summary = summarize_records(read_records(args.input), n_boot=args.bootstrap,
                                correlation_boot=args.correlation_bootstrap, seed=args.seed,
                                expected_offsets=expected, include_correlations=not args.skip_correlations)
    summary['input_file'] = str(args.input.resolve())
    summary['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    destination = args.out_dir / 'statistical_summary.json'
    destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    for name in ('pair_metrics', 'configuration_summaries', 'contrasts', 'shape_error_correlations'):
        _csv(args.out_dir / f'{name}.csv', summary[name])
    print(json.dumps({'output': str(destination), 'n_independent_pairs': summary['n_independent_pairs'],
                      'dataset_pair_counts': summary['dataset_pair_counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
