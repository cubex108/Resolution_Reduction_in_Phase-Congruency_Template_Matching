"""Pair-level descriptive analysis of a complete supplemental stress run."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derived_seed(seed,key):
    return int.from_bytes(hashlib.sha256(json.dumps([seed,key],sort_keys=True).encode()).digest()[:8],'little')


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def aggregate_pairs(rows,protocol):
    """Average seeds within offsets, then equal-weight offsets within pairs."""
    groups=defaultdict(lambda:defaultdict(dict)); seen=set()
    required_offsets={tuple(x) for x in protocol['offsets']}
    conditions={'unperturbed':{None},**{f'gamma_k{k}':set(range(protocol['realizations'])) for k in protocol['gamma_shape_k']}}
    pair_ids={p['pair_id'] for p in protocol['pairs']}
    for r in rows:
        key=(r['pair_id'],r['condition'],r['strategy'],r['measure']); offset=tuple(r['offset']); seed=r['seed']
        if key+(offset,seed) in seen:raise ValueError('Duplicate metric record')
        seen.add(key+(offset,seed))
        if r['pair_id'] not in pair_ids or r['condition'] not in conditions or offset not in required_offsets:
            raise ValueError('Unexpected pair, condition or offset')
        if r['strategy'] not in protocol['strategies'] or r['measure'] not in ('MI','CCRE'):
            raise ValueError('Unexpected strategy/measure')
        if not math.isfinite(r['err']) or r['err']<0:raise ValueError('Invalid error')
        groups[key][offset][seed]=r
    expected={(p,c,s,m) for p in pair_ids for c in conditions for s in protocol['strategies'] for m in ('MI','CCRE')}
    if set(groups)!=expected:raise ValueError('Incomplete pair/condition settings')
    output=[]
    for (pair,condition,strategy,measure),offsets in sorted(groups.items()):
        if set(offsets)!=required_offsets:raise ValueError('Incomplete offsets')
        if any(set(seeds)!=conditions[condition] for seeds in offsets.values()):raise ValueError('Incomplete random realizations')
        offset_mse=[np.mean([r['err']**2 for r in seeds.values()]) for seeds in offsets.values()]
        success=[np.mean([r['success3'] for r in seeds.values()]) for seeds in offsets.values()]
        mse=float(np.mean(offset_mse))
        output.append(dict(pair_id=pair,condition=condition,strategy=strategy,measure=measure,
                           n_offsets=len(offsets),n_realizations=len(conditions[condition]),
                           mse=mse,rmse_px=math.sqrt(mse),success_rate=float(np.mean(success))))
    return output


def percentile(x):return [float(v) for v in np.percentile(x,[2.5,97.5])]


def summarize_pairs(pair_rows,protocol):
    groups=defaultdict(dict)
    for row in pair_rows:groups[(row['condition'],row['strategy'],row['measure'])][row['pair_id']]=row
    results=[]; differences=[]; paired_rows=[]
    for key,pairs in sorted(groups.items()):
        condition,strategy,measure=key; ids=sorted(pairs); n=len(ids)
        mse=np.array([pairs[p]['mse'] for p in ids]); success=np.array([pairs[p]['success_rate'] for p in ids])
        rng=np.random.default_rng(derived_seed(protocol['bootstrap_seed'],key))
        index=rng.integers(0,n,(protocol['bootstrap_replicates'],n))
        summary=dict(condition=condition,strategy=strategy,measure=measure,n_pairs=n,pair_ids=ids,
                     n_realizations=pairs[ids[0]]['n_realizations'],n_offsets=len(protocol['offsets']),
                     pooled_rmse_px=float(np.sqrt(mse.mean())),pooled_rmse_ci95=percentile(np.sqrt(mse[index].mean(axis=1))),
                     mean_pair_rmse_px=float(np.sqrt(mse).mean()),success_rate=float(success.mean()),
                     success_rate_ci95=percentile(success[index].mean(axis=1)),
                     empirical_success_constant=bool(np.all(success==success[0])))
        results.append(summary)
        if condition=='unperturbed':continue
        base=groups[('unperturbed',strategy,measure)]
        bmse=np.array([base[p]['mse'] for p in ids]); bs=np.array([base[p]['success_rate'] for p in ids])
        d=np.sqrt(mse)-np.sqrt(bmse)
        differences.append(dict(condition=condition,strategy=strategy,measure=measure,n_pairs=n,
             reference='same ten pairs, same six offsets, unperturbed observed images',
             delta_pooled_rmse_px=float(np.sqrt(mse.mean())-np.sqrt(bmse.mean())),
             delta_pooled_rmse_ci95=percentile(np.sqrt(mse[index].mean(axis=1))-np.sqrt(bmse[index].mean(axis=1))),
             mean_paired_rmse_delta_px=float(d.mean()),mean_paired_rmse_delta_ci95=percentile(d[index].mean(axis=1)),
             success_rate_delta=float((success-bs).mean()),success_rate_delta_ci95=percentile((success-bs)[index].mean(axis=1))))
        for j,p in enumerate(ids):
            paired_rows.append(dict(pair_id=p,condition=condition,strategy=strategy,measure=measure,
                stress_rmse_px=math.sqrt(mse[j]),unperturbed_rmse_px=math.sqrt(bmse[j]),
                paired_rmse_delta_px=float(d[j]),success_rate_delta=float(success[j]-bs[j])))
    return results,differences,paired_rows


def seed_variation(rows):
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        groups[(r['condition'],r['strategy'],r['measure'],r['seed'])][r['pair_id']].append(r)
    scenarios=[]; settings=defaultdict(list)
    for (condition,strategy,measure,seed),pairs in groups.items():
        mse=np.mean([np.mean([r['err']**2 for r in values]) for values in pairs.values()])
        value=dict(condition=condition,strategy=strategy,measure=measure,seed=seed,n_pairs=len(pairs),pooled_rmse_px=float(np.sqrt(mse)))
        scenarios.append(value);settings[(condition,strategy,measure)].append(value['pooled_rmse_px'])
    variability=[]
    for (condition,strategy,measure),v in sorted(settings.items()):
        q=np.percentile(v,[0,25,50,75,100])
        variability.append(dict(condition=condition,strategy=strategy,measure=measure,n_seed_scenarios=len(v),
                           seed_rmse_min=float(q[0]),seed_rmse_q1=float(q[1]),seed_rmse_median=float(q[2]),
                           seed_rmse_q3=float(q[3]),seed_rmse_max=float(q[4]),
                           interpretation='descriptive across seeded cohorts; not independent image samples or confidence intervals'))
    return scenarios,variability


def summarize_diagnostics(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['condition']].append(r)
    result=[]
    for condition,values in sorted(groups.items()):
        out=dict(condition=condition,n_fields=len(values),n_pairs=len({r['pair_id'] for r in values}))
        for metric in ('multiplier_mean','multiplier_variance','upper_clipped_fraction','clipped_mass_mean','input_mean','pre_clip_mean','post_clip_mean'):
            data=[r[metric] for r in values]
            out[metric+'_mean']=float(np.mean(data));out[metric+'_min']=float(np.min(data));out[metric+'_max']=float(np.max(data))
        result.append(out)
    return result


def main():
    protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    completion=json.loads((HERE/'completion.json').read_text(encoding='utf-8'))
    if completion['status']!='complete' or completion['run_signature']!=protocol['run_signature']:
        raise ValueError('A completed matching run is required')
    if sha(HERE/'stress_rows.jsonl')!=completion['result_sha256'] or sha(HERE/'perturbation_diagnostics.json')!=completion['perturbation_sha256']:
        raise ValueError('Source hash mismatch')
    rows=[json.loads(line) for line in (HERE/'stress_rows.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows)!=protocol['expected_metric_rows']:raise ValueError('Wrong result count')
    diagnostics=json.loads((HERE/'perturbation_diagnostics.json').read_text(encoding='utf-8'))
    pairs=aggregate_pairs(rows,protocol)
    summaries,differences,paired_differences=summarize_pairs(pairs,protocol)
    scenarios,variation=seed_variation(rows)
    diag_summary=summarize_diagnostics(diagnostics)
    report=dict(analysis_scope='exploratory normalized-image multiplicative stress; not calibrated SAR speckle',
                source_sha256=completion['result_sha256'],run_signature=protocol['run_signature'],
                independent_resampling_unit='pair identifier; no verified parent-scene independence',
                n_pairs=10,n_offsets=6,seed_replication='30 nested realizations at each k; not independent pairs',
                gamma_shape_interpretation='synthetic severity parameter, not measured sensor looks',
                clipping='clip observed normalized target times Gamma multiplier to [0,1]; no realization min-max',
                bootstrap_replicates=protocol['bootstrap_replicates'],bootstrap_seed=protocol['bootstrap_seed'],
                intervals='percentile pair bootstrap, conditional on the selected pairs and realized seed panel',
                significance_testing='none; supplementary descriptive comparisons do not enter original Holm families',
                aggregation='equal seeds within offsets, then equal offsets within pairs, then equal pair weights',
                pair_metrics=pairs,configuration_summaries=summaries,relative_to_unperturbed=differences,
                paired_differences=paired_differences,seed_scenarios=scenarios,seed_variation=variation,
                perturbation_summaries=diag_summary,baseline_consistency=json.loads((HERE/'baseline_consistency.json').read_text(encoding='utf-8')))
    out=HERE/'statistics';out.mkdir(exist_ok=True)
    (out/'stress_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    write_csv(out/'source_rows.csv',rows);write_csv(out/'perturbation_diagnostics.csv',diagnostics)
    for key in ('pair_metrics','configuration_summaries','relative_to_unperturbed','paired_differences','seed_scenarios','seed_variation','perturbation_summaries'):
        write_csv(out/(key+'.csv'),report[key])
    (out/'analysis_signature.json').write_text(json.dumps(dict(source_sha256=report['source_sha256'],summary_sha256=sha(out/'stress_summary.json'),analysis_source_sha256=sha(__file__)),indent=2),encoding='utf-8')
    print(json.dumps(dict(status='complete',raw_rows=len(rows),pair_summaries=len(pairs),configurations=len(summaries),relative_comparisons=len(differences))))


if __name__=='__main__':main()
