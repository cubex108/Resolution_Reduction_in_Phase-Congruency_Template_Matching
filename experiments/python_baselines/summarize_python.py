"""Complete denominators, paired descriptive intervals and cross-run audit."""
import json,csv,sys,hashlib
from pathlib import Path
import numpy as np
from run_python_benchmark import ROOT,HERE,canonical_hash,dump,STRATEGIES,sha

def csvwrite(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (HERE/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def main():
    # Re-analysis needs saved observations, not provider images or the execution machine.
    p=json.loads((HERE/'protocol.json').read_text(encoding='utf8'))
    assert p['signature']==canonical_hash({k:v for k,v in p.items() if k!='signature'})
    assert all(sha(ROOT/rel.replace('\\','/'))==value for rel,value in p['source_hashes'].items())
    assert p['primary_protocol_sha256']==sha(ROOT/'results/protocol.json')
    rows=[]
    for pair in p['pairs']:
        obj=json.loads((HERE/'pairs'/(pair['pair_id']+'.json')).read_text(encoding='utf8'))
        assert obj['signature']==p['signature'] and obj['rows_hash']==canonical_hash(obj['rows'])
        assert len(obj['rows'])==66;rows+=obj['rows']
    assert len(rows)==8052
    keys=[(r['pair_id'],r['arm'],r['system'],r['measure'],tuple(r['offset'])) for r in rows]
    assert len(keys)==len(set(keys))
    checks=[]
    # Added zero-translation scalar observations must decode the frozen primary zero-offset maps identically.
    lookup={k:r for k,r in zip(keys,rows)}
    for pair in p['pairs']:
        original=json.loads((ROOT/'results'/'main_pairs'/(pair['pair_id']+'.json')).read_text(encoding='utf8'))
        for r in original:
            if r['strategy'] in STRATEGIES and r['offset']==[0,0]:
                new=lookup[(r['pair_id'],'image_translation',r['strategy'],r['measure'],(0,0))]
                assert new['success3']==r['success3']
                if new['valid']:assert abs(new['err']-r['err'])<1e-12
                checks.append(r['pair_id'])
    assert len(checks)==976
    csvwrite('observations.csv',rows)
    with (HERE/'observations.jsonl').open('w',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    summaries=[];pairs=[];contrasts=[]
    for arm in ['prior_offset','image_translation']:
        for ds in ['SAR','Label','NIR']:
            group=[r for r in rows if r['arm']==arm and r['dataset']==ds]
            systems=sorted({(r['system'],r['measure']) for r in group})
            pids=sorted({r['pair_id'] for r in group});n=len(pids)
            seed=int.from_bytes(hashlib.sha256((arm+ds).encode()).digest()[:8],'little')
            ids=np.random.default_rng(seed).integers(0,n,(10000,n));success_by={}
            for system,measure in systems:
                rr=[r for r in group if r['system']==system and r['measure']==measure]
                assert len(rr)==6*n
                success=[];sumsq=[];validcounts=[]
                for pid in pids:
                    pr=[r for r in rr if r['pair_id']==pid];assert len(pr)==6
                    vv=[r['err'] for r in pr if r['valid']]
                    success.append(np.mean([r['success3'] for r in pr]));sumsq.append(sum(x*x for x in vv));validcounts.append(len(vv))
                    pairs.append(dict(arm=arm,dataset=ds,system=system,measure=measure,pair_id=pid,
                                      success_rate=success[-1],valid=len(vv),trials=6,
                                      valid_rmse=np.sqrt(np.mean(np.square(vv))) if vv else None))
                sr=np.array(success);ss=np.array(sumsq);nv=np.array(validcounts)
                boot_success=sr[ids].mean(1);boot_den=nv[ids].sum(1)
                boot_rmse=np.sqrt(np.divide(ss[ids].sum(1),boot_den,out=np.full(len(ids),np.nan),where=boot_den>0))
                summary=dict(arm=arm,dataset=ds,system=system,measure=measure,n_pairs=n,trials=len(rr),
                             valid=int(nv.sum()),missing=int(len(rr)-nv.sum()),successes=sum(r['success3'] for r in rr),
                             success_rate=float(sr.mean()),success_ci95=np.quantile(boot_success,[.025,.975]).tolist(),
                             valid_rmse=float(np.sqrt(ss.sum()/nv.sum())) if nv.sum() else None,
                             valid_rmse_ci95=np.nanquantile(boot_rmse,[.025,.975]).tolist() if np.any(boot_den>0) else None,
                             rmse_bootstrap_defined=int((boot_den>0).sum()))
                if system=='RIFT_Py':
                    vals=[r['grid_rmse'] for r in rr if r['valid']]
                    summary['valid_grid_rmse']=float(np.sqrt(np.mean(np.square(vals)))) if vals else None
                summaries.append(summary);success_by[(system,measure)]=sr
            if arm=='image_translation':
                for external in ['HOPC_Py','RIFT_Py']:
                    a=success_by[(external,'NCC' if external=='HOPC_Py' else 'affine')]
                    for reference in ['full_pc','mfpc']:
                        d=a-success_by[(reference,'MI')]
                        contrasts.append(dict(dataset=ds,first=external,second=reference+' / MI',n_pairs=n,
                                              delta_success_percentage_points=float(d.mean()*100),
                                              ci95=(np.quantile(d[ids].mean(1),[.025,.975])*100).tolist(),
                                              inference='descriptive unadjusted paired bootstrap; no p value'))
    csvwrite('configuration_summaries.csv',summaries);csvwrite('pair_metrics.csv',pairs);csvwrite('paired_success_contrasts.csv',contrasts)
    out=dict(signature=p['signature'],observation_rows=len(rows),zero_shift_primary_checks=len(checks),
             configuration_summaries=summaries,paired_success_contrasts=contrasts,
             prior_only=dict(rmse=float(np.sqrt(np.mean([y*y+x*x for y,x in p['offsets']]))),success_rate=1/6))
    if (HERE/'timing.json').exists():
        t=json.loads((HERE/'timing.json').read_text(encoding='utf8'));assert t['signature']==p['signature'] and t['complete']
        assert len(t['rows'])==900
        csvwrite('timing_observations.csv',t['rows']);t_summary=[]
        for ds in ['SAR','Label','NIR']:
            for system in p['timing']['systems']:
                medians=[np.median([r['time_s'] for r in t['rows'] if r['pair_id']==pid and r['system']==system])
                         for pid in p['timing']['pairs'] if next(x['dataset'] for x in p['pairs'] if x['pair_id']==pid)==ds]
                t_summary.append(dict(dataset=ds,system=system,pairs=5,repeats_per_pair=10,
                                      median_pair_median_s=float(np.median(medians)),
                                      pair_median_min_s=float(min(medians)),pair_median_max_s=float(max(medians))))
        out['timing_summaries']=t_summary;csvwrite('timing_summaries.csv',t_summary)
    dump(HERE/'summary.json',out)
    print(json.dumps(out,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
