"""Separate, frozen Python-baseline experiment; never changes primary results."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,time,hashlib,argparse,platform,concurrent.futures
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(1)
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'code'))
from imaging_core import extract_feature
from matching_metrics import similarity_maps,summarize_map
from baseline_core import hopc,rift,rift_match,translate,descriptor_ncc_map,HOPC_PC,RIFT_PC
from run_benchmark import sha,canonical_hash,dump,runtime_fingerprint,warmup

OFFSETS=[(0,0),(5,0),(0,5),(10,10),(-20,15),(40,-30)]
STRATEGIES=['raw','full_pc','down_pc','mfpc']
SYSTEMS=STRATEGIES+['HOPC_Py','RIFT_Py']
SEED=20261017

def hashes():
    return {str(p.relative_to(ROOT)):sha(p) for p in
            [HERE/'baseline_core.py',Path(__file__),HERE/'validate_python.py',
             ROOT/'code'/'imaging_core.py',ROOT/'code'/'matching_metrics.py',ROOT/'code'/'run_benchmark.py']}

def freeze():
    if (HERE/'protocol.json').exists():
        print('Already frozen: '+load()['signature']);return
    old=json.loads((ROOT/'results'/'protocol.json').read_text(encoding='utf8'))
    validation=json.loads((HERE/'validation.json').read_text(encoding='utf8'))
    assert all(x['passed'] for x in validation['checks']), 'Validation must pass before freeze'
    warmup()
    p=dict(schema=1,status='Frozen before this experiment result inspection; not preregistered',
           frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),seed=SEED,
           primary_protocol_sha256=sha(ROOT/'results'/'protocol.json'),primary_signature=old['run_signature'],
           pairs=old['pairs'],input_signatures=old['input_signatures'],offsets=OFFSETS,
           arms={'prior_offset':'HOPC_Py only, original shifted-search protocol; 732 additional rows',
                 'image_translation':'Zero-filled integer translation of complete target work image; fixed centred search; 10 outputs per pair-shift; 7320 rows'},
           scalar_strategies=STRATEGIES,scalar_measures=['MI','CCRE'],grid_step=2,bins=32,
           HOPC=dict(pc=HOPC_PC,cell_pixels=3,block_cells=3,bins=8,gaussian_sigma_pixels=1.5,
                     block_normalization='L2; no clipping',block_interval=5,
                     matching='global-vector NCC; full-work descriptors; sparse block centres; nominal T admissibility'),
           RIFT=dict(pc=RIFT_PC,scope='simplified no-rotation author release, Python reimplementation',
                     fast='OpenCV FAST 9-16; threshold 13; nonmax; top 5000',patch_parameter=96,
                     patch_actual_pixels=97,cells=[6,6],orientation_bins=6,ratio=1.,
                     consensus='seeded 3-point affine; threshold 2 px; confidence .999; min 50/max 10000; refit LS'),
           truth='Known added translation relative to nominal dataset alignment; no independent point truth',
           endpoint='Euclidean error at central template centre, (W-1)/2; success <3 px with valid output',
           rift_extra_endpoint='Affine error on 5x5 grid over source region still in target; not fit points',
           failures='All 732 pair-shifts per system retained; nonfinite/flat/unavailable outputs are failed success trials. RMSE conditional on valid output, with denominator. No imputation.',
           inference='Pair is resampling unit; six shifts are repeated measures. 10000 paired bootstrap percentile CIs; descriptive, unadjusted; no new hypothesis-test family. SAR parent-scene grouping unknown.',
           timing=dict(pairs=[p['pair_id'] for d in ['SAR','Label','NIR'] for p in [q for q in old['pairs'] if q['dataset']==d][:5]],
                       repeats=10,warmup=3,offset=[10,10],systems=SYSTEMS,threads=1,
                       scope='Serial two-image feature extraction, matching and decoding. Scalar systems jointly compute MI/CCRE; descriptor systems compute one matcher. In-memory; I/O, translation, JIT excluded.',
                       order='Seeded shuffled system order each repetition; technical repeats, not extra image pairs'),
           limitations=['HOPC descriptor plus common search, not complete original HOPC registration pipeline',
                        'RIFT FAST and consensus are disclosed substitutes; no MATLAB parity or rotation claim',
                        'RIFT uses global keypoints/affine geometry; scalar and HOPC use local translation search; system-level comparison',
                        'Zero-filled boundaries and changed overlap are part of translated-input stress test; no natural geometric ground truth'],
           runtime=runtime_fingerprint(),source_hashes=hashes(),validation_sha256=sha(HERE/'validation.json'))
    p['signature']=canonical_hash(p);dump(HERE/'protocol.json',p)
    print('FROZEN '+p['signature'],flush=True)

def load(inputs=False):
    p=json.loads((HERE/'protocol.json').read_text(encoding='utf8'))
    assert p['signature']==canonical_hash({k:v for k,v in p.items() if k!='signature'})
    assert p['source_hashes']==hashes(),'Execution code changed after freeze'
    assert p['primary_protocol_sha256']==sha(ROOT/'results'/'protocol.json')
    assert p['runtime']==runtime_fingerprint(),'Runtime changed after freeze'
    if inputs:
        for x in p['input_signatures']:
            assert sha(ROOT/'results'/'crops'/(x['pair_id']+'.npz'))==x['npz_sha256']
    return p

def record(p,arm,system,measure,offset,err,valid,**extra):
    return dict(dataset=p['dataset'],pair_id=p['pair_id'],arm=arm,system=system,measure=measure,
                offset=list(offset),err=float(err) if valid else None,valid=bool(valid),
                success3=bool(valid and err<3.),**extra)

def ncc_record(p,arm,offset,fullmap):
    W,T,S=p['geometry'];ts=(W-T)//2;s=(W-S)//2;dy,dx=offset
    sr,sc=(s+dy,s+dx) if arm=='prior_offset' else (s,s)
    rows=np.arange(0,S-T+1,2);cols=rows.copy()
    score=fullmap[sr+rows[:,None],sc+cols[None,:]]
    truth=(ts-sr,ts-sc) if arm=='prior_offset' else (ts-sr+dy,ts-sc+dx)
    finite=np.isfinite(score)
    if not finite.any():return record(p,arm,'HOPC_Py','NCC',offset,None,False,status='undefined_NCC')
    peak=np.nanmax(score);low=np.nanmin(score)
    if peak-low<=max(abs(peak),abs(low),1e-300)*1e-12:
        return record(p,arm,'HOPC_Py','NCC',offset,None,False,status='flat_NCC')
    iy,ix=np.unravel_index(np.argmax(np.where(finite,score,-np.inf)),score.shape)
    estimate=[int(rows[iy]),int(cols[ix])];err=np.linalg.norm(np.array(estimate)-truth)
    return record(p,arm,'HOPC_Py','NCC',offset,err,True,status='ok',estimated_top_left=estimate,
                  true_top_left=list(truth),valid_candidates=int(finite.sum()),peak=float(peak))

def scalar_records(p,offset,strategy,fa,fb):
    W,T,S=p['geometry'];ts=(W-T)//2;s=(W-S)//2;dy,dx=offset
    maps=similarity_maps(fa[ts:ts+T,ts:ts+T],fb[s:s+S,s:s+S],step=2,bins=32,
                         template_range=(0,1),search_range=(0,1))
    out=[]
    for m in ['MI','CCRE']:
        info=summarize_map(maps[m],maps['rows'],maps['cols'],(ts-s+dy,ts-s+dx),(T,T))
        out.append(record(p,'image_translation',strategy,m,offset,info['err'],not info['degenerate'],
                          status='flat_map' if info['degenerate'] else 'ok',
                          estimated_top_left=info['estimated_top_left'],true_top_left=info['true_top_left']))
    return out

def rift_record(p,offset,first,second):
    payload=(p['pair_id']+str(tuple(offset))+str(SEED)).encode()
    seed=int.from_bytes(hashlib.sha256(payload).digest()[:8],'little')
    h,meta=rift_match(first,second,seed)
    if h is None:return record(p,'image_translation','RIFT_Py','affine',offset,None,False,**meta)
    W=p['geometry'][0];dy,dx=offset;c=(W-1)/2
    pred=h@np.array([c,c,1.]);err=np.linalg.norm(pred[:2]-[c+dx,c+dy])
    xx=np.linspace(max(0,-dx),min(W-1,W-1-dx),5);yy=np.linspace(max(0,-dy),min(W-1,W-1-dy),5)
    x,y=np.meshgrid(xx,yy);grid=np.column_stack((x.ravel(),y.ravel(),np.ones(25)))
    residual=grid@h[:2].T-(grid[:,:2]+[dx,dy])
    return record(p,'image_translation','RIFT_Py','affine',offset,err,True,affine=h.tolist(),
                  grid_rmse=float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),**meta)

def one_pair(arg):
    p,signature=arg;protocol=load();assert protocol['signature']==signature
    target=HERE/'pairs'/(p['pair_id']+'.json')
    if target.exists():
        old=json.loads(target.read_text(encoding='utf8'))
        assert old['signature']==signature and old['rows_hash']==canonical_hash(old['rows'])
        return p['pair_id'],'cached'
    checksum=next(x['npz_sha256'] for x in protocol['input_signatures'] if x['pair_id']==p['pair_id'])
    source=ROOT/'results'/'crops'/(p['pair_id']+'.npz');assert sha(source)==checksum
    warmup();start=time.perf_counter()
    with np.load(source,allow_pickle=False) as f:a,b=f['a'],f['b']
    W,T,S=p['geometry'];ts=(W-T)//2
    ha=hopc(a);hb=hopc(b);ra=rift(a);rb=rift(b)
    full=descriptor_ncc_map(ha,hb,ts,T)
    rows=[ncc_record(p,'prior_offset',offset,full) for offset in OFFSETS]
    source_features={s:extract_feature(a,s) for s in STRATEGIES}
    for offset in OFFSETS:
        dy,dx=offset;target_image=translate(b,dy,dx)
        if dy or dx:
            hb=hopc(target_image);rb=rift(target_image);full=descriptor_ncc_map(ha,hb,ts,T)
        rows.append(ncc_record(p,'image_translation',offset,full))
        rows.append(rift_record(p,offset,ra,rb))
        for s in STRATEGIES:
            rows.extend(scalar_records(p,offset,s,source_features[s],extract_feature(target_image,s)))
    assert len(rows)==66
    dump(target,dict(signature=signature,rows_hash=canonical_hash(rows),rows=rows,
                     diagnostic_parallel_wall_seconds=time.perf_counter()-start))
    return p['pair_id'],round(time.perf_counter()-start,2)

def main(workers):
    p=load(inputs=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one_pair,(pair,p['signature'])) for pair in p['pairs']]
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            print(i,len(futures),*f.result(),flush=True)

def timed_query(p,a,b,system):
    W,T,S=p['geometry'];ts=(W-T)//2
    if system=='HOPC_Py':
        fa,fb=hopc(a),hopc(b);return ncc_record(p,'image_translation',(10,10),descriptor_ncc_map(fa,fb,ts,T))
    if system=='RIFT_Py':return rift_record(p,(10,10),rift(a),rift(b))
    return scalar_records(p,(10,10),system,extract_feature(a,system),extract_feature(b,system))

def timing():
    protocol=load(inputs=True);warmup();rng=np.random.default_rng(SEED);rows=[]
    for pid in protocol['timing']['pairs']:
        p=next(x for x in protocol['pairs'] if x['pair_id']==pid)
        with np.load(ROOT/'results'/'crops'/(pid+'.npz'),allow_pickle=False) as f:a,b=f['a'],translate(f['b'],10,10)
        for _ in range(3):
            for s in SYSTEMS:timed_query(p,a,b,s)
        for rep in range(10):
            for s in rng.permutation(SYSTEMS):
                t=time.perf_counter();timed_query(p,a,b,str(s));elapsed=time.perf_counter()-t
                rows.append(dict(dataset=p['dataset'],pair_id=pid,system=str(s),repeat=rep,time_s=elapsed))
        dump(HERE/'timing.json',dict(signature=protocol['signature'],complete=len(rows)==900,rows=rows))
        print('timing',pid,len(rows),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','main','timing']);parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='main':main(args.workers)
    else:timing()
