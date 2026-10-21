"""Independent mathematical/coordinate checks, before looking at benchmark outcomes."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import hashlib,json,time,warnings
import numpy as np
import cv2
from scipy import ndimage
from baseline_core import *
cv2.setNumThreads(1)
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    rng=np.random.default_rng(481);checks=[]
    def check(name,condition,**extra):
        checks.append(dict(name=name,passed=bool(condition),**extra))
        if not condition:raise AssertionError(checks[-1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from phasepack.phasecong import phasecong
    for shape in [(33,35),(32,36)]:
        a=rng.random(shape)
        for name,params in [('hopc',HOPC_PC),('rift',RIFT_PC)]:
            x=phase_features(a,params);y=phasecong(a,**params)
            de=float(np.max(np.abs(x['eo']-np.asarray(y[5]))))
            dp=float(np.max(np.abs(x['maximum']-y[0])))
            check('independent_phasepack_agreement_'+name+'_'+str(shape),de<1e-11 and dp<1e-10,complex_response_max_abs=de,maximum_moment_max_abs=dp)
    for value in [0.,1.]:
        p=phase_features(np.full((64,64),value),HOPC_PC)
        h=hopc_from_phase(p['pc2'],p['angle'])
        check('constant_'+str(value),np.isfinite(h).all() and np.max(np.abs(h))==0)
        xy,d=rift(np.full((128,128),value));check('constant_rift_'+str(value),len(xy)==len(d)==0)
    a=ndimage.gaussian_filter(rng.random((128,128)),1.)
    h1=hopc(a);h2=hopc(1-a)
    err=float(np.max(np.abs(h1-h2)))
    check('unsigned_orientation_contrast_inversion',err<2e-6,descriptor_max_abs=err)
    # Literal trilinear pixel votes, independent of production convolution kernels.
    mag=rng.random((15,17));angles=rng.random(mag.shape)*np.pi;d=hopc_from_phase(mag,angles)
    ref=np.zeros((3,3,8));yc,xc=7,8
    for iy in range(9):
        for ix in range(9):
            v=angles[yc+iy-4,xc+ix-4]*8/np.pi;lo=int(v);f=v-lo
            for cy in range(3):
                for cx in range(3):
                    dy=iy+.5-(cy+.5)*3;dx=ix+.5-(cx+.5)*3
                    weight=max(1-abs(dy)/3,0)*max(1-abs(dx)/3,0)*np.exp(-(dx*dx+dy*dy)/4.5)*mag[yc+iy-4,xc+ix-4]
                    ref[cy,cx,lo%8]+=weight*(1-f);ref[cy,cx,(lo+1)%8]+=weight*f
    ref=ref.ravel();ref/=np.linalg.norm(ref)
    diff=float(np.max(np.abs(d[yc,xc]-ref)));check('trilinear_histogram_literal_reference',diff<1e-7,max_abs=diff)
    a=rng.random((30,32,4)).astype('float32');b=rng.random(a.shape).astype('float32')
    m=descriptor_ncc_map(a,b,4,13,interval=3);q=a[4:15:3,4:15:3].astype(float).ravel();errs=[]
    for r,c in [(0,0),(3,7),(15,20),(20,22)]:
        if r>=m.shape[0] or c>=m.shape[1]:continue
        v=b[r:r+11:3,c:c+11:3].astype(float).ravel();errs.append(abs(float(np.corrcoef(q,v)[0,1])-m[r,c]))
    check('fft_ncc_matches_direct_vectors',max(errs)<1e-10,max_abs=float(max(errs)))
    mim=rng.integers(1,7,(128,140));points=np.array([[60,60],[3,5],[70,70],[139,0]])
    xy,d=rift_descriptors(mim,points);check('rift_invalid_point_filter',np.array_equal(xy,points[[0,2]]))
    ref=[]
    for orient in range(1,7):
        for col in range(6):
            for row in range(6):
                # MATLAB one-based inclusive endpoints, translated explicitly.
                ya=int(np.floor(row*97/6+1+.5))-1;yb=int(np.floor((row+1)*97/6+.5))
                xa=int(np.floor(col*97/6+1+.5))-1;xb=int(np.floor((col+1)*97/6+.5))
                patch=mim[12:109,12:109];ref.append(np.count_nonzero(patch[ya:yb,xa:xb]==orient))
    ref=np.array(ref,dtype=float);ref/=np.linalg.norm(ref)
    diff=float(np.max(np.abs(d[0]-ref)));check('rift_97px_histogram_matlab_index_reference',diff<1e-7,max_abs=diff)
    points=rng.random((40,2))*100;true=np.array([[1.03,.04,7],[-.02,.97,-4],[0,0,1]])
    target=np.column_stack((points,np.ones(40)))@true[:2].T;target[25:]=rng.random((15,2))*100
    est,meta=robust_affine(points,target,481)
    check('affine_known_transform_with_outliers',est is not None and np.max(np.abs(est-true))<1e-8,**meta)
    est,meta=robust_affine(np.ones((8,2)),np.ones((8,2)),481)
    check('degenerate_affine_explicit_failure',est is None,**meta)
    a=np.array([[0,1,2],[3,4,5],[6,7,8]]);m=translate(a,1,-1)
    check('integer_translation_no_wrap',np.array_equal(m,[[0,0,0],[1,2,0],[4,5,0]]))
    image=unit(ndimage.gaussian_filter(rng.random((240,240)),1.2));target=translate(image,10,-6)
    start=time.perf_counter();da=hopc(image);db=hopc(target);m=descriptor_ncc_map(da,db,96,48)
    peak=np.unravel_index(np.nanargmax(m[40:153,40:153]),(113,113));peak=np.array(peak)+40
    check('hopc_known_translation',np.array_equal(peak,[106,90]),estimated_origin=peak.tolist(),expected=[106,90])
    ra=rift(image);rb=rift(target);est,meta=rift_match(ra,rb,481)
    truth=np.array([[1,0,-6],[0,1,10],[0,0,1]])
    error=None if est is None else float(np.max(np.abs(est-truth)))
    check('rift_known_translation',error is not None and error<.05,transform_max_abs_error=error,**meta)
    check('rift_self_matching',len(match_descriptors(ra[1],ra[1]))==len(ra[1]),keypoints=len(ra[0]))
    out=dict(status='passed',checks=checks,check_count=len(checks),synthetic_feature_seconds=time.perf_counter()-start,
             core_sha256=sha(ROOT/'baseline_core.py'),scope='Mathematical, reference-port and synthetic tests; no native MATLAB or official binary equivalence claim.',
             validation_input_seed=481)
    (ROOT/'validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
