"""Transparent Python reimplementations; no claim of MATLAB/binary equivalence.

Algorithm sources: Ye et al. TGRS 2017, doi:10.1109/TGRS.2017.2656380;
Li et al. TIP 2020, doi:10.1109/TIP.2019.2959244; pinned author MATLAB
release 7ea830e2f13cc3c226f975fe9e98b7666a8f26fb (simplified RIFT).
The phase-congruency construction follows Peter Kovesi's phasecong3 and
is independently checked against PhasePack. See IMPLEMENTATION.md for
parameters, disclosed port decisions, and licensing attribution.
"""
import numpy as np
import cv2
from scipy import fft, signal, ndimage
from functools import lru_cache

HOPC_PC=dict(nscale=4,norient=6,minWaveLength=3,mult=2.1,sigmaOnf=.55,k=2.,cutOff=.5,g=10.)
RIFT_PC=dict(nscale=4,norient=6,minWaveLength=3,mult=1.6,sigmaOnf=.75,k=1.,cutOff=.5,g=3.)

def unit(a):
    a=np.asarray(a,dtype=np.float64)
    r=np.ptp(a)
    return np.zeros_like(a) if r<=1e-14 else (a-a.min())/r

@lru_cache(maxsize=6)
def filter_bank(shape,nscale,norient,minWaveLength,mult,sigmaOnf):
    h,w=shape
    def grid(n):
        return np.arange(-(n-1)/2,(n-1)/2+1)/(n-1) if n%2 else np.arange(-n/2,n/2)/n
    x,y=np.meshgrid(grid(w),grid(h)); r=np.hypot(x,y)
    lowpass=np.fft.ifftshift(1/(1+(r/.45)**30))
    theta=np.fft.ifftshift(np.arctan2(-y,x)); r=np.fft.ifftshift(r);r[0,0]=1
    filters=[]
    for o in range(norient):
        angle=o*np.pi/norient
        delta=np.abs(np.arctan2(np.sin(theta-angle),np.cos(theta-angle)))
        spread=(1+np.cos(np.minimum(delta*norient/2,np.pi)))/2
        scales=[]
        for s in range(nscale):
            radial=np.exp(-np.log(r*minWaveLength*mult**s)**2/(2*np.log(sigmaOnf)**2))*lowpass
            radial[0,0]=0;scales.append(radial*spread)
        filters.append(scales)
    return np.asarray(filters)

def phase_features(image,parameters):
    a=np.asarray(image,dtype=np.float64)
    if a.ndim!=2 or min(a.shape)<3 or not np.isfinite(a).all():raise ValueError('finite scalar image required')
    p=parameters;eps=1e-4
    bank=filter_bank(a.shape,**{k:p[k] for k in ['nscale','norient','minWaveLength','mult','sigmaOnf']})
    im=fft.fft2(a); pcs=[]; amp=[]; weighted=[];cs=[]; oddx=np.zeros_like(a);oddy=oddx.copy();responses=[]
    for o in range(p['norient']):
        eo=fft.ifft2(im[None,:,:]*bank[o],axes=(-2,-1));responses.append(eo)
        amplitudes=np.abs(eo); total=amplitudes.sum(0);cs.append(total)
        e=eo.real.sum(0);v=eo.imag.sum(0); norm=np.hypot(e,v)+eps
        energy=(eo.real*(e/norm)+eo.imag*(v/norm)-np.abs(eo.real*(v/norm)-eo.imag*(e/norm))).sum(0)
        tau=np.median(amplitudes[0])/np.sqrt(np.log(4))
        tau*=sum(p['mult']**(-s) for s in range(p['nscale']))
        threshold=max(tau*(np.sqrt(np.pi/2)+p['k']*np.sqrt((4-np.pi)/2)),eps)
        width=(total/(amplitudes.max(0)+eps)-1)/(p['nscale']-1)
        weight=1/(1+np.exp(p['g']*(p['cutOff']-width)))
        en=weight*np.maximum(energy-threshold,0)
        pc=np.divide(en,total,out=np.zeros_like(en),where=total>0)
        pcs.append(pc);amp.append(total);weighted.append(en)
        oddx+=v*np.cos(o*np.pi/p['norient']);oddy+=v*np.sin(o*np.pi/p['norient'])
    pcs=np.asarray(pcs); angles=np.arange(p['norient'])*np.pi/p['norient']
    cx=pcs*np.cos(angles)[:,None,None];cy=pcs*np.sin(angles)[:,None,None]
    xx=np.sum(cx*cx,0)/(p['norient']/2);yy=np.sum(cy*cy,0)/(p['norient']/2);xy=4*np.sum(cx*cy,0)/p['norient']
    maximum=(xx+yy+np.hypot(xy,xx-yy)+eps)/2
    total=np.sum(amp,0)
    pc2=np.divide(np.sum(weighted,0),total+eps)
    return dict(maximum=maximum,pc=pcs,pc2=pc2,angle=np.mod(np.arctan2(oddy,oddx),np.pi),
                mim=np.argmax(np.asarray(cs),axis=0)+1,eo=np.asarray(responses),bank=bank)

@lru_cache(maxsize=4)
def cell_kernels(cell=3,ncells=3):
    size=cell*ncells; yy,xx=np.mgrid[:size,:size];kernels=[]
    for j in range(ncells):
        for i in range(ncells):
            dy=yy+.5-(j+.5)*cell;dx=xx+.5-(i+.5)*cell
            spatial=np.maximum(1-np.abs(dx)/cell,0)*np.maximum(1-np.abs(dy)/cell,0)
            # Gaussian centred on each cell, sigma=half cell width; explicit choice.
            kernels.append(spatial*np.exp(-(dx*dx+dy*dy)/(2*(cell/2)**2)))
    return kernels

def hopc_from_phase(amplitude,angle,cell=3,ncells=3,bins=8):
    a=np.asarray(amplitude,dtype=np.float64);theta=np.mod(angle,np.pi)*bins/np.pi
    lo=np.floor(theta).astype(int)%bins;frac=theta-np.floor(theta)
    votes=np.stack([a*((lo==b)*(1-frac)+(((lo+1)%bins)==b)*frac) for b in range(bins)],axis=2)
    out=[]
    for ker in cell_kernels(cell,ncells):
        out.extend(cv2.filter2D(votes[:,:,b],cv2.CV_64F,ker,borderType=cv2.BORDER_CONSTANT) for b in range(bins))
    d=np.stack(out,axis=2)
    norms=np.linalg.norm(d,axis=2,keepdims=True)
    d=np.divide(d,norms,out=np.zeros_like(d),where=norms>1e-15)
    return d.astype(np.float32)

def hopc(image):
    p=phase_features(image,HOPC_PC)
    return hopc_from_phase(p['pc2'],p['angle'])

def descriptor_ncc_map(a,b,start,template,interval=5):
    """All valid origins, sparse-centre descriptor NCC; FFT checked against direct NCC."""
    points=np.arange(0,template-interval+1,interval)
    sample=np.asarray(a[start+points[:,None],start+points[None,:]],dtype=np.float64)
    k=int(points[-1]+1);channels=a.shape[2]
    kernel=np.zeros((k,k,channels));kernel[points[:,None],points[None,:]]=sample
    size=(fft.next_fast_len(b.shape[0]+k-1),fft.next_fast_len(b.shape[1]+k-1))
    bf=fft.rfftn(np.asarray(b,dtype=np.float64),s=size,axes=(0,1))
    kf=fft.rfftn(kernel[::-1,::-1,:],s=size,axes=(0,1))
    cross=fft.irfftn(np.sum(bf*kf,axis=2),s=size,axes=(0,1))[k-1:b.shape[0],k-1:b.shape[1]]
    ones=np.zeros((k,k));ones[points[:,None],points[None,:]]=1
    sb=signal.fftconvolve(b.astype(np.float64).sum(2),ones[::-1,::-1],mode='valid')
    sq=signal.fftconvolve(np.square(b.astype(np.float64)).sum(2),ones[::-1,::-1],mode='valid')
    n=sample.size;sa=sample.sum(); va=np.sum(sample*sample)-sa*sa/n
    vb=np.maximum(sq-sb*sb/n,0);den=np.sqrt(np.maximum(va,0)*vb)
    return np.divide(cross-sa*sb/n,den,out=np.full_like(den,np.nan),where=den>1e-12)

def round_matlab(x):return np.floor(np.asarray(x)+.5).astype(int)

def rift_descriptors(mim,points,patch_size=96,norient=6):
    """Author release's inclusive 97x97 patch and Fortran histogram order.
    Invalid-keypoint index typo is corrected by filtering the current keypoint.
    """
    points=round_matlab(points);h,w=mim.shape;r=patch_size//2
    good=(points[:,0]>=r)&(points[:,0]<w-r)&(points[:,1]>=r)&(points[:,1]<h-r)
    points=points[good];des=[]
    n=2*r+1;edges=round_matlab(np.arange(7)*n/6)
    for x,y in points:
        patch=mim[y-r:y+r+1,x-r:x+r+1];d=np.zeros((6,6,norient))
        for j in range(6):
            for i in range(6):
                sub=patch[edges[j]:edges[j+1],edges[i]:edges[i+1]]
                d[j,i]=np.bincount(sub.ravel(),minlength=norient+1)[1:norient+1]
        v=d.ravel(order='F'); norm=np.linalg.norm(v);des.append(v/norm if norm else v)
    return points.astype(np.float64),np.asarray(des,dtype=np.float32).reshape(-1,36*norient)

def rift(image):
    p=phase_features(image,RIFT_PC)
    u=np.floor(unit(p['maximum'])*255+.5).astype(np.uint8)
    detector=cv2.FastFeatureDetector_create(threshold=13,nonmaxSuppression=True,type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    points=sorted(detector.detect(u),key=lambda k:(-k.response,k.pt[1],k.pt[0]))[:5000]
    xy=np.array([k.pt for k in points],dtype=float).reshape(-1,2)
    return rift_descriptors(p['mim'],xy)

def match_descriptors(a,b):
    if len(a)<1 or len(b)<2:return np.empty((0,2),dtype=int)
    matcher=cv2.BFMatcher(cv2.NORM_L2,crossCheck=False);pairs=matcher.knnMatch(a,b,k=2)
    candidates=[];used=set()
    for pair in pairs:
        if len(pair)<2:continue
        first,second=pair
        if first.distance < second.distance and first.trainIdx not in used:
            candidates.append((first.queryIdx,first.trainIdx));used.add(first.trainIdx)
    return np.asarray(candidates,dtype=int).reshape(-1,2)

def affine_fit(x,y):
    d=np.column_stack((x,np.ones(len(x))))
    if len(x)<3 or np.linalg.matrix_rank(d)<3:return None
    p=np.linalg.lstsq(d,y,rcond=None)[0]
    return np.vstack((p.T,[0,0,1]))

def robust_affine(x,y,seed=20261017,max_trials=10000,threshold=2.):
    """FSC-style three-point consensus + least squares, with disclosed repairs.
    Uniform sampling includes last match; collinear samples rejected; seeded RNG.
    Adaptive 0.999 confidence stopping, minimum 50 trials, maximum 10000.
    """
    n=len(x)
    if n<3:return None,dict(status='fewer_than_three_matches',matches=n,inliers=0,trials=0)
    rng=np.random.default_rng(seed);best=np.zeros(n,dtype=bool);best_sum=float('inf');limit=min(max_trials,max(1,n*(n-1)*(n-2)//6));trials=0
    design=np.column_stack((x,np.ones(n)))
    while trials<limit:
        ids=rng.choice(n,3,replace=False);trials+=1
        if abs(np.linalg.det(design[ids]))<1e-8:continue
        h=affine_fit(x[ids],y[ids])
        if h is None:continue
        errors=np.linalg.norm((design@h[:2].T)-y,axis=1);inside=errors<threshold
        count=int(inside.sum());score=float(errors[inside].sum())
        if count>best.sum() or (count==best.sum() and score<best_sum):
            best=inside;best_sum=score
            ratio=count/n
            if ratio>=1:bound=50
            elif ratio>0:bound=max(50,int(np.ceil(np.log(.001)/np.log1p(-ratio**3))))
            else:bound=max_trials
            limit=min(limit,bound)
    h=affine_fit(x[best],y[best])
    if h is None or not np.isfinite(h).all():return None,dict(status='degenerate_geometry',matches=n,inliers=int(best.sum()),trials=trials)
    return h,dict(status='ok',matches=n,inliers=int(best.sum()),trials=trials)

def rift_match(first,second,seed):
    x,a=first;y,b=second;matches=match_descriptors(a,b)
    h,meta=robust_affine(x[matches[:,0]],y[matches[:,1]],seed=seed)
    return h,{**meta,'keypoints_a':len(x),'keypoints_b':len(y)}

def translate(image,dy,dx):
    """Integer translation without resampling; zero-fill, no wraparound."""
    h,w=image.shape;out=np.zeros_like(image)
    ys=slice(max(0,-dy),min(h,h-dy));xs=slice(max(0,-dx),min(w,w-dx))
    out[slice(max(0,dy),min(h,h+dy)),slice(max(0,dx),min(w,w+dx))]=image[ys,xs]
    return out
