"""Subtract the complete requested panel common log ratio, never a scored subset."""
import numpy as np

ARMS=("absolute","full_delta","target_delta","common_delta")

def endpoints(start,trained,arm):
    a=np.asarray(start,np.float64);b=np.asarray(trained,np.float64)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[1]!=2 or len(a)<2:raise ValueError('Need a complete target panel and separate mean/bulk axes')
    for q in (a,b):
        if not np.isfinite(q).all() or (q<0).any():raise ValueError('Bad probability')
        np.testing.assert_allclose(q.sum(-1),1,atol=6e-7,rtol=0)
    if arm=='absolute':return b.copy()
    if arm not in ARMS:raise ValueError(arm)
    if arm=='full_delta':return b.copy()
    if np.array_equal(a,b):return a.copy()  # Actual zero correction must stay bit-exact.
    r=np.log(b.clip(1e-30))-np.log(a.clip(1e-30))
    common=r.mean(0,keepdims=True)  # Compute on all300, NOT the scored local subset.
    value=common if arm=='common_delta' else r-common
    result=a*np.exp(value);result/=result.sum(-1,keepdims=True)
    if not np.isfinite(result).all():raise ValueError('Overflow; no silent correction clipping')
    return result


