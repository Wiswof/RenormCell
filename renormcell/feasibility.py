"""Frozen emitter's exact bulk projection, reported separately from requested and integer moments."""
import numpy as np


def projected_bulk(probability, requested, depths):
    p = np.asarray(probability,dtype=np.float64); bulk = np.asarray(requested,dtype=np.float64).copy()
    depths = np.asarray(depths,dtype=np.int64); z = depths/depths.mean()
    assert p.shape == bulk.shape and np.isfinite(p).all() and np.isfinite(bulk).all()
    assert (p >= 0).all() and (bulk >= 0).all() and (depths > 0).all()
    assert abs(p.sum()-1) < 1e-8 and abs(bulk.sum()-1) < 1e-8
    lower = (z.min()+.01*(1-z.min()))*p; upper = (z.max()-.01*(z.max()-1))*p
    lo,hi = 0.,1.
    if np.ptp(depths) == 0: bulk = p.copy()
    else:
        for _ in range(1024):
            if np.clip(bulk*hi,lower,upper).sum() >= 1: break
            hi *= 2
            if not np.isfinite(hi): raise ValueError('Bulk support cannot satisfy the feasible moment bounds')
        else: raise ValueError('Cannot bracket the bulk projection')
    for _ in range(70):
        mid = (lo+hi)/2
        if np.clip(bulk*mid,lower,upper).sum() < 1: lo = mid
        else: hi = mid
    bulk = np.clip(bulk*((lo+hi)/2),lower,upper)
    error = float(np.abs(bulk-requested).sum())
    if error > .03: raise ValueError(f'Requested bulk projection too large: {error}')
    return bulk,error
