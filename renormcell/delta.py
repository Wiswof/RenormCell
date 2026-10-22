"""Apply ONLY the learned query change, not a new absolute bulk calibration, to a fixed scaffold."""
import numpy as np


def moments(reference,start,trained,measured):
    reference=np.asarray(reference,np.float64);a=np.asarray(start,np.float64);b=np.asarray(trained,np.float64)
    measured=np.asarray(measured,bool)
    if reference.shape!=a.shape or a.shape!=b.shape or a.shape[-2]!=2 or a.shape[-1]!=len(measured):
        raise ValueError('Distinct mean/bulk axes must match')
    for value in (reference,a,b):
        if not np.isfinite(value).all() or (value<0).any() or np.count_nonzero(value[...,~measured]):raise ValueError('Bad probability/mask')
        np.testing.assert_allclose(value.sum(-1),1,rtol=0,atol=6e-7)
    if np.array_equal(a,b):return reference.copy()
    delta=np.log(b.clip(1e-30))-np.log(a.clip(1e-30))
    result=reference*np.exp(delta);result/=result.sum(-1,keepdims=True)
    if not np.isfinite(result).all():raise ValueError('Invalid learned ratio; no silent clipping')
    return result


def counts(template,start,trained,measured,seed):
    from .atlas import dual_moment_counts
    from .rounding import repair_bulk_totals
    from . import atlas as emitter
    from .feasibility import projected_bulk
    x=np.asarray(template)
    if not np.isfinite(x).all() or (x<0).any() or not np.equal(x,np.floor(x)).all() or x[:,~np.asarray(measured,bool)].any():
        raise ValueError('Bad integer scaffold')
    depths=x.sum(1,dtype=np.int64)
    if not (depths>0).all():raise ValueError('Empty scaffold cell')
    reference=np.stack(((x/depths[:,None]).mean(0),x.sum(0,dtype=np.float64)/depths.sum()))
    request=moments(reference,start,trained,measured)
    if np.array_equal(start,trained):
        return x.copy(),dict(zero_delta_exact_scaffold=True,bulk_projection_L1=0.,integer_mean_L1=0.,integer_bulk_L1=0.)
    projected,error=projected_bulk(request[0],request[1],depths)
    emitter.repair_bulk_totals=repair_bulk_totals
    result=dual_moment_counts(x,request[0],request[1],depths=depths,seed=seed,iterations=300)
    np.testing.assert_array_equal(result.sum(1,dtype=np.int64),depths)
    assert not result[:,~np.asarray(measured,bool)].any()
    return result,dict(zero_delta_exact_scaffold=False,bulk_projection_L1=error,
        integer_mean_L1=float(np.abs((result/depths[:,None]).mean(0)-request[0]).sum()),
        integer_bulk_L1=float(np.abs(result.sum(0,dtype=np.float64)/depths.sum()-request[1]).sum()),
        integer_bulk_L1_to_projected=float(np.abs(result.sum(0,dtype=np.float64)/depths.sum()-projected).sum()))

