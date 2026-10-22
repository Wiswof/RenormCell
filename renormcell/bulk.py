"""One coherent constraint: learned raw bulk; normalized mean emerges from actual cells."""
import numpy as np
from .rounding import repair_bulk_totals
from .delta import moments


def counts(template,start,trained,measured,seed):
    original=np.asarray(template);mask=np.asarray(measured,bool)
    if original.ndim!=2 or not np.isfinite(original).all() or (original<0).any() or not np.equal(original,np.floor(original)).all():
        raise ValueError('Need finite nonnegative integer cell counts')
    if original[:,~mask].any():raise ValueError('Nonzero unmeasured gene')
    depths=original.sum(1,dtype=np.int64)
    if not (depths>0).all():raise ValueError('Empty scaffold cell')
    reference=np.stack(((original/depths[:,None]).mean(0),original.sum(0,dtype=float)/depths.sum()))
    requested=moments(reference,start,trained,mask);b=requested[1]
    if np.array_equal(start[1],trained[1]):
        return original.copy(),dict(zero_delta_exact_scaffold=True,integer_bulk_L1=0.,normalized_mean_not_forced=True)
    # Positive measured support allows row/column matrix balancing. This is NOT dual-moment fitting.
    x=.999*original.astype(float)+.001*depths[:,None]*b
    desired=b*depths.sum();error=np.inf
    for step in range(1000):
        column=x.sum(0);x*=np.divide(desired,column,out=np.zeros_like(column),where=column>0)
        x*=depths[:,None]/x.sum(1,keepdims=True)
        if step%5==4:
            error=float(abs(x.sum(0)/depths.sum()-b).sum())
            if error<1e-8:break
    if error>=1e-8:raise ValueError(f'Raw-bulk balancing did not converge: {error}; no tolerance relaxation')
    result=np.floor(x).astype(np.int32);fractions=x-result;rng=np.random.default_rng(seed)
    for i,depth in enumerate(depths):
        remaining=int(depth-result[i].sum())
        if remaining:
            cumulative=np.cumsum(fractions[i]);cumulative*=remaining/cumulative[-1]
            where=np.searchsorted(cumulative,np.arange(remaining)+rng.random(),side='right')
            np.add.at(result[i],where,1)
    result=repair_bulk_totals(result,x)
    assert (result>=0).all() and not result[:,~mask].any()
    np.testing.assert_array_equal(result.sum(1,dtype=np.int64),depths)
    actual_bulk=result.sum(0,dtype=float)/depths.sum();actual_mean=(result/depths[:,None]).mean(0)
    detail=dict(zero_delta_exact_scaffold=False,normalized_mean_not_forced=True,bulk_balancing_iterations=step+1,
        fractional_bulk_L1=error,integer_bulk_L1=float(abs(actual_bulk-b).sum()),
        realized_mean_L1_from_network_requested_mean=float(abs(actual_mean-requested[0]).sum()),
        realized_mean_L1_from_reference=float(abs(actual_mean-reference[0]).sum()),support_smoothing=.001)
    return result,detail

