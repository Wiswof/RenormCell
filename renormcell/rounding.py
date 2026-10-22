"""Same stable integer marginal repair; compile only the Python gene/row loops."""
import numpy as np
from numba import njit

@njit(cache=True,nogil=True)
def _redistribute(integer,expected,difference):
    capacity=np.zeros(len(integer),np.int64)
    for gene in np.flatnonzero(difference>0):
        excess=int(difference[gene])
        rounded=integer[:,gene]-np.floor(expected[:,gene]).astype(np.int64)
        rows=np.flatnonzero(rounded>0)
        if rounded[rows].sum()<excess: raise ValueError('Initial rounding fell below its integer floor')
        score=-(integer[rows,gene]-expected[rows,gene])
        rows=rows[np.argsort(score,kind='mergesort')]
        if excess<=len(rows):
            for row in rows[:excess]:
                integer[row,gene]-=1; capacity[row]+=1
        else:
            for row in rows:
                take=min(excess,int(rounded[row])); integer[row,gene]-=take; capacity[row]+=take
                excess-=take
                if not excess: break
    genes=np.flatnonzero(difference<0)
    genes=genes[np.argsort(difference[genes],kind='mergesort')]
    for gene in genes:
        deficit=int(-difference[gene])
        while deficit:
            rows=np.flatnonzero(capacity>0)
            if len(rows)==0: raise AssertionError('Column repair exhausted row capacity')
            take=min(deficit,len(rows)); score=expected[rows,gene]-integer[rows,gene]
            rows=rows[np.argsort(-score,kind='mergesort')]
            for row in rows[:take]:
                integer[row,gene]+=1; capacity[row]-=1
            deficit-=take
    if np.any(capacity): raise AssertionError('Column repair lost counts')
    return integer

def repair_bulk_totals(integer,expected):
    expected=np.asarray(expected,np.float64)
    if integer.shape!=expected.shape or integer.ndim!=2: raise ValueError('Count and expectation axes differ')
    if not np.isfinite(expected).all() or (expected<0).any() or (integer<0).any(): raise ValueError('Invalid count expectations')
    row_totals=integer.sum(1,dtype=np.int64); columns=expected.sum(0)
    target=np.floor(columns).astype(np.int64); remaining=int(row_totals.sum()-target.sum())
    if not 0<=remaining<=len(target): raise ValueError('Expected and integer grand totals differ')
    if remaining: target[np.argsort(columns-target,kind='stable')[-remaining:]]+=1
    result=_redistribute(integer,expected,integer.sum(0,dtype=np.int64)-target)
    if (result<0).any() or not np.array_equal(result.sum(1,dtype=np.int64),row_totals):
        raise AssertionError('Column repair changed a cell depth')
    if not np.array_equal(result.sum(0,dtype=np.int64),target): raise AssertionError('Column repair failed bulk totals')
    return result


