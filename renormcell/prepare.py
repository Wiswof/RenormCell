"""Curated, author-called CRISPRi counts -> matched-control source statistics.

This is NOT an automatic guide caller. Resolve guide labels and study-specific QC first.
"""
import argparse
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from .controls import population_features
from .io import digest, write_json


def prepare(path, genes, output):
    path,output=Path(path),Path(output)
    if output.exists(): raise FileExistsError(f'Refusing to overwrite {output}')
    a=ad.read_h5ad(path)
    required={'study','cell_line','context','target_gene','guide','assay','is_control','quality_pass'}
    if not required <= set(a.obs): raise ValueError(f'Missing author-curated columns: {required-set(a.obs)}')
    if a.obs[list(required)].isna().any().any() or not a.var_names.is_unique or not a.obs_names.is_unique:
        raise ValueError('Missing labels or duplicate gene symbols')
    if not a.obs['assay'].eq('CRISPRi').all(): raise ValueError('Do not mix CRISPRa/KO/unknown assays')
    for key in ('is_control','quality_pass'):
        if a.obs[key].dtype.kind!='b': raise ValueError(f'{key} must be explicit author-curated booleans')
    a=a[a.obs.quality_pass].copy()
    if not len(a): raise ValueError('No retained cells')
    if a.obs[['study','cell_line','context','target_gene','guide']].astype(str).apply(
            lambda col: col.str.strip().eq('').any()).any(): raise ValueError('Empty curated label')
    calls=a.obs.groupby(['study','context','guide'],observed=True)[['target_gene','is_control']].nunique()
    if (calls>1).any().any(): raise ValueError('Conflicting guide-to-target/control labels within a context')
    raw=sparse.csr_matrix(a.X,dtype=np.float64)
    if not np.isfinite(raw.data).all() or (raw.data<0).any() or not np.equal(raw.data,np.floor(raw.data)).all():
        raise ValueError('X must be integer RNA counts, not log-normalized or imputed expression')
    # Missing measured genes are zero-padded but remain FALSE in the mask, never observed zeros.
    present=pd.Index(a.var_names).get_indexer(genes); mask=present>=0
    if 'measured' in a.var:
        if a.var['measured'].dtype.kind!='b': raise ValueError('var.measured must be boolean')
        source_mask=a.var['measured'].to_numpy()
        if raw[:,~source_mask].nnz: raise ValueError('Nonzero count in a gene declared unmeasured')
        mask[mask] &= source_mask[present[mask]]
    if not mask.any(): raise ValueError('No shared genes')
    coo=raw[:,present[mask]].tocoo()
    x=sparse.csr_matrix((coo.data,(coo.row,np.flatnonzero(mask)[coo.col])),shape=(len(a),len(genes)))
    if (np.asarray(x.sum(1)).ravel()<=0).any(): raise ValueError('Zero depth after gene projection')
    records=[];responses=[];controls=[];teachers=[];counts=[]
    axes=['study','cell_line','context']
    for identity, indices in a.obs.groupby(axes,sort=True,observed=True).indices.items():
        obs=a.obs.iloc[indices]; ntc=np.asarray(indices)[obs.is_control.to_numpy()]
        if len(ntc)<30: raise ValueError(f'{identity}: need >=30 matched, author-called NTC cells')
        control=population_features(x[ntc])[[1,0]]
        pert=obs.loc[~obs.is_control]
        for target,barcodes in pert.groupby('target_gene',sort=True,observed=True).groups.items():
            if not str(target) or str(target).lower() in ('unknown','unassigned','control','ntc'):
                raise ValueError('Unknown target must not become a perturbation label')
            rows=a.obs_names.get_indexer(barcodes)
            stats=population_features(x[rows])[[1,0]]
            probability=np.expm1(stats.astype(float))/50000
            probability/=probability.sum(1,keepdims=True)
            responses.append(stats-control);controls.append(control);teachers.append(probability.astype(np.float32));counts.append(len(rows))
            records.append(dict(zip(axes,identity),target_gene=str(target),n_cells=len(rows),
                n_controls=len(ntc),eligible_source=len(rows)>=20))
    if not records: raise ValueError('No target groups')
    output.mkdir(parents=True)
    for name,value in [('response',responses),('controls',controls),('teacher_probability',teachers),('cells',counts)]:
        np.save(output/f'{name}.npy',np.asarray(value))
    np.save(output/'mask.npy',np.broadcast_to(mask,(len(records),len(mask))))
    pd.DataFrame(records).to_csv(output/'groups.csv',index=False)
    write_json(output/'genes.json',list(genes))
    write_json(output/'MANIFEST.json',dict(input_sha256=digest(path),groups=len(records),cells=len(a),
        axis_order=['equal_cell_mean','raw_bulk'],scale=50000,
        raw_library_note='Depth after model-gene projection, not unfiltered whole-transcriptome library size',
        author_calling='Explicit input labels; this script neither infers guides nor reproduces each author caller',
        files={p.name:digest(p) for p in output.iterdir() if p.is_file()}))
    print('PREPARE_COMPLETE',output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='Curated integer-count h5ad; see data/README.md')
    p.add_argument('--genes',required=True,help='gene_names.csv (gene_name column)')
    p.add_argument('--output',required=True)
    args=p.parse_args();prepare(args.input,pd.read_csv(args.genes)['gene_name'].astype(str).tolist(),args.output)


if __name__=='__main__':main()
