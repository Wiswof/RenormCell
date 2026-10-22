"""Two frozen recipes: neural moments -> Atlas/K448 -> integer counts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import anndata as ad
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch
from . import atlas, bulk, delta, shape
from .controls import Decoder, control_template, population_features, shape_seed
from .inputs import Inputs
from .io import CountWriter, digest, write_json
from .readout import endpoints


def log(message): print(time.strftime('%Y-%m-%d %H:%M:%S'),message,flush=True)


def requested_moments(root, targets, mean, raw_bulk, extended):
    axes = json.loads((root/'atlas/axes.json').read_text())
    ids = np.array([axes['targets'].index(t) for t in targets])
    with np.load(root/'atlas/cd4.npz') as z: cd4,available = z['effect'][ids],z['available'][ids]
    result = []
    for space,control,amplitude in [('log2fc',mean,.6),('bulk_delta',raw_bulk,.3)]:
        suffix = '_extended' if extended else ''
        with np.load(root/f'atlas/{space}{suffix}.npz') as z:
            effect,coverage = z['effects'][ids],z['coverage'][ids]
        effect,_ = atlas.add_cd4_family(effect,coverage,cd4,available,.5,
            space=space,control_probability=control)
        p = atlas.desired_mean(control,effect,space=space,amplitude=amplitude,clip=3)
        p,_ = atlas.apply_promoter_prior(p,control,targets,axes['genes'],root/'atlas/promoters.csv',.15)
        result.append(p)
    return result


def predict(args):
    root, panel, output = Path(args.assets),Path(args.data),Path(args.output)
    if output.exists(): raise FileExistsError(f'Refusing to overwrite {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    axes = json.loads((root/'atlas/axes.json').read_text())
    genes = pd.read_csv(panel/'gene_names.csv')['gene_name'].astype(str).to_numpy()
    targets = pd.read_csv(panel/'pert_counts.csv')['target_gene'].astype(str).to_numpy()
    if genes.tolist() != axes['genes']: raise ValueError('Expected the frozen 18,533-gene order')
    if len(set(targets)) != len(targets) or not set(targets) <= set(axes['targets']):
        raise ValueError('Unknown/repeated target: see data/README.md to extend the bank')
    if len(targets)<2: raise ValueError('Provide the whole panel, not one target (centering depends on it)')
    if args.workers<1 or args.limit is not None and args.limit<1: raise ValueError('Invalid worker/limit count')
    contexts = args.contexts or [p.stem.removeprefix('context_') for p in sorted(panel.glob('context_*.h5ad'))]
    if not contexts or len(set(contexts)) != len(contexts): raise ValueError('Missing/repeated contexts')
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(args.threads); torch.set_float32_matmul_precision('high')
    queries = dict(genes=genes,targets=targets)
    controls = {}
    for context in contexts:
        a = ad.read_h5ad(panel/f'context_{context}.h5ad')
        if a.var_names.tolist() != genes.tolist(): raise ValueError(f'{context}: wrong control gene axis')
        measured = a.var['measured'].to_numpy(bool) if 'measured' in a.var else np.ones(len(genes),bool)
        if not measured.all(): raise ValueError('Published count recipes require the complete measured gene panel')
        controls[context] = (population_features(a.X),measured)
        del a
    recipes = [('protected','protected',('protected0','protected20'))]
    if args.profile == 'final-candidate': recipes.append(('gene','gene',('gene0','gene100')))
    for _,bank,checkpoints in recipes:
        reader = Inputs(root,bank,device)
        for name in checkpoints:
            for context in contexts:
                values = reader.predict(root/f'weights/{name}.pt',*controls[context],targets)
                queries.update({f'{name}_{context}_{k}':v for k,v in values.items()})
                log(f'QUERY_READY {name} context={context} targets={len(targets)} device={device}')
        del reader
        if device.startswith('cuda'): torch.cuda.empty_cache()
    query_path = output.with_suffix('.queries.npz'); np.savez(query_path,**queries)
    chosen = targets[:args.limit] if args.limit else targets
    details = []
    shape.repair_exact_columns = shape.repair_exact_columns_fast
    with threadpool_limits(limits=1), CountWriter(output,chosen,genes,contexts,400) as writer:
        for ci,context in enumerate(contexts):
            path = panel/f'context_{context}.h5ad'
            a = ad.read_h5ad(path); decoder = Decoder(a); del a
            seed_index = dict(A=0,B=1,C=2,D=0,E=1,F=2).get(context,ci)
            template,depths,mean,raw_bulk = control_template(path,genes,400,4,20260910+seed_index)
            wanted = requested_moments(root,targets,mean,raw_bulk,args.profile=='final-candidate')
            mask = controls[context][1]
            end = endpoints(queries[f'gene0_{context}_probability'],queries[f'gene100_{context}_probability'],
                            'target_delta') if args.profile=='final-candidate' else None

            def generate(i):
                t = str(targets[i]); seed = 20261017+i+1
                raw = atlas.dual_moment_counts(template,wanted[0][i],wanted[1][i],depths=depths,
                    seed=atlas.seed_for(context+':'+t,20260910))
                x,error = shape.shape_group(raw.astype(float),decoder,448,
                    shape_seed(f'official:{context}:{t}'),dual_moment=True,flip_direction=True)
                x,detail = delta.counts(x,queries[f'protected0_{context}_probability'][i],
                    queries[f'protected20_{context}_probability'][i],mask,seed)
                if end is not None:
                    x,extra = bulk.counts(x,queries[f'gene0_{context}_probability'][i],end[i],mask,seed)
                    detail.update(extra)
                expected = depths*max(1,int(1_000_000//depths.max()))
                np.testing.assert_array_equal(x.sum(1,dtype=np.int64),expected)
                return x,dict(context=context,target=t,shape_mean_L1=error,**detail)

            # Bound outstanding dense blocks; never retain all 300 generated populations in RAM.
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for first in range(0,len(chosen),args.workers):
                    for x,detail in pool.map(generate,range(first,min(first+args.workers,len(chosen)))):
                        writer.append(x); details.append(detail)
                        log(f'COUNTS_READY context={context} target={detail["target"]} '
                            f'groups={len(details)}/{len(chosen)*len(contexts)} cells=400')
    pd.DataFrame(details).to_csv(output.with_suffix('.diagnostics.csv'),index=False)
    write_json(output.with_suffix('.json'),dict(profile=args.profile,contexts=contexts,targets=len(chosen),
        cells_per_target=400,genes=len(genes),complete_panel=args.limit is None,
        prediction_sha256=digest(output),query_sha256=digest(query_path),device=device,
        weights={name:digest(root/f'weights/{name}.pt') for _,_,names in recipes for name in names},
        note='No automatic submission. A limited demo is NOT a competition submission.'))
    log(f'PREDICTION_COMPLETE {output}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=('official-best','final-candidate'),default='official-best')
    p.add_argument('--assets',default='.assets')
    p.add_argument('--data',required=True,help='gene_names.csv, pert_counts.csv, context_<label>.h5ad')
    p.add_argument('--output',required=True)
    p.add_argument('--contexts',nargs='+',help='Explicit order controls the published sampling seeds')
    p.add_argument('--device',help='cuda, cuda:0, or cpu; default: CUDA when available')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--limit',type=int,help='Demo only; query/centering still use the FULL panel')
    predict(p.parse_args())


if __name__ == '__main__': main()
