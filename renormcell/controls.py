"""Raw control populations; normalized mean and raw bulk remain separate."""
import hashlib
from pathlib import Path
import anndata as ad
import numpy as np
from scipy import sparse

def population_features(raw):
    raw=raw.tocsr().astype(np.float64);raw.sum_duplicates();raw.eliminate_zeros()
    if not np.isfinite(raw.data).all() or (raw.data<0).any() or not np.equal(raw.data,np.floor(raw.data)).all():
        raise ValueError('Controls must be finite nonnegative integer counts')
    depth=np.asarray(raw.sum(1)).ravel()
    if not len(depth) or (depth<=0).any():raise ValueError('Empty/zero-library control cells')
    p=raw.multiply(1/depth[:,None]).tocsr();mean=np.asarray(p.mean(0)).ravel()
    bulk=np.asarray(raw.sum(0)).ravel()/depth.sum()
    variance=np.maximum(np.asarray(p.power(2).mean(0)).ravel()-mean**2,0)
    return np.stack((np.log1p(50000*bulk),np.log1p(50000*mean),
                     np.log1p(50000**2*variance),p.getnnz(axis=0)/len(depth))).astype(np.float32)



class Decoder:
    def __init__(self, controls):
        self.x = sparse.csr_matrix(controls.X, dtype=np.float64)
        depth = np.asarray(self.x.sum(1)).ravel()
        self.cpm = self.x.multiply((1_000_000 / depth)[:, None]).tocsr()
        self.mean = np.asarray(self.cpm.mean(0)).ravel()
        self.gate = np.flatnonzero(self.mean > 5)
        columns = self.cpm[:, self.gate].tocsc()
        self.sorted = [
            np.sort(columns.data[columns.indptr[i]:columns.indptr[i + 1]])
            for i in range(len(self.gate))
        ]
        self.zeros = np.asarray([self.x.shape[0] - len(values) for values in self.sorted])

    def psi(self, gate_index, values):
        control = self.sorted[gate_index]
        zeros = self.zeros[gate_index]
        lo = np.searchsorted(control, values, "left")
        hi = np.searchsorted(control, values, "right")
        result = zeros + lo + .5 * (hi - lo)
        result[values == 0] = .5 * zeros
        return result


def control_template(path, genes, cells, pool_k, seed):
    controls = ad.read_h5ad(path)
    if not np.array_equal(controls.var_names, genes):
        raise ValueError("Control gene order differs from gene_names.csv")
    raw = sparse.csr_matrix(controls.X, dtype=np.float64)
    if (
        not np.isfinite(raw.data).all()
        or (raw.data < 0).any()
        or (raw.data != np.floor(raw.data)).any()
    ):
        raise ValueError("Controls must contain raw nonnegative integer counts")
    library = np.asarray(raw.sum(axis=1)).ravel()
    if (library <= 0).any() or len(library) < cells * pool_k:
        raise ValueError(
            "Need positive-depth controls and at least cells_per_target * pool_k donors"
        )
    mean = np.zeros(len(genes))
    for left in range(0, len(library), 256):
        block = raw[left : left + 256].toarray() / library[left : left + 256, None]
        mean += block.sum(axis=0)
    mean /= len(library)
    bulk = np.asarray(raw.sum(axis=0)).ravel()
    bulk /= bulk.sum()
    selected = np.random.default_rng(seed).choice(len(library), cells * pool_k, replace=False)
    selected = selected[np.argsort(library[selected], kind="stable")]
    template = raw[selected].toarray() / library[selected, None]
    template = template.reshape(cells, pool_k, len(genes)).mean(axis=1)
    depths = np.rint(library[selected].reshape(cells, pool_k).mean(axis=1)).astype(np.int64)
    if depths.min() < 1 or depths.max() > 1000000:
        raise ValueError("Invalid predicted library depths")
    return (template, depths, mean, bulk)



def shape_seed(label):
    return 20260928+int.from_bytes(hashlib.sha256(label.encode()).digest()[:4],"little")

