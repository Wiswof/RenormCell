"""Sparse streaming count writer and integrity helpers."""
import hashlib
import json
from pathlib import Path
import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

class CountWriter:
    """Append integer count blocks; retain int64 CSR pointers beyond 2^31 NNZ."""

    def __init__(self, path, targets, genes, contexts, cells):
        self.path = Path(path)
        self.rows = 0
        self.nnz = 0
        self.nobs = len(targets) * len(contexts) * cells
        self.ngenes = len(genes)
        obs = pd.DataFrame(
            {
                "target_gene": np.tile(np.repeat(targets, cells), len(contexts)),
                "context": np.repeat(contexts, len(targets) * cells),
            },
            index=[f"{c}_{t}_{i}" for c in contexts for t in targets for i in range(cells)],
        )
        ad.AnnData(
            sparse.csr_matrix((self.nobs, len(genes)), dtype=np.float32),
            obs=obs,
            var=pd.DataFrame(index=genes),
        ).write_h5ad(self.path)
        self.file = h5py.File(self.path, "r+")
        group = self.file["X"]
        for name in ["data", "indices", "indptr"]:
            del group[name]
        self.data = group.create_dataset(
            "data",
            shape=(0,),
            maxshape=(None,),
            dtype="float32",
            chunks=(1048576,),
            compression="lzf",
            shuffle=True,
        )
        self.indices = group.create_dataset(
            "indices",
            shape=(0,),
            maxshape=(None,),
            dtype="int32",
            chunks=(1048576,),
            compression="lzf",
            shuffle=True,
        )
        self.pointers = group.create_dataset("indptr", shape=(self.nobs + 1,), dtype="int64")
        self.pointers[0] = 0

    def append(self, counts):
        if (
            counts.ndim != 2
            or counts.shape[1] != self.ngenes
            or self.rows + len(counts) > self.nobs
        ):
            raise ValueError("Unexpected count block shape")
        if (
            not np.isfinite(counts).all()
            or (counts < 0).any()
            or (counts != np.floor(counts)).any()
        ):
            raise ValueError("Expected finite nonnegative integer counts")
        if (counts.sum(axis=1, dtype=np.float64) > 1000000).any():
            raise ValueError("Cell depth exceeds 1,000,000")
        block = sparse.csr_matrix(counts.astype(np.uint32))
        end = self.nnz + block.nnz
        if end > 4750000000:
            raise ValueError("Sparse element limit exceeded")
        self.data.resize((end,))
        self.indices.resize((end,))
        self.data[self.nnz : end] = block.data
        self.indices[self.nnz : end] = block.indices
        self.pointers[self.rows + 1 : self.rows + len(counts) + 1] = (
            block.indptr[1:].astype(np.int64) + self.nnz
        )
        self.rows += len(counts)
        self.nnz = end

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        complete = self.rows == self.nobs and int(self.pointers[-1]) == self.nnz
        self.file.close()
        if kind is None and (not complete):
            raise ValueError("Incomplete prediction output")



def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")

