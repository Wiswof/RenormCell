# RenormCell

Source-conditioned, gene-aware perturbation response prediction for VCC 2026.
Two frozen recipes, pretrained weights, compact source statistics, raw-count processing,
and a standalone inference command. No connection to our research workspace is needed.

| Profile | Model/readout | Evidence |
|---|---|---|
| `official-best` | AtlasShift + K448 scaffold; ProgramDual32 learned relative mean/bulk correction | Official validation **0.1592489705417653** |
| `final-candidate` | Above + pooled-source GeneProgramDual64 pass100, full-panel-centered relative **bulk** correction | Exploratory final candidate; **no official final score claimed** |

The candidate is **not** a demonstrated improvement over the official-best profile.
It also extends the Atlas scaffold with qualified K562 evidence for new targets;
the neural priors remain exactly those used during training. These two releases do
**not** use our earlier Flow, autoencoder or PIE experiments.

## One-command deployment

Linux, Python **3.12**, approximately 6 GB free disk for downloaded/extracted assets,
32 GB system RAM recommended. One NVIDIA GPU is enough for neural inference; the
count emitter is CPU-bound. Tested on an A100 80 GB with PyTorch 2.11.0 + CUDA 12.8.
CPU fallback is supported but is slower and is not claimed bit-identical to CUDA.
The first installation additionally downloads PyTorch and dependencies.

```bash
git clone https://github.com/Wiswof/RenormCell.git
cd RenormCell
bash run.sh --profile official-best --data /path/to/panel --output runs/best.h5ad
```

For the other version:

```bash
bash run.sh --profile final-candidate --data /path/to/panel --output runs/candidate.h5ad
```

The script creates an isolated environment, installs pinned dependencies, downloads
the [versioned assets](https://github.com/Wiswof/RenormCell/releases/tag/v1.0.0), checks
SHA256 for every file, and generates predictions. It **never submits anything** or
requests VCC/GitHub credentials. Official controls must be obtained by the user under
the competition's current access rules; they are not redistributed here.

Input directory:

```text
panel/
  gene_names.csv        # gene_name column, exactly the published 18,533-gene order
  pert_counts.csv       # target_gene column, complete requested panel in its original order
  context_A.h5ad        # raw integer control cells, genes in exactly the same order
  context_B.h5ad
  context_C.h5ad        # final panel uses D/E/F instead
```

Each context needs at least 1,600 positive-library controls. Each context/target
produces **400 nonnegative integer-count cells**, with `context` and `target_gene`
in `obs`. The shipped source banks support the **471-target union** of the historical
validation and final panels; unknown targets are rejected rather than guessed.
The original model has 19,790 training target IDs; this compact inference release
does not pretend to contain references for all of them.

Small end-to-end check using your authorized panel (still queries/centers all targets):

```bash
bash run.sh --profile final-candidate --data /path/to/panel \
  --contexts D --limit 1 --workers 1 --output runs/demo.h5ad
```

`--limit` is a demo, **not** a valid submission. Never replace a full panel with a
scoring subset: candidate centering depends on every target. Published A/B/C and
D/E/F control sampling seeds retain their original context indices.

## What runs

```text
raw control population + measured historical target responses + ESM2 target features
  -> independent equal-cell-mean / raw-bulk neural heads
  -> trained-vs-initial relative correction (initial network != plain Atlas)
  -> Atlas/K448 count scaffold
  -> constrained integer count generation
```

The official-best recipe applies `q20/q0` to the scaffold's **actual** two moments.
The candidate then centers `log(q100/q0)` across the full panel and adjusts only raw
bulk. Its normalized mean emerges from the generated counts; it is not independently
forced to an incompatible second target. Original convergence/feasibility guards
remain enabled. A failed guard aborts the run—no target skipping or silent fallback.

Outputs: `.h5ad`, `.queries.npz`, `.diagnostics.csv`, and a `.json` with hashes,
profile, context/target counts and weight fingerprints. Existing outputs are not
overwritten. `COUNTS_READY` is progress; `PREDICTION_COMPLETE` means the file is complete.
Full 900-condition generation is substantially slower than neural prediction;
`--workers` controls bounded CPU concurrency, not additional GPU training.

## Data, processing, reproducibility

- [Data inventory and processing contract](data/README.md), [32-source audit table](data/sources.csv).
- [Pinned raw-file URLs/checksums](data/raw_downloads.json), [training recipes](data/training_recipes.json).
- [Model/score records](MODEL_CARD.md), [portability checks](tests/PORTABILITY.json).
- [Research notes / 研究心得](NOTE.md), [third-party attribution](THIRD_PARTY.md).

```bash
.venv/bin/python -m renormcell.download_raw --study k562 --output raw
.venv/bin/python -m renormcell.download_raw --study k562 --file-index 0 --output raw
.venv/bin/python -m renormcell.prepare --input curated.h5ad \
  --genes /path/to/panel/gene_names.csv --output prepared/my_source
.venv/bin/python -m unittest discover -s tests -v
```

Preprocessing expects **author-curated** single-target CRISPRi calls and matched NTCs.
It does not infer guides from expression, mix CRISPRa/KO, or call normalized data raw
counts. Pretrained inference is turnkey. Rebuilding the complete 29.68-million-cell
training corpus from publisher deposits still requires the study-specific author
calling/QC steps documented in the inventory; the small release banks are **not**
that full training corpus. Exact original checkpoint retraining is not promised by
the generic preprocessing command. Original training budgets/objectives are recorded,
not silently replaced by training on the 2,150 inference references.

Existing Python environment:

```bash
python -m pip install -e .
python -m renormcell.assets --output .assets
python -m renormcell.predict --assets .assets --data /path/to/panel --output runs/prediction.h5ad
```

For CPU-only installation set `TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`
before the first `run.sh`. `RENORM_PYTHON`, `RENORM_VENV`, and `RENORM_ASSETS` customize
the interpreter/environment/cache locations. For Docker, build `docker build -t
renormcell .`, fetch assets on the host, and mount the panel, assets and output:

```bash
docker run --rm --gpus all -v "$PWD/.assets:/assets:ro" \
  -v /path/to/panel:/panel:ro -v "$PWD/runs:/out" renormcell \
  --assets /assets --profile final-candidate --data /panel --output /out/prediction.h5ad
```

Dockerfile supplied; Docker daemon execution was not available in the publishing
environment. The native Python inference path was actually tested.

## License

Our code and model-weight contributions: [MIT](LICENSE). AtlasShift code keeps its
[original MIT notice](licenses/AtlasShift-MIT.txt). Data are **not blanket MIT**:
source-specific terms and attribution apply; see [THIRD_PARTY.md](THIRD_PARTY.md).
No individual-cell training RNA, official controls, secrets or submission tokens are
included. Research software; no guarantee of a competition score or biological accuracy.
