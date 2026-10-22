# Data and processing

## Inference: no training-data download required

`python -m renormcell.assets` retrieves four SHA256-pinned release archives.
They contain stripped model weights, aggregate source statistics, measured-gene
masks, target features and frozen Atlas effects. No cell-level historical RNA,
barcodes, official control populations or hidden truth is redistributed.

`assets.json` pins every archive and the per-file `FILES.json` manifest. The
uncompressed inference assets are approximately 2.2 GB; compressed archives are
listed with exact sizes in the manifest. `atlas/axes.json` defines gene order,
supported targets, original target IDs and the native-prior target subset.
Each source bank has 2,150 referenced rows; the remapping preserves source order,
responses and the exact appropriate matched controls. `sources.csv` inside each
archive maps those rows back to biological study/context/target metadata.

The two neural axes are **[equal-cell mean, raw bulk]**. Response arrays are
`log1p(50000 * target_probability) - log1p(50000 * matched_control_probability)`.
Controls are the corresponding `log1p` probabilities, not normalized single cells.

## Publisher inventory

[sources.csv](sources.csv) records all 32 audited study identities, original links,
authors/PubMed IDs when available, CRISPRi/guide/QC scope, TRAIN group/cell totals,
rights statements and captured primary-metadata hashes. **27** identities have
nonzero retained contributions to the released checkpoints. The others are listed
for audit completeness, not counted as extra training data.

Atlas scaffold prerequisites are separately listed in [raw_downloads.json](raw_downloads.json):
K562 GWPS, HCT116, HEK293T, public H1 2025, CD4 differential-expression statistics,
GENCODE v47. HCT116/HEK293T statistics originate from the public
[X-Atlas-Orion collection](https://huggingface.co/datasets/slaf-project/X-Atlas-Orion).
The scaffold retains original full-source common-response centering; it is not
recentered on only the queried target subset. ESM2 features are precomputed target
sequence features, not a learned control-population encoder.

Optional raw file discovery/download:

```bash
python -m renormcell.download_raw --study GSE281048 --output raw
python -m renormcell.download_raw --study k562 --file-index 0 --output raw
```

GEO supplementary file indices include metadata and mixed experimental arms; being
listed does **not** approve the whole deposit for training. Apply the per-study
scope in `sources.csv`. Known pinned publisher SHA256 values are checked; if a
publisher hash was not captured, the downloaded-file receipt explicitly says so
and records the observed SHA256. A metadata SHA is not a raw RNA checksum.
Public source licenses remain source-specific; see [THIRD_PARTY.md](../THIRD_PARTY.md).

## Processing your own curated counts

`renormcell.prepare` is deliberately small. Input AnnData:

- `X`: finite nonnegative **integer RNA counts**; unique cells and gene symbols.
- `obs`: `study`, `cell_line`, `context`, `target_gene`, `guide`, `assay`,
  `is_control`, `quality_pass`. The last two must be actual booleans.
- `assay` must be `CRISPRi`; guides/targets/QC must already follow publisher calls.
- `context` must encode every relevant donor, treatment, dose, replicate, lane and
  library identity needed for matched controls. Do not merge incompatible batches.
- `is_control` means explicit same-state author NTC, **not** unassigned/safe-targeting/
  inferred nonresponders. Each context needs >=30 retained NTCs.
- Distinct-target combinations, CRISPRa/KO arms, ambiguous calls, duplicate cells,
  protein/ATAC layers and reused normalized representations must be excluded first.

```bash
python -m renormcell.prepare --input curated.h5ad \
  --genes /path/to/panel/gene_names.csv --output prepared/study
```

The script aligns genes, keeps an explicit missing-gene mask, builds matched-control
dual statistics and target teachers, and writes group metadata/checksums. Groups
with fewer than20 cells can remain teachers but are not eligible reference sources.
Library statistics are after model-gene projection, not full unfiltered library size.
This script neither guesses guide identity from expression nor claims to reproduce
all32 author-specific upstream callers. Preserve the original calling/QC files.

## Training and evaluation isolation

The released training recipe used 5,224,705 groups and 29,680,402 cell contributions.
Each pass visited every retained group; capped cell-line/context/target weights
balanced losses without oversampling tiny groups. It did not draw each line at
exactly equal frequency. [training_recipes.json](training_recipes.json) records
seed, optimizer, objective, world/batch/pass budgets.

These compact deployment references are **not** the full training corpus or an
exact retraining dataset. Reproducing original training from raw deposits requires
publisher-specific calls, exclusion manifests and the complete pseudobulk teachers;
do not train on these2,150 rows and call it a reproduction of29.68M cells.

When adding data, deduplicate overlapping releases before group aggregation,
exclude the target evaluation study/family from response references and centering,
and fit PCA/programs/calibration only on TRAIN. Do not claim H1 is universally
unseen: historical H1 is part of Atlas. KOLF's historical cell holdout is not an
unseen study. Final anonymous controls alone do not reveal the final cell line.
