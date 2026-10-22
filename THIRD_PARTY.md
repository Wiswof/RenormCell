# Attribution and data terms

- **AtlasShift / Virtual Cell Challenge 2026**, commit
  `d24ce4fdae0cd7cba3ba29546cd8737094eae98a`:
  <https://github.com/kaipengm2/Virtual-Cell-Challenge-2026>.
  `renormcell/atlas.py` and the control/count-writer implementation derive from
  this MIT-licensed code. Full notice: [licenses/AtlasShift-MIT.txt](licenses/AtlasShift-MIT.txt).
  Retained methods include source-centered response transfer, dual moments and promoter prior.
- **ESM2**, protein sequence features:
  <https://github.com/facebookresearch/esm>;
  Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with
  a language model*, Science (2023), <https://doi.org/10.1126/science.ade2574>.
- **cell-eval2**, evaluation definitions:
  <https://github.com/ArcInstitute/cell-eval2>. No official scoring server or hidden
  ground truth is included; local composite changes are not official points.
- Public study authors, accession/DOI links and per-study scope are in
  [data/sources.csv](data/sources.csv). Please cite the relevant original studies,
  not only this repository, when reusing their experimental evidence.

## Distinguish code from data

The MIT license grants rights in our original code/weight contributions; it does
not relicense third-party experimental data. The distributed source features are
transformed **aggregate numerical statistics**, not individual-cell RNA matrices,
and retain upstream source attribution. Changes include gene projection, author
QC selection, matched-NTC subtraction, technical pooling where specified and
source-common subtraction for Atlas effects.

Confirmed publisher records include:

- Replogle K562, Figshare article20029387 / file35775507: CC-BY-4.0.
- KOLF2.1J, Figshare article27261219 / file64650261: CC-BY-4.0.
- VIP, Zenodo18460279: CC-BY-4.0.
- CD4/GSE314342: retain the authors' processed-data attribution; an MIT declaration
  for an assigned-guide/processed file is not assumed to cover every raw RNA file.

GEO accession availability does not create a blanket MIT/CC license. We retain
original study links/metadata and do not distribute those raw deposits. NCBI's
[data usage policy](https://www.ncbi.nlm.nih.gov/home/about/policies/) states its
distribution policy and cautions about submitters' possible intellectual-property
claims. Historical public H1 release URLs/checksums likewise do not establish an
independent blanket RNA redistribution license. Consult each publisher's terms
for your intended reuse, especially before redistributing original data.

Official VCC2026 controls and authentication are user-supplied under the current
official rules. No private study data, original single-cell RNA, individual-cell
barcodes, official control matrices or credentials are included in Git or release
archives. This repository does not grant rights it does not own.
