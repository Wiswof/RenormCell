# Frozen models

## official-best

- Official VCC 2026 validation score: **0.1592489705417653**.
- Entry: `DJU2PxQod5gWYtcgz0jq`; panel `vcc2026-val-1`, partition `val`.
- Anchor: `vcc2026-valA-r4+vcc2026-valB-r4+vcc2026-valC-r4`.
- Previous fixed K448 reference: `0.15884544993318364`; difference **+0.00040352060858164784**.
- Architecture: independent mean/bulk ProgramDual heads, rank32 gene-dependent source gates,
  width64, 6,644,996 parameters. Both initial and trained snapshots are required.
- 20 full group passes, 12,760 optimizer updates. Full-data budget in
  [training_recipes.json](data/training_recipes.json).
- Recipe: exact AtlasShift/K448 scaffold, then original dual-moment `q20/q0`
  count reconstruction, correction strength1.0; no reused Flow/AE/PIE weights.
- Historical submitted prediction SHA256:
  `488c99d2e42019b031ade10b45c5215b24089fe9f142a70a7ecc43e8c841b149`.
  The portable writer may serialize rows/HDF5 differently; compare aligned counts,
  not whole-file hashes, when testing a new output.

Official normalized metric credits (not raw six metrics):

| PDS | MSE | DE log-FC accuracy | Fidelity | Magnitude reach | DE Jaccard |
|---:|---:|---:|---:|---:|---:|
|0.6813491806898805|0.0513254325129048|0.1375858051705812|-0.0173276946313969|0.1043503465118846|-0.0017892470032624|

## final-candidate

- Architecture: GeneProgramDual64, width128, hidden512, four residual blocks per head,
  learned target-ID/ESM features, signed gene-dependent source coefficients and a
  rank64 residual; **29,243,652 parameters**.
- 100 full group passes, 63,800 updates. Same full-data cohort as above:
  5,224,705 pseudobulk groups / 29,680,402 retained training-cell contributions.
  These are metadata counts, not independent biological replicates.
- Pooled source responses use their **target-matched pooled controls**.
- Recipe: official-best scaffold/correction, plus full-current-panel-centered
  `log(q100/q0)` **bulk-only** count reconstruction. Same original frozen
  source-common subtraction; qualified extra K562 rows extend the final Atlas scaffold
  without changing the neural model's native-target priors.
- Exploratory, repeatedly exposed local development macro change **+0.0322523**;
  conditional target-bootstrap interval **[+0.0129925,+0.0518193]**.
  H1 +0.0728820 / KOLF +0.0289518 / DLD -0.0050770.
- This interval is **not independent test evidence**, not an official score
  forecast, and does not account for all model-selection uncertainty.
- KOLF/DLD DE Jaccard deteriorated; direct observed bulk error did not improve.
  Part of the favorable SDK-MSE change comes from prediction-variance deduction.
- A related log1p readout scored 0.131558 on validation; that is **not** this
  relative-bulk output. A related dual-relative reconstruction failed feasibility
  guards; we do not disable those guards or claim that failed variant was submitted.
- Final phase does not give an immediate public quality score. No final-score
  improvement or rank is asserted here.

## Scope and limitations

Assets contain only the reference rows needed for the 471-target validation/final
union: 2,150 rows in each compact source bank. Identical row counts do not imply
identical responses/controls. Missing measured source genes use explicit masks.
Twenty-seven study identities have nonzero contributions to these checkpoints'
TRAIN groups; the 32-source inventory also lists reviewed/excluded/auxiliary sources.
Atlas's four principal studies/CD4 prior are separately documented.

H1 is not universally unseen across Atlas assets; KOLF is a same-study cell holdout,
not an unseen cell line. Historical DLD scoring data were isolated from neural
TRAIN, but final anonymous context identity cannot be inferred from target overlap.
Do not equate target coverage with biological inset coverage.

Public exports strip optimizer/RNG state and private paths; every parameter tensor
is preserved exactly. Release manifests pin exported files; `atlas/provenance.json`
records original checkpoint fingerprints. CUDA query parity and actual400 count
regression details are in [PORTABILITY.json](tests/PORTABILITY.json).
