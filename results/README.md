# Six-metric result figures

| Figure | Evidence | Compared recipes |
|---|---|---|
| [Official validation](official_six_credits.png) | Recorded normalized official credits, same panel/anchor | Atlas+K448 vs `official-best` |
| [H1](h1_six_metrics.png) | 35-target guide-heldout development panel | All three recipes |
| [KOLF](kolf_six_metrics.png) | 189-target same-study cell holdout | All three recipes |
| [DLD-1](dld_six_metrics.png) | 43-target source-isolated scoring panel | All three recipes |

The orange `final-candidate` points refer to the full-panel-centered **relative100
bulk readout**, not the failed dual readout or the earlier log1p submission.
The blue `official-best` points use the recipe with recorded official score0.159249;
its local metric values are not that official score. Gray is fixed Atlas+K448.

## Read the plots correctly

- Local outputs: actual400 integer cells per target, original matched controls,
  fixed generation seed20261017, cell-eval2 0.16.0.
- Raw local PDS/fidelity/reach/Jaccard are higher-is-better; normalized MSE and DE
  log-FC NMAE are lower-is-better. All **official normalized credits** are
  higher-is-better, including MSE credit. Do not mix these two scales.
- Local error bars: 95% descriptive target bootstrap, 2,000 paired target resamples.
  The same draws are used for all recipes within a context. PDS uses precomputed
  full-panel ranks rather than re-ranking each bootstrap panel. Intervals exclude
  cell-line and generation-seed uncertainty and do not correct for repeated tuning.
- Undefined per-target metrics remain undefined. The number of finite targets is
  printed beneath each point; no zeros are substituted. An interval is omitted if
  fewer than95% of bootstrap aggregate values are defined.
- Normalized MSE is a **ratio of sums** of capped unbiased-error numerators and
  unbiased-distance denominators, not an average of per-target ratios. Its CSV
  exports include these two sufficient-statistic fields.
- The official aggregates have no per-target uncertainty receipt; no artificial
  error bars are added. `final-candidate` has no official final score—missing is not0.

## Download the numbers / redraw

- [local_aggregates.csv](local_aggregates.csv): all54 raw context/profile/metric values.
- [local_intervals.csv](local_intervals.csv): bootstrap limits, finite targets,
  invalid-resample fractions. Computed directly from the exported per-target values.
- [per_target_metrics.csv](per_target_metrics.csv): model metric results and the two
  MSE aggregation components. **No cell-level RNA or hidden official final truth**.
- [official_validation.csv](official_validation.csv): recorded previous/current
  official credits, overall score and their differences.
- [provenance.json](provenance.json): original result-file hashes, panels/protocol
  and explicit limitations; no private filesystem paths or credentials.

From the repository root:

```bash
python -m pip install -e '.[plots]'
python scripts/plot_results.py
```

No model inference, GPU, original RNA or scoring-server access is required to redraw
these figures. The script verifies all54 aggregates against the per-target export
before plotting. The data are copied from recorded evaluations; the script does
not rerun the official evaluator or estimate a new competition score.

## Limits, including unfavorable results

These local panels have repeatedly been used for development. H1 is not universally
unseen across Atlas inputs; KOLF is a same-study cell split, not an unseen study.
DLD's source was isolated from neural TRAIN, but this scoring panel is now exposed.
None of this proves independence of the anonymous final D/E/F cell backgrounds.
The candidate's final-only new-target K562 scaffold extension is not validated by
these older local panels. Better SDK-MSE can include prediction-variance deduction;
direct raw-bulk error is not established to improve. KOLF/DLD Jaccard worsens.

These figures document both benefits and tradeoffs, not a claim that the candidate
has surpassed the official-best score or will win the competition.
