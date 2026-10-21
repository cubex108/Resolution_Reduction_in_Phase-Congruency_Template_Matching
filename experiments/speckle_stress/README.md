# Supplemental normalized-image multiplicative stress test

This directory is independent of the frozen primary experiment. `protocol.json` precedes the new matching outcomes. Original core files, main protocol, primary/control/timing results and crops are read-only inputs whose hashes are checked at entry and completion. Internal planning, validation narratives and manuscript drafts are kept in `no_public/` and are not part of this release.

The Gamma construction follows the cited despeckling-evaluation model; it is not evidence that the local images are calibrated intensity samples.

## Reproduction

Use the existing Python 3.10 benchmark environment and run commands from this directory. To verify the model and regenerate the analysis of saved results:

```powershell
python -m unittest -v test_stress.py
python summarize_stress.py
```

The original execution sequence was `run_stress.py freeze`, `run_stress.py pilot`, then `run_stress.py run --workers 2`. Do not repeat `freeze` on the delivered directory: it intentionally fails if the protocol exists. Re-executing matching requires the frozen primary protocol, compatible runtime and saved source crops, which are not included in the source-data ZIP. Completed per-pair checkpoints in the full local directory's `parts/` permit interrupted matching to resume. The ZIP omits these duplicate checkpoints but retains every final metric record and perturbation diagnostic. A valid `completion.json` is required before summary generation. Do not conduct another formal timing benchmark concurrently. Runtime fields in the pilot or per-pair checkpoints are diagnostics, not paper speed measurements.

## Validation

Seven tests cover the Gamma mean/variance construction, reproducible and separated random streams, exact clipping without realization rescaling, unperturbed identity, invalid inputs, incomplete/duplicate records and seed nesting within pairs. The full baseline consistency check compares all 600 unperturbed metric rows (five strategies, six offsets, two measures, ten pairs) to the unchanged primary result, including error, success, selected location, ties, flat-map flags and shape descriptors. It must pass before results are marked complete.

All conclusions must preserve the limits recorded in `protocol.json`: normalized observed images, unknown radiometry, pre-existing fluctuations, additional synthetic multiplication, and the nonlinear clipping boundary.

## Outputs

- `stress_rows.jsonl`: all 54,600 metric rows, including the 600 unperturbed control records.
- `perturbation_diagnostics.json`: 900 stochastic fields and ten identity controls, with clipping, means, variance and hashes.
- `completion.json`, `baseline_consistency.json`: completion and preservation checks.
- `statistics/source_rows.csv`: all metric rows in tabular form.
- `statistics/pair_metrics.csv`: ten pair units per configuration; repeated seeds/offsets are aggregated within pairs.
- `statistics/configuration_summaries.csv`: 40 complete condition/strategy/measure summaries.
- `statistics/relative_to_unperturbed.csv`: 30 paired descriptive stress-minus-control comparisons; no p values.
- `statistics/paired_differences.csv`: individual pair contributions to those differences.
- `statistics/seed_scenarios.csv`, `seed_variation.csv`: descriptive variation across seeded cohorts, not additional independent samples.
- `statistics/perturbation_diagnostics.csv`, `perturbation_summaries.csv`: explicit saturation and moment changes.
- `statistics/stress_summary.json`, `analysis_signature.json`: combined analysis and provenance.
