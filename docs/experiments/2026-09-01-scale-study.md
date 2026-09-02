# Recipe scale study — 2026-09-01

This is a non-normative diagnostic record. It does not amend recipe mechanics,
card thresholds, or Tier 2 statuses. The reviewed card protocols remain the
reproduction authority; the scaled runs below deliberately vary one of their
seed count, sample count, optimiser-step budget, or hidden width.

## Environment and evidence

- Commit: `fcc14ebb873467df15dae30dd5675e97d323fee7`, plus the CNFlow
  `ExternalBatches` repair described below.
- Host: macOS 13.7.8, Intel Core i5-7500 (4 logical CPUs).
- Runtime: Python 3.11.13, PyTorch 2.2.2 CPU, NumPy 1.26.4. NumPy was held
  below 2 because the available PyTorch 2.2.2 wheel was compiled against the
  NumPy 1.x ABI.
- Statistics are replicate means plus sample standard errors. Seed extensions
  continue each benchmark's existing integer replicate index, preserving its
  documented base-seed mapping.
- Full per-replicate JSON is retained locally under `runs/scale-study/`; `runs/`
  is intentionally gitignored as an execution-artifact directory.

## Results

| Recipe | Deliberate change | Replicates | Primary result | Relevant secondary result |
|---|---:|---:|---:|---:|
| FixMatch | reviewed baseline, 3,000 steps | 10 | EMA treatment-NLL ratio `0.9022 +/- 0.0297` | trained ratio `0.8819 +/- 0.0441`; mask rate `0.7881 +/- 0.0149`; impurity `0.0514 +/- 0.0023` |
| SSDML | reviewed baseline, 4,000 rows | 20 | staged absolute ATE error `0.4266 +/- 0.0121` | hidden-treatment accuracy `0.6488 +/- 0.0022` |
| SSDML | seeds 20 -> 100, 4,000 rows | 100 | staged absolute ATE error `0.4291 +/- 0.0059` | hidden-treatment accuracy `0.6499 +/- 0.0011` |
| SSDML | rows 4,000 -> 16,000 | 20 | staged absolute ATE error `0.2565 +/- 0.0076` | hidden-treatment accuracy `0.6906 +/- 0.0016` |
| TARNet extension | former reviewed recipe, realisations 1-10, 3,000 final-checkpoint steps | 10 | sqrt-PEHE `1.4582 +/- 0.1156` | absolute ATE error `0.2829 +/- 0.0654` |
| TARNet extension | realisations 1-100, 3,000 final-checkpoint steps | 100 | sqrt-PEHE `1.3945 +/- 0.0415` | absolute ATE error `0.3722 +/- 0.0309` |
| TARNet extension | steps 3,000 -> 9,000, realisations 1-10 | 10 | sqrt-PEHE `1.9921 +/- 0.1448` | absolute ATE error `0.6433 +/- 0.1429` |
| TARNet extension | hidden widths 2x, realisations 1-10 | 10 | sqrt-PEHE `1.7241 +/- 0.1097` | absolute ATE error `0.4820 +/- 0.1035` |
| TARNet | corrected outcome-only MSE; GitHub demo archive; validation selection every 200 steps | 100 | sqrt-PEHE `0.7675 +/- 0.0353` | absolute ATE error `0.2143 +/- 0.0148`; selected step `960 +/- 88.9` |
| TARNet | corrected outcome-only MSE; full paper archive; validation selection every 200 steps | 1,000 | sqrt-PEHE `0.8003 +/- 0.0130` | absolute ATE error `0.2048 +/- 0.0053`; selected step `1065.6 +/- 27.9` |
| CNFlow | repaired baseline, 3,000 steps | 4 | paired conditional-NLL difference `+2.2948 +/- 0.1697` nat/row | paired sqrt-PEHE difference `-0.1214 +/- 0.0790` |
| CNFlow | steps 3,000 -> 9,000 | 4 | paired conditional-NLL difference `+3.5163 +/- 0.7982` nat/row | paired sqrt-PEHE difference `-0.5334 +/- 0.0636` |

The SSDML row increase preserves 50% treatment MCAR and five folds. It changes
only `_ROWS`; its posterior stage remains 500 steps. The 16,000-row result is
inside the card's numeric error threshold of 0.30, while the 100-seed 4,000-row
result decisively is not. The small-data miss is therefore persistent bias at
the reviewed scale, but the bias shrinks with additional data.

The first four TARNet rows above were subsequently identified as the xty2
missing-treatment extension, not the published method: they include a
propensity loss, use half-scaled unit-Gaussian NLL, and report the final
checkpoint. They are retained as extension diagnostics and no longer count as
TARNet reproduction evidence. The two corrected rows use exact weighted MSE,
no propensity head, and minimum-validation-objective selection. The first uses
the reference repository's separate 100-realisation GitHub demo. The second
checksum-pins the full archives linked by the reference README and runs the
paper's complete 1,000-realisation experiment. The 2x-width extension changes
encoder widths from `[200, 200, 200]` to `[400, 400, 400]` and outcome-arm
widths from `[100, 100, 100]` to `[200, 200, 200]`, increasing total parameters
from 166,804 to 653,604. Both
the 9,000-step and larger-model panels use realisations 1-10 and otherwise keep
the reviewed protocol fixed.

The CNFlow panels use only four paired replicates, so their standard errors are
diagnostic rather than status-grade. Each flow/Gaussian pair sees identical
populations, missingness, ordered batches, and initial shared parameters.

## Findings

1. **FixMatch is robust on the current executable baseline.** Every declared
   threshold passes, including the EMA comparison that the paper reports.
2. **SSDML's reviewed miss is not seed noise.** Increasing from 20 to 100 seeds
   barely moves the center and halves its standard error. Four times the data,
   however, reduces staged ATE error by about 40% and clears the numeric target.
3. **The former TARNet gap was an implementation-fidelity failure.** Once the
   propensity objective was separated, exact MSE restored, and validation
   checkpoint selection implemented, the 100-realisation diagnostic fell from
   `1.3945` to `0.7675`. The full 1,000-realisation result is
   `0.8003 +/- 0.0130`, inside the predeclared reproduction interval with enough
   margin to change the card status to `reproduced`. The extension's
   longer/larger failures remain useful evidence that scale could not repair
   the wrong objective or selection rule.
4. **CNFlow is not merely undertrained.** Tripling constant-rate Adam steps
   worsens conditional NLL relative to the Gaussian comparator. Its effect
   estimates remain relatively better, but that does not satisfy the card's
   conditional-density claim.

These are scaling diagnostics, not licence to retune reviewed recipes. The
TARNet and CNFlow negative results still require equation-by-equation fidelity
audits before simplifying or changing their mechanics.

## CNFlow execution repair

Before the scale run, CNFlow's benchmark recipe failed at construction because
its gradient stage declared no sampler. Card section 4 already assigns batch
ownership to an external `BatchSource`, and section 6 fixes the ordered
256-row stream, so the benchmark now declares `ExternalBatches()`. This is a
wiring repair to make the existing reviewed protocol executable, not a new
mechanic. The four-seed baseline above is the first fresh evidence after that
repair; the old ten-seed card ledger was not rewritten.
