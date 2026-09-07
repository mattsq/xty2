# ReMixMatch corrected benchmark, 2026-09-07

The corrected benchmark meets **eight of nine** required criteria and remains
`deviating`. Its true-training-marginal advantage is `0.00994868 +/- 0.0303877`,
which does not clear the unchanged one-standard-error rule. The actual Tier 2
pytest **passed** because it verifies agreement with the recorded status.
This is project-local mechanism evidence, not a reproduction of the paper's
image benchmarks.

## Protocol and execution

- Source commit: `86b6501ca814376fc59406e11bd9980dd32e7312`, clean checkout.
- Protocol digest: `9ad7a145cf7acd5edda84771d1254b10db4a5b01cb6e7d5f09b04d70b4ed65e6`.
- Ten replicates: base seeds `90000 + 100*i`, `i=0,...,9`.
- Four arms: full, `no_remixmatch`, `no_alignment`, `no_mixup`.
- 3,000 optimiser steps per fit: 40 fits, 120,000 steps total; eight workers.
- Linux x86_64 CPU; Python 3.12.13, PyTorch 2.14.0+cpu, NumPy 2.2.6,
  pytest 9.1.1; one OpenMP/MKL thread per worker.
- No protocol deviation. No seeds, numerical thresholds or guardrail definitions changed.
- Test result: **1 passed in 858.83s (14m18s)**.
- [Raw JSON](results/remixmatch-86b6501ca814.json) contains all per-seed values,
  criteria, exact means and sample standard errors.
- [Complete pytest log](results/remixmatch-86b6501ca814-pytest.txt).

Executed from the source commit:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XTY2_TIER2_WORKERS=8 \
  XTY2_WRITE_TIER2_LEDGERS=1 PYTHONUNBUFFERED=1 \
  python -m pytest tests/benchmarks/test_remixmatch.py -vv --durations=0
```

The benchmark correction removes pseudo-targets and pooled MixUp from the
former baseline's labelled loss. `no_remixmatch` uses observed-treatment NLL
on the same first strong view and retains the shared causal terms. The
historical `supervised_only` ratios are not interchangeable with this
comparison. The production ReMixMatch recipe was not changed by this patch.

## Required results

Each mean must clear its numeric bound by at least one sample standard error.

| Metric | Mean +/- stderr | Bound | Pass |
|---|---:|---:|:---:|
| `full_vs_no_remixmatch_student_macro_NLL_ratio` | 0.74089039 +/- 0.03410872 | <= 1 | yes |
| `full_vs_no_remixmatch_ema_macro_NLL_ratio` | 0.84298572 +/- 0.020343268 | <= 1 | yes |
| `full_vs_no_alignment_student_macro_NLL_ratio` | 0.89283872 +/- 0.067596619 | <= 1 | yes |
| `full_vs_no_alignment_ema_macro_NLL_ratio` | 0.94897136 +/- 0.035994882 | <= 1 | yes |
| `full_vs_no_remixmatch_outcome_NLL_ratio` | 0.98547643 +/- 0.0032395779 | <= 1.05 | yes |
| `alignment_marginal_L1_advantage` | 0.009948677 +/- 0.030387716 | >= 0 | no |
| `terminal_pretext_accuracy` | 0.36120313 +/- 0.0036219493 | >= 0.25 | yes |
| `terminal_mixed_lambda_min` | 0.50007851 +/- 2.1763096e-05 | >= 0.5 | yes |
| `terminal_mixed_lambda_max` | 0.99999768 +/- 1.0756194e-06 | <= 1 | yes |

The student and EMA ratios against the corrected baseline both clear their
bounds. The alignment comparison's true-marginal miss is close to the
historical `0.0103444 +/- 0.0305`; the old ledger row remains unchanged.
A small numerical difference across runs does not establish a causal effect
of the baseline repair on the other arms.

## Marginal diagnosis

The required guardrail compares the last 128 pre-update, unaligned student
weak-anchor prediction means with the realised all-training treatment
histogram. The new diagnostics also compare these windows with the estimated
labelled prior and realised unlabelled truth. Hidden labels are evaluator-only.

| Informational diagnostic | Mean +/- stderr |
|---|---:|
| `labelled_vs_true_training_marginal_L1` | 0.17753906 +/- 0.027324289 |
| `labelled_vs_true_unlabelled_marginal_L1` | 0.189375 +/- 0.029145909 |
| `true_unlabelled_vs_training_marginal_L1` | 0.011835938 +/- 0.0018216193 |
| `alignment_labelled_marginal_L1_advantage` | 0.11662852 +/- 0.011286087 |
| `alignment_true_unlabelled_marginal_L1_advantage` | 0.0028856953 +/- 0.029568318 |

Positive advantages mean alignment reduces L1 distance. Every seed moves
closer to the estimated labelled prior, while only five improve against true
unlabelled truth. Finite-label target error is measurable; the causal account
remains untested. Bias correction cannot remove sampling error in 64 labels.

| Base seed | Advantage vs training truth | Advantage vs labelled estimate | Advantage vs unlabelled truth |
|---:|---:|---:|---:|
| 90000 | -0.14532530 | 0.14532530 | -0.14532530 |
| 90100 | 0.09643947 | 0.15024233 | 0.08446030 |
| 90200 | 0.04061428 | 0.06101943 | 0.03123928 |
| 90300 | 0.09251293 | 0.10889255 | 0.08053376 |
| 90400 | 0.16094368 | 0.15143478 | 0.15742193 |
| 90500 | 0.04310633 | 0.11768609 | 0.03321049 |
| 90600 | 0.00490271 | 0.10816071 | -0.00603479 |
| 90700 | -0.05688116 | 0.07788102 | -0.05945182 |
| 90800 | -0.02912213 | 0.16462957 | -0.03641379 |
| 90900 | -0.10770403 | 0.08101336 | -0.11078309 |

## Merge assessment and follow-up

The [fidelity policy](../FIDELITY.md) allows a documented `deviating` result.
The marginal miss blocks a `reproduced` claim, but does not alone require a
merge veto. The invalid baseline has been corrected and the full benchmark
rerun; merging this implementation should preserve its explicit limitation.
The numerical classification benefit applies only to this fixture.

The most defensible next experiment is a paired comparison of estimated-prior
alignment, oracle true-unlabelled-prior alignment and no alignment. Replace
only the alignment numerator after the shared uniform first step. Predeclare
the seed set, budget and paired contrasts before execution. Keep the oracle
in evaluator-owned diagnostic code; do not change production defaults or
substitute it for acceptance evidence. An oracle rescue would implicate prior
estimation; no rescue would direct investigation toward the augmentation,
sharpening and optimisation interaction without proving any one cause.
This oracle experiment has not been run or implemented.

## Targeted verification

The correction passed 371 targeted invariant, reporting, navigation, deviation
and smoke checks in 51.79 seconds, plus Ruff lint/format and strict mypy on the
changed Python files. These include baseline loss/gradient isolation, paired
initialisation and settings, hidden-label evaluator isolation, marginal-state
immutability, and preservation of a failing guardrail despite favourable
informational diagnostics. Source-commit [remote CI](https://github.com/mattsq/xty2/actions/runs/34061302261)
passed lint, typecheck and Tier 0/1. The complete Tier 2 run above supplies the
new scientific evidence independently of those implementation checks.

After recording the result, 340 reporting, navigation and deviation-debt
checks passed. A direct audit confirmed the unchanged protocol digest and
historical ledger row, all ten-value means and standard errors, acceptance
flags, and the byte-identical committed copy of the runner JSON.
