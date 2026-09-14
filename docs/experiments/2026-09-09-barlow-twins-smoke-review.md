# Barlow Twins Tier 1 review

Reviewed `c9d5eed`, including Claude's Tier 1 implementation in `d378d16`.
The loss arithmetic and projector match the card's pinned author-code variant.
The four-arm construction, seeds, 200/300-step overrides, unchanged marginal
ramp, cached training views and frozen-BN embedding evaluation follow §6.2–§6.3.
The compiled plan still agrees with §3–§4; all 66 applicable card leaves pass
their existing by-value checks. No training implementation or acceptance bound
changes in this correction. Status remains `smoke-passing`.

## Corrections

- Compare actual treatment-mask identities across arms within each stage.
  Forty observed treatments in each arm does not establish identical labels.
- Check feature standard deviations and outcome means/standard deviations
  against the training data, and compare all fitted statistics across arms.
  The original study checked only feature means.
- Compare checkpoint buffers as well as parameters at the stage transition.
- Report conditional-mean treatment-effect RMSE against the analytic DGP
  effect, using the shared causal evaluation helpers. Multiply the predicted
  contrast by the fitted outcome scale to return to original DGP units.
  This is informational, as §6.4 requires; there is no directional assertion.
- Correct the card's count of Claude's listed study mutants from six to eight.

Pretraining and downstream stages use different missingness seeds under the
existing loader. The comparison checks corresponding stages across arms;
pretraining does not consume labels. The downstream masks are identical across
all four arms, including the seed-shifted no-pretraining arm.

## Verification

On CPU, Python 3.12.14, PyTorch 2.14.0+cpu, NumPy 2.4.6, one thread:

- `uv run pytest tests/invariants`: 1,576 passed, including 49 Barlow Twins
  invariants.
- `uv run pytest -q -s tests/smoke/test_barlow_twins.py`: all three declared
  seeds passed in 102.02 seconds, with all four arms at the full smoke budget.
- `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy --strict` (213 source files), and `git diff --check`: passed.
- The required `uv run pytest tests/invariants tests/smoke` command passed
  the invariant section and all three Barlow Twins studies, then was interrupted
  in the unrelated CNFlow smoke fit. No full smoke-suite pass is claimed.

Commands used the existing installed environment through
`UV_PROJECT_ENVIRONMENT`, with `UV_NO_SYNC=1`, `PYTHONPATH=.` and single-thread
BLAS settings. Dependencies were not modified.

Mutation checks shortened training to two steps per stage on base 419, keeping
the four arms and diagnostic evaluation. A returned population with a rolled
treatment mask (same count), or with 0.1 added separately to `x_scale`,
`y_location`, or `y_scale`, passed the original study. Each now fails its new
assertion. Adding one to checkpoint buffers fails the direct transition-state
comparison. These mutations were confined to the harness and removed before
the declared-budget runs.

## Rerun results

[The review JSON](2026-09-09-barlow-twins-smoke-review.json) records all arm
metrics, the source parent, test-module SHA-256, runtime and paired differences.
The test was run from that parent plus this uncommitted test correction;
this is Tier 1 evidence, not a committed-tree Tier 2 reproduction.

| Base | Full alignment | Full active fraction | Redundancy: diagonal-only minus full | Outcome NLL: full minus no-pretraining |
|---|---|---|---|---|
| 419 | 0.974638 | 1.0 | 0.629242 | -0.045904 |
| 523 | 0.975310 | 1.0 | 0.404182 | -0.013469 |
| 631 | 0.979006 | 1.0 | 0.486923 | -0.006165 |

Conditional-mean treatment-effect RMSE, in original DGP outcome units:

| Base | Full | Diagonal-only | No pretraining | VICReg |
|---|---|---|---|---|
| 419 | 1.049489 | 0.712797 | 1.290845 | 0.876616 |
| 523 | 0.677728 | 0.671180 | 0.591345 | 0.823684 |
| 631 | 1.315794 | 1.957118 | 1.370340 | 1.658996 |

These measurements differ from Claude's original environment and remain a
separate record. They support the smoke mechanism checks, but do not assess
the ten-seed, full-budget bounds or establish reliable treatment-effect recovery.
