# VICReg valid-view confirmation

The approved valid-view contract is **reproduced** on its fixed synthetic
fixture. All four unchanged one-standard-error bounds pass on ten fresh seeds.
This is a fixture-specific mechanism reproduction with explicitly supplied,
target-preserving views that use privileged knowledge of the data generator.
It is not an ImageNet reproduction or a general tabular augmentation policy.

## Prospective execution

The owner approved the [contract amendment](../proposals/vicreg-valid-views.md)
after the [paired diagnostic](2026-09-08-vicreg-views.md). Confirmation used
exactly the predeclared bases `290000+100*i`, i=0..9, separate from the original
`190000+100*i` study. All 40 fits completed: full, no variance, no covariance,
and no pretraining on each seed. No seed, checkpoint, threshold, loss weight,
architecture or training-budget selection was performed.

The source was committed before execution:
[`06060b63d37ccf4ba9554dac0e21f7935ff0535f`](https://github.com/mattsq/xty2/commit/06060b63d37ccf4ba9554dac0e21f7935ff0535f),
tree `e0906bdfa72ce052a3eef2541522d24ae76222d0`. The working tree was clean.
The source commit and complete result are published together on the PR branch.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m experiments.vicreg_confirmation \
  --output runs/vicreg-confirmation-06060b6 --workers 6
```

The driver executes the canonical benchmark's replicate function, saves every
seed, then scores those exact outputs through the canonical benchmark without
retraining. Its scoring hook replaces replicate execution only. Python was
3.12.14, PyTorch 2.14.0+cpu, NumPy 2.4.6; six workers each used one deterministic
Torch CPU thread. The full configuration is in the saved environment and plan.

A workspace reset required restoring the implementation before confirmation.
The full source validation was rerun after restoration. No confirmation fit
was interrupted or replaced; all ten belong to the single completed run above.

## Required results

| Metric | Mean ± SE | Assessed bound | Required bound | Result |
|---|---:|---:|---:|---|
| Full-arm embedding spread | 0.768910 ± 0.005103 | 0.763807 | ≥ 0.5 | pass |
| Variance-ablation spread gap | 0.758907 ± 0.005103 | 0.753803 | ≥ 0.1 | pass |
| Covariance-ablation redundancy gap | 124.709074 ± 4.183431 | 120.525643 | ≥ 0.01 | pass |
| Pretraining outcome NLL cost | −0.027345 ± 0.017587 | −0.009758 | ≤ 0.05 | pass |

The lower-bound tests use mean minus SE; outcome cost uses mean plus SE.
SE is the sample standard deviation divided by sqrt(10). These are the
original aggregate criteria. They do not require every individual outcome
cost to be below 0.05: seed 290500 has cost 0.053650. All per-seed values,
including this one, are retained.

The full arm has zero collapsed dimensions on every seed. The no-variance
arm has all dimensions below the declared 0.1 standard-deviation cutoff on
every seed. Removing the covariance term increases redundancy by a large
paired amount. The outcome criterion establishes the specified transfer
cost guardrail, not treatment-effect identification or universal improvement.

## View validity and pairing

The reflection, x4 sign flip and training-marginal x5 draw are exactly those
specified before the original diagnostic. They preserve Bayes propensity and
both conditional outcome means, while changing inputs nontrivially. In this
confirmation, mean absolute propensity shift is 2.59e-9 and conditional-outcome
mean shift is 8.42e-9, consistent with floating-point roundoff. The mean number
of changed coordinates is 3.0216; 99.961% of paired rows have distinct views.
Every seed passes the predeclared target-error tolerance of 2e-5.

The implementation checks actual pretraining row/view draw hashes across
coefficient arms and exactly two executed views per pretraining step. Existing
benchmark checks enforce shared data/masks/initial states, matched downstream
streams, training-only statistics, and head/expander isolation. The validity
guards run before each replicate returns its evidence. Fine-tuned predictions
use clean held-out rows; terminal pretraining embeddings use frozen training
BN statistics and shared held-out views.

## Smoke diagnostic correction

The valid-view smoke study exposed an undefined diagnostic in the collapsed
no-variance control: both its diagonal covariance energy and total covariance
energy can be zero. The old smoke test asserted a positive denominator and
computed a redundancy ratio for that arm. It failed on seed 191.

The smoke diagnostic now follows the existing Tier 2 contract: redundancy is
reported for full and no-covariance arms only, where the denominator remains
required to be positive. Spread is still measured for the collapsed control.
No undefined ratio is assigned a passing value, and no required threshold was
changed. All three original smoke seeds pass with this correction.

## Validation and evidence

Before the source commit: 1,515 invariants, all 115 smoke tests across 19
modules, Ruff lint/format and strict mypy passed. Both experiment drivers also
passed strict mypy. The focused VICReg suite passed 46 tests. Substituting the
first view for the second and removing the nontriviality guard each made the
new tests fail before restoration. A reduced-budget plumbing check validated
execution and scoring only; its numbers are not confirmation evidence.

After execution, all 44 metric vectors, their means and sample SEs, and all four
required decisions were independently checked against the ten raw seed files
using Python's statistics module. The recorded card specification digest
matches the run. The card retains the original failed `a9fec5cc9623` ledger
row and appends this approved-contract result.

Artifacts: [complete benchmark result](results/vicreg-confirmation-06060b6/vicreg.json),
[independent verification and SHA-256 manifest](results/vicreg-confirmation-06060b6/verification.json),
[validation record](results/vicreg-confirmation-06060b6/validation.json),
[environment](results/vicreg-confirmation-06060b6/environment.json),
[rendered plan](results/vicreg-confirmation-06060b6/plan.txt), and all ten
`seed-*.json` files in the same directory.

The original marginal-corruption adaptation remains a failed adaptation.
Its historical ledger and diagnostic reports are unchanged; the new status
validates the explicitly approved view contract.
