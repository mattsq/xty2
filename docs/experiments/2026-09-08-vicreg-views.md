# VICReg paired view experiment: completed results

The target-preserving view policy meets **all four original bounds** on the
ten-seed, full-budget study. The unchanged marginal-corruption policy still
misses spread. This supports a consequential augmentation choice as the source
of the miss; neither loss arithmetic nor the architecture/optimiser has to
change for the same mechanism targets to pass.

The canonical card remains `deviating`: this diagnostic does not silently
replace its declared views. The [proposed contract amendment](../proposals/vicreg-valid-views.md)
sets out a route to `reproduced` that retains the original failure and does not
lower any acceptance threshold.

## Predeclaration, execution and provenance

- [Protocol](2026-09-08-vicreg-views-protocol.md) and implementation were committed
  to PR #48 at `520fdd4c870e7cdceb8f438a54769b51977b7149` before execution.
- The local execution commit is `56c77bcea8be468e3fc551aa9717203b6d6753fa`.
  It materialises the **identical Git tree** of that remote commit:
  `684f940a84aad548aa842a5f8084d424de003f84`. The local commit differs only in
  commit metadata/history, not tracked file contents. The machine output records
  the actual local commit and tree instead of substituting a remote SHA.
- Protocol SHA256: `833039a456f6fa0f42e6c1fe1f556ee4c96d0ada956c68c21334105a64497e1e`.
- Python 3.12.13, PyTorch 2.14.0+cpu, NumPy 2.4.6, Linux x86_64; deterministic
  algorithms, one torch thread per worker, six workers. Both policies run in
  this same environment. Earlier results used another Python/torch build; their
  downstream values are not substituted for new paired observations.
- Ten bases `190000+100*i`, i=0..9. The original train/test/model/execution
  offsets, 1024/2048 rows, 40 observed treatments, batch 128, 1000 pretraining
  steps, 3000 downstream steps, architecture, weights and optimiser are fixed.
- Seventy fits: full/no-variance/no-covariance under each of two view policies,
  plus one shared no-pretraining control per seed. All are complete.
- Raw [summary](results/vicreg-views-520fdd4/summary.json), per-seed JSON,
  environment and both compiled plans are in
  [the result directory](results/vicreg-views-520fdd4/).

There was no parameter/threshold search or early stopping. An initial launch
stopped at the clean-tree check before fitting; the successful run started from
the committed source tree. No numerical protocol change was made after results
were inspected.

## Required measurements

Values are mean +/- sample SE over the ten paired seeds. Lower bounds require
mean minus SE to meet the threshold; the upper bound requires mean plus SE.
Every measurement and decision below was independently recomputed from the
per-seed values using Python's `statistics` module.

| Target | Marginal corruption | Target-preserving symmetry | Original bound |
|---|---:|---:|---:|
| Full-arm spread | 0.268756 +/- 0.001818 **fail** | 0.766295 +/- 0.003455 **pass** | >=0.5 |
| Variance-ablation spread gap | 0.258753 +/- 0.001819 | 0.751048 +/- 0.007494 | >=0.1 |
| Covariance-ablation redundancy gap | 473.623237 +/- 0.358229 | 122.601333 +/- 4.148089 | >=0.01 |
| Outcome NLL cost of pretraining, nat/row | -0.006444 +/- 0.005247 | -0.010098 +/- 0.006223 | <=0.05 |

Both attribution bounds and both outcome guardrails pass. In the symmetry full
arm no dimension is below the separate 0.1 collapse diagnostic threshold.
Removing covariance increases the top eigenvalue share from
`0.04396 +/- 0.00093` to `0.86543 +/- 0.02070`: the covariance effect is not
merely inflating marginal standard deviations.

## Training effects versus evaluation effects

All entries use terminal pretraining checkpoints with frozen training BN
buffers and the same sixteen held-out batches. Each model is evaluated under
both predeclared held-out view policies; no parameter or buffer is fitted here.

| Full-arm training policy | Evaluate marginal views | Evaluate symmetry views |
|---|---:|---:|
| Marginal corruption | 0.268756 +/- 0.001818 | 0.280793 +/- 0.002339 |
| Target-preserving symmetry | 0.775195 +/- 0.004004 | 0.766295 +/- 0.003455 |
| Paired training-policy increase | 0.506439 +/- 0.004942 | 0.485502 +/- 0.004530 |

The large increase persists on the *original* corrupted evaluation inputs.
Changing evaluation inputs alone moves the original model by much less.
The predeclared decision rule is therefore met: all four symmetry targets pass,
and spread increases under both evaluation policies.

## What the views preserve and change

The original view replaces exactly three of six coordinates. The symmetry
reflects the first four coordinates along `(3,5,0,-8)` with probability 1/2,
independently flips x4's sign with probability 1/2, and redraws x5 from its
training marginal. Transformations use original-scale coordinates and return
through the same train-fitted scaler.

The reflection preserves the cluster log-odds, outcome baseline and effect;
the sign flip preserves x4's squared outcome contribution; x5 affects neither
target. This also preserves the signal mixture's Gaussian density. It uses
known fixture equations, not observed treatment/outcome labels. These are
oracle, fixture-specific views, not an automatically learned augmentation.

| View diagnostic | Marginal corruption | Symmetry |
|---|---:|---:|
| Mean absolute Bayes propensity change | 0.235271 +/- 0.001724 | 2.58e-9 |
| Mean absolute conditional-outcome-mean change | 0.410014 +/- 0.002475 | 8.54e-9 |
| Changed coordinates per row | 3 exactly | 3.012671 +/- 0.006097 |
| Standardised feature MSE | 1.007487 +/- 0.003877 | 0.889137 +/- 0.006668 |

Outcome changes average both treatment-specific means in original outcome
units. Symmetry errors are floating-point roundoff, below the predeclared 2e-5
limit. The original propensity-only diagnostic missed that x4 is
outcome-relevant: replacing both cluster-independent columns is not a valid
outcome-preserving control.

Equal changed-coordinate counts do not equate perturbation geometry or energy.
The intervention identifies an effect of the **view policy**, not a separately
identified effect of information preservation while every other augmentation
property is held constant. Both views are nontrivial and independent, with
training-only donor provenance and no held-out fitting.

## Scale diagnostic and interpretation

Doubling the original full model's embeddings raises its held-out VICReg loss
from `21.4752 +/- 0.0063` to `29.3062 +/- 0.2602`, even though it would make
spread exceed 0.5. The output layer can reach the bound, but this rescaling
worsens the objective. The failed bound was not an architectural impossibility.

The same code, optimiser and budget obtain much greater spread under valid
views. This is stronger evidence for the augmentation explanation than the
earlier single-seed training-log interventions. It does not establish that no
other optimiser or weight choice could improve the original corruption policy,
nor does it reproduce ImageNet performance or establish treatment-effect
identification. All informational treatment/ATE values remain in the artifact.

## Validation

- 47 VICReg tests passed: original loss invariants, all three declared smoke
  seeds, new symmetry checks on three seeds and actual-draw tracing checks.
- Mutating the reflection direction, dropping the inverse scaling, or removing
  the x5 donor draw each failed the new invariants, then was reverted.
- Actual executor row/view hashes match across coefficient ablations within
  each policy. Actual pretraining row hashes match across policies. The existing
  benchmark checks additionally verify downstream stream pairing, masks,
  initial states, head isolation and expander isolation.
- All 1,513 invariants passed in the wider suite. Ruff lint/format and
  strict mypy passed locally; GitHub's lint, typecheck and full invariant jobs
  also passed on the predeclared implementation.
- All 115 repository-wide smoke tests passed across 19 modules, each run in a
  separate pytest process with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and
  `XTY2_GRADIENT_DIAGNOSTICS=0` (the CI diagnostics policy).
- No published-method reproduction or canonical-card promotion is inferred
  from this experiment alone.
