# CNFlow intermittent smoke failure, 2026-10-23

## Finding

The failing check is a one-draw generalisation claim, not an exception in the
flow or a non-finite training loss:

```python
assert marginal.marginal_outcome_nll < complete_case.marginal_outcome_nll - 0.05
```

Two CI jobs reported exactly the same failure:

- [main at 333bac8](https://github.com/mattsq/xty2/actions/runs/39896104430/job/120226824098)
- [dffff88](https://github.com/mattsq/xty2/actions/runs/39886916195/job/120197120134)

Both reported marginal NLL **2.2949752807617188** and complete-case NLL
**1.5301986932754517**, a difference of **+0.7647765874862671** nats. The
other CNFlow smoke assertions passed. In the first job, marginal ATE was
0.6992563 and sqrt-PEHE 0.3286079; complete-case ATE was 0.7456975 and
sqrt-PEHE 0.2907665. These are finite trained models, not numerical blow-ups.

[The intervening 9ac95a5 job passed](https://github.com/mattsq/xty2/actions/runs/39892878095/job/120216474235).
The CNFlow code and test did not change between these commits. The passing
and failing jobs installed Torch 2.14.1+cpu and NumPy 2.4.6 on Python 3.11.17.
They ran in different runner regions (westus3 versus northcentralus). The
logs do not identify a CPU instruction path, so a specific hardware cause is
not established. Platform-dependent floating-point trajectories are consistent
with this evidence; changing library versions is not needed to trigger it.

The earlier attempt in `d060df3` increased the training population from 128 to
512 while retaining the 0.05-nat gate. That did not establish a stable ordering.
The fixture still has only 256 observed treatments, noise standard deviation
0.1, and 3,000 steps of a large unregularised conditional density model.
Exact observed-data likelihood does not guarantee that this finite-sample
optimisation improves held-out likelihood relative to complete cases.

## Implementation audit

Checked the recipe, distribution, objectives and executor against card §3–§5
and the pinned [XTYLearner implementation](https://github.com/mattsq/XTYLearner/blob/35734ec2d5a62d54a59eca38d1e31423da31e1ea/xtylearner/models/cnflow_model.py):

- The five nflows 0.14 RQ-autoregressive transforms, 128-wide two-block
  conditioners, eight bins, ReLU, no dropout/normalisation, explicit linear
  tails and bound 3 match the card. The declared departures from XTYLearner,
  including omitted training jitter and changed tail bound, remain explicit.
- Candidate treatment is one-hot context, not an invertible event coordinate.
  Density evaluates the forward transform from outcome to base noise and adds
  its log-Jacobian; means use the inverse and fixed antithetic draws.
- Missing-treatment loss is `-logsumexp(log p(t|x) + log p(y|x,t))`, with
  gradients into both heads. Complete-case terms select observed rows. The
  mixer applies population reduction and the declared unit objective weights.
- The paired smoke fits use identical initial state and ordered batches.
  `run_stage` restores compiled initial parameters before executing each fit.

No discrepancy requiring a production recipe change was found in this audit.
This does not establish good held-out density performance. The recorded Tier 2
result is already `deviating` (flow minus Gaussian conditional NLL
1.88504 +/- 0.225), and remains so.

## Repair

Withdraw the unstable Tier 1 NLL ordering assertion, retaining both paired
fits, their finite-metric checks, and the original loss/propensity/ATE checks.
Add a direct one-step executor test at seeds 17, 18 and 19:

1. Generate 48 ordinary finite rows and hide all treatments, with unit weights.
2. Independently enumerate the two treatment densities and differentiate their
   probability-weighted mixture. Compare this loss with the executor's total.
3. Require nonzero descent along that reference gradient separately for the
   encoder, flow and propensity head.
4. Flip the hidden treatment placeholders and require the exact same update.
5. Run the complete-case ablation from identical parameters and require zero
   loss and no parameter update.

Only this probe uses one step; the full paired fit retains its 3,000-step budget.
No optimiser, architecture, threshold, seed selection, checkpoint selection,
Tier 2 metric or benchmark status is changed to manufacture a favourable fit.

Mutation checks observed failing: detach the propensity path; detach the
outcome path; multiply the marginal term by zero; reverse its sign; construct
candidate treatments from the hidden placeholders. All three clean seeds
passed. Fifteen NaN/positive-infinity/negative-infinity injections across the
trace and four measured fit metrics were rejected by the finite checks.

## Validation

Local runtime: Python 3.12.14, Torch 2.14.1+cpu, NumPy 2.5.4, nflows 0.14.
The production training implementation and compiled plan are unchanged.

The original, unmodified smoke file passed all four tests locally in 347.17s.
Thus the old gate can pass locally while failing on the CI runner. This local
runtime differs in Python/NumPy; the same-version passing/failing evidence is
the pair of CI jobs above, not this local comparison.

Additional full-budget paired fits changed only the two initialisation
calls from `torch.manual_seed(17)` to seeds 18 and 19, retaining data
seeds 1/2, batch seed 99 and executor seed 23. The results were:

| Initialisation | Arm | Held-out marginal NLL | Propensity NLL | ATE | sqrt-PEHE | First mixed loss | Last-100 mean mixed loss |
|---|---|---:|---:|---:|---:|---:|---:|
| 18 | marginal | 2.44201875 | 0.08578645 | 0.41486812 | 0.63487279 | 2.25633574 | -2.05763730 |
| 18 | complete case | 1.31032538 | 0.11392897 | 0.77910805 | 0.25862750 | 1.10617852 | -1.00255547 |
| 19 | marginal | 2.53604388 | 0.10082000 | 0.74784565 | 0.28545749 | 4.68505383 | -1.95245816 |
| 19 | complete case | 0.83242834 | 0.13250908 | 0.73371166 | 0.32346565 | 2.27554917 | -0.95697176 |

The old likelihood gate fails by a wide margin for both initialisations,
despite falling training loss and passing the other three original checks.
This is consistent with poor generalisation, not evidence of a detached
marginal loss. The mixed training losses use different objectives and are not
an arm-to-arm likelihood comparison.

The new three-seed mechanism test passed both normally and with
`MKL_CBWR=COMPATIBLE ATEN_CPU_CAPABILITY=default` (3 tests in each run).

A full-budget seed-17 fit under those compatibility settings also passed the
old likelihood gate: marginal NLL 1.27780437 versus complete-case 2.31461024.
It did **not** reproduce the failed CI trajectory. Comparing pre-fit tensors
between the normal and compatibility settings found maximum absolute changes
of 1.714e-6 in training inputs, 1.371e-6 in test inputs, and 5.066e-7 in model
state (including buffers); treatment labels/masks/weights and ordered row IDs
were identical. Seeded inputs and initial state can therefore differ at the
last bits across CPU execution settings, but these probes do not identify the
specific numerical path responsible for the CI failure.

All required pre-commit checks passed:

- `uv run --no-sync pytest tests/invariants`: **1,739 passed** (23.51s).
- `uv run --no-sync pytest tests/smoke -n 4 --dist loadfile`: **125 passed**
  (983.62s), including the full paired CNFlow fit and all three mechanism seeds.
- `uv run --no-sync ruff check .`: passed.
- `uv run --no-sync ruff format --check .`: 237 files already formatted.
- `uv run --no-sync mypy --strict`: no issues in 232 source files, with the
  repository's unmodified configuration.

Tier 2 was not rerun: no training code, recipe mechanics or Tier 2 protocol
changed, and this patch makes no new performance claim.
