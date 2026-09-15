# SimSiam Tier 2 mechanism study

The prospective local claim is **deviating**: two of four required bounds pass.
The full method retains projection spread and meets the downstream NLL cost
guardrail, but both stop-gradient and predictor attribution gaps are negative
on every replicate. No threshold, seed, checkpoint, architecture, view or
optimisation setting was selected after observing the result.

## Execution and provenance

All ten bases `310000+100*i`, i=0..9, completed all four arms: full,
no stop-gradient, no predictor, and no pretraining. The three pretraining arms
ran 1000 steps each; all four downstream arms ran 3000 steps. Training used
1024 rows with 40 observed treatments, and evaluation used 2048 held-out rows.
Terminal pretraining diagnostics used 16 disjoint batches of 128, two shared
oracle views per batch, eval mode and frozen training BN statistics.

The run used committed source
[`c14090a43e2e2e758084b4e4b220ce07a0a790c2`](https://github.com/mattsq/xty2/commit/c14090a43e2e2e758084b4e4b220ce07a0a790c2),
tree `6b3856be4f1efa200c419a7992c38f47ce5d6ce0`, in a clean detached
worktree. The source is retained by tag `simsiam-tier2-source-2026-09-14`.
The complete implementation, benchmark registration and result land together
on PR #55. The final constructor preserves the older cosine objective's
positional arguments; this changes field ordering only. The measured recipe
passes all fields by keyword, and its plan and numerical values are unchanged.

```bash
git checkout simsiam-tier2-source-2026-09-14
python -m xty2.evaluation.runner --recipe simsiam --workers 3
```

The result date is the run's UTC date, 2026-09-14. Execution used Python
3.11.12, PyTorch 2.2.2 and NumPy 1.26.4 on Intel macOS, with three processes,
one Torch thread per worker and deterministic algorithms. These versions
support the local Intel Mac; the latest PyTorch wheel does not. No claim of
identical floating-point results across PyTorch releases is made.

## Required results

| Metric | Mean +/- SE | Assessed bound | Required bound | Result |
|---|---:|---:|---:|---|
| full_projection_spread | 0.896711 +/- 0.013111 | 0.883600 | >= 0.5 | pass |
| stop_gradient_spread_gap | -0.066665 +/- 0.011058 | -0.077722 | > 0 | fail |
| predictor_spread_gap | -0.078406 +/- 0.012552 | -0.090959 | > 0 | fail |
| pretraining_outcome_nll_cost | -0.029322 +/- 0.032332 | 0.003010 | <= 0.05 | pass |

SE is the sample standard deviation (ddof=1) divided by sqrt(10). Differences
are formed within seed before aggregation. Strict positive attribution bounds
use `mean - SE > 0`, including a regression check at exact equality.

The NLL guardrail passes at mean plus SE = 0.003010 nat/row.
This is not a consistent downstream improvement: four seeds have positive NLL
cost, and base 310800 contributes the largest negative difference. Treatment
effect RMSE remains diagnostic rather than evidence of causal identification.

## Absolute arm measurements

| Arm | Projection spread | Projection effective rank | Raw projection norm | Outcome NLL | Effect RMSE |
|---|---:|---:|---:|---:|---:|
| full | 0.896711 +/- 0.013111 | 1.566101 +/- 0.112220 | 13.037829 +/- 0.437358 | 1.239831 +/- 0.006052 | 1.008405 +/- 0.062146 |
| no_stop | 0.963376 +/- 0.004199 | 1.911183 +/- 0.051683 | 14.248582 +/- 0.199030 | 1.222526 +/- 0.004397 | 0.814007 +/- 0.047486 |
| no_predictor | 0.975118 +/- 0.001994 | 1.631853 +/- 0.033341 | 14.673234 +/- 0.113514 | 1.224686 +/- 0.008975 | 0.882770 +/- 0.065404 |
| no_pretrain | n/a | n/a | n/a | 1.269153 +/- 0.033894 | 1.105476 +/- 0.119043 |

Spread alone conceals severe rank concentration. The full arm's mean projection
effective rank is 1.566101 out of 256, and
its mean encoder effective rank is 1.051538.
Thus passing the spread guard cannot support a claim that the representation
retained high-dimensional information. All reported near-zero vector fractions
are zero; the observed effect is not a numerical epsilon-floor artefact.

## Per-seed required values

| Base | Full spread | Stop-gradient gap | Predictor gap | NLL cost |
|---|---:|---:|---:|---:|
| 310000 | 0.884690 | -0.090988 | -0.078632 | -0.007734 |
| 310100 | 0.946360 | -0.019248 | -0.036037 | 0.005481 |
| 310200 | 0.910407 | -0.044826 | -0.066277 | -0.030842 |
| 310300 | 0.893384 | -0.073218 | -0.084705 | -0.014140 |
| 310400 | 0.894833 | -0.078827 | -0.079900 | 0.003603 |
| 310500 | 0.905225 | -0.057085 | -0.068852 | 0.024076 |
| 310600 | 0.884359 | -0.065709 | -0.083419 | -0.002319 |
| 310700 | 0.871151 | -0.101838 | -0.100096 | -0.028234 |
| 310800 | 0.964644 | -0.011933 | -0.018311 | -0.309296 |
| 310900 | 0.812059 | -0.122977 | -0.167835 | 0.066185 |

## Fidelity audit

The negative attribution result triggered an audit against paper equations
(1)-(4), Algorithm 1 and the pinned author `SimSiam` builder:

- Both directional cosines carry weight 0.5 and average eligible rows once.
  Independent scalar and autograd oracles cover ordinary and near-zero norms.
- Target-only detach leaves both prediction branches trainable. Removing the
  predictor changes the prediction ports to X_PROJ and removes the component,
  rather than leaving a random frozen predictor.
- The projector has three linear layers, hidden affine BN and ReLU, and final
  non-affine BN. Its sampled final bias is a fixed checkpoint buffer.
  The two-layer predictor has hidden BN and no output BN.
- Each module processes the two views separately in order. Tests check two BN
  updates, sequential running statistics and loss-level cache reuse, and reject
  a concatenated-view mutant.
- All common initial tensors are compared exactly after constructing the same
  full graph for each arm. The no-predictor graph is pruned only afterwards.
- Preprocessing and the MCAR mask are realised once using the downstream seed
  and shared across stages and arms. Actual step traces hash row IDs, masks,
  feature/outcome values and treatment values; actual transformed inputs are
  compared through the existing view trace hook.
- Every stage starts with a distinct optimiser with empty state. Transfer
  checks compare parameters and buffers to the saved pretraining checkpoint;
  downstream heads start identically and projector/predictor state is unchanged
  by fine-tuning.
- Oracle views preserve analytic propensity and both conditional outcome means.
  The largest held-out target error is 4.768372e-7,
  below the imported tolerance of 2e-5.
- Encoder/head initialisation, Adam 0.001 with constant LR, no decay/clipping,
  tabular widths and oracle views agree with the declared departures. The run
  does not reproduce the source image-data/SGD protocol.

No equation, topology, detach, data-pairing or optimiser-reset mismatch was found
by these checks. The local optimisation and data departures remain plausible
reasons for different ablation behaviour; this experiment does not identify
which departure caused it. Both failed attribution bounds remain in the ledger.

**Follow-up.** The [2026-09-15 re-audit](2026-09-15-simsiam-fidelity-audit.md)
did identify them, and also found that the instrument those bounds are built on
cannot express the claim they make. Nothing above is retracted — every check
listed here still passes — but the "plausible reasons" sentence is superseded:
the encoder initialiser and the optimiser both participate, and card section
6.4's spread statistic scores 0.99 on an embedding of exact rank one.

## Validation and evidence

The three declared Tier 1 bases 42, 142 and 242 passed at 128/256 steps,
512 training rows and 512 held-out rows, without directional gates. The
full-card value check covers every answered section-4 entry and the actual
compiled plan. The named mutation checks reject both missing half-weights,
removed target detach, detached prediction, a swapped target view,
concatenated BN passes, affine output BN, a retained no-predictor component and
a count-preserving mask swap. The positional-argument regression was also
observed failing before its compatibility fix.

Ruff lint and format checks pass. Strict mypy reports seven existing
`no-any-return` errors under the local PyTorch 2.2.2 type definitions; the same
seven files fail in a clean checkout of the original PR branch, and no new
file has a type error. They are `xty2/views/pretext.py`,
`xty2/objectives/soft_weighting.py`, `xty2/objectives/curriculum.py`,
`xty2/evaluation/benchmarks/common.py`, `tests/invariants/test_cluster_fixture.py`,
`tests/smoke/test_simmatch.py` and `tests/invariants/test_flexmatch.py`.

Complete machine-readable evidence:

- [All 75 metric vectors, summaries and decisions](results/simsiam-c14090a/simsiam.json)
- [Actual compiled plan](results/simsiam-c14090a/plan.txt)
- [Execution environment](results/simsiam-c14090a/environment.json)
- [Method contract and result ledger](../recipes/simsiam.md)
