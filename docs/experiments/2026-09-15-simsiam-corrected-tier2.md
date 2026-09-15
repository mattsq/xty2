# SimSiam Tier 2 after the fidelity re-audit

The [2026-09-15 re-audit](2026-09-15-simsiam-fidelity-audit.md) found the
implementation's transcription faithful, card §6.4's instrument unable to
express the claim it scored, and two declared deviations jointly suppressing the
source's ordering. The repository owner accepted both of its proposals and
directed that deviation 3 move with deviation 7 rather than wait. This is the
rerun those amendments call for.

The card moves to **`reproduced`**: all five of §6.4's bounds pass on ten seeds
of a stream no earlier run or audit had touched. Stop-gradient and the predictor
each account for a measurable share of both how far the optimiser stays from the
cosine's trivial optimum and how much rank the transferred representation keeps,
and pretraining costs 0.009 nat/row downstream against a 0.05 guardrail. This is
this card's local tabular claim on a fixed synthetic fixture with privileged
views. §2's exclusions stand: no ImageNet reproduction, no superiority to any
sibling recipe, no general tabular augmentation policy, and no causal
identification.

## What changed since `c14090a43e2e`

Three amendments, all reviewed before this ran:

1. **Deviation 7 withdrawn.** `mlp_encoder` takes the paper's own initialiser —
   supplement §A's default PyTorch `U(-sqrt(k), sqrt(k))` — instead of the
   `normal std=0.1/sqrt(fan_in)` inherited from the sibling tabular encoders.
   All three projector BatchNorm layers are back in their normalising regime
   (`var/(var+eps)` of 0.89, 0.99989 and 0.99991 against 7.5e-6, 0.078 and
   0.999), and `X_REPR` reaches the projector at 0.57–0.73 instead of 3.8e-4.
2. **Deviation 3 narrowed.** Pretraining runs the source's optimiser: SGD with
   momentum 0.9, base LR 0.05 under its own linear scaling rule (0.025 at batch
   128), weight decay 1e-4 on every parameter layer including the BatchNorm
   scales and biases, and `adjust_learning_rate`'s half-cosine anneal. The
   downstream stage keeps Adam, which is deviation 4's local protocol and has no
   counterpart in the source. What still departs: batch 128 rather than 512, a
   1000-step rather than 100-epoch horizon with the anneal re-based onto it,
   single-process float32, and no synchronised BatchNorm.
3. **Card §6.4's bounds replaced.** The three spread-based bounds are retired —
   `S` scores 0.99 on an embedding of exact rank one, so none of them could
   express attribution — and four paired bounds take their place, two on view
   alignment and two on encoder effective rank. The NLL guardrail is unchanged.

The anneal needed a schedule `DESIGN.md` §6 did not have. `CosineDecay` is
FixMatch's partial turn, which reads 0.707 at the midpoint where the source
reads 0.5, and `WarmupCosine` carries the same half cosine only behind a warm-up
of at least one step, which would shift the horizon. `CosineAnneal` is the
addition, on `paws.md`'s `WarmupCosine` precedent and the `lr-schedules` ledger
row that builds the family when a reviewed card names one.

## Execution and provenance

Replicates ran at `base = 520000 + 100 * i`, `i = 0..9` — a stream disjoint from
the failed run's 310000–310900 and from the audit's probe bases 42–442, so no
seed scoring this result has been seen before. 1,024 training rows at offset 0
with 40 observed treatments, 2,048 held-out rows at offset 10,000, §4's 1,000
pretraining and 3,000 fine-tuning steps at batch 128, and all four arms on every
seed. Terminal pretraining diagnostics used 16 disjoint batches of 128, two
shared oracle views each, eval mode on frozen training BatchNorm buffers.

The run used committed source `946601def994`, tree
`3b09609a25dedcacbcd6f37bc4523a95dbfe99c3`, in a clean detached worktree. That
commit was tagged `simsiam-tier2-source-2026-09-15` in the run environment, but
**the tag is not on the remote**: the session that produced this result could
push branches and not tags (HTTP 403), and the commit was amended into the one
this document lands on, so it is reachable only by that tag. Recreating it needs
someone with tag permission; until then the reconstruction below is the record,
and the tree hash above is what it must produce.

```bash
git checkout simsiam-tier2-source-2026-09-15
python -m xty2.evaluation.runner --recipe simsiam --workers 4
```

Execution used Python 3.11.15, PyTorch 2.14.0 and NumPy 2.4.6 on Linux x86-64,
with four worker processes, one deterministic Torch thread each via
`configure_worker`. No claim of identical floating-point results across PyTorch
releases is made. The complete machine-readable evidence is
[all 88 metric vectors with their per-seed values](results/simsiam-946601d/simsiam.json),
the [actual compiled plan](results/simsiam-946601d/plan.txt) and the
[execution environment](results/simsiam-946601d/environment.json).

Exactly one library file differs between that tag and the commit this document
lands on: `simsiam_study` re-bases the anneal's horizon whenever it shortens the
pretraining budget. At §4's own 1,000 steps it substitutes
`CosineAnneal(steps=1000)` for `CosineAnneal(steps=1000)` — a frozen dataclass
for an equal one — so it cannot move a number here and only affects Tier 1,
which previously ran a 12.8% prefix of the curve at its 128-step budget. The
rest of the diff is this document, the result artifacts, the §6.1 ledger row and
status, and `tests/invariants/test_simsiam.py`'s new assertion that the horizon
and the budget agree, which is what closed that gap. `git diff --stat
simsiam-tier2-source-2026-09-15 HEAD -- xty2/` is the check where the tag
exists. Where it does not, the source tree is this commit with that one hunk
reverted — the `optimiser=replace(...)` argument dropped from `study`'s
`replace(pretrain, steps=pretrain_steps)` and the `CosineAnneal` import with it
— which must hash to the tree named above.

## Required results

| §6.4 bound | Mean ± SE | Assessed | Required | Result |
|---|---:|---:|---:|---|
| `stop_gradient_alignment_gap` | 0.037296 ± 0.002564 | 0.034731 | > 0.0 | pass |
| `predictor_alignment_gap` | 0.037796 ± 0.002549 | 0.035246 | > 0.0 | pass |
| `stop_gradient_rank_gap` | 1.061061 ± 0.099844 | 0.961217 | > 0.0 | pass |
| `predictor_rank_gap` | 0.226683 ± 0.164019 | 0.062663 | > 0.0 | pass |
| `pretraining_outcome_nll_cost` | 0.009385 ± 0.003815 nat/row | 0.013201 | ≤ 0.05 | pass |

`SE` is the sample standard deviation (ddof=1) over the ten replicate values
divided by sqrt(10), and every difference is formed within seed before
aggregation. A bound passes only when the mean satisfies it by at least one
`SE`, which is what `MetricResult.passed` enforces.

Three of the four attribution bounds also hold on every individual seed: the
smallest per-seed alignment gaps are +0.026889 and +0.026897, and the smallest
stop-gradient rank gap is +0.563555. **`predictor_rank_gap` does not.** It is
negative on five of the ten seeds, ranges from −0.349087 to +1.016107, and
passes only as the aggregate criterion it is declared to be, by 0.063 against a
standard error of 0.164. It is much the weakest of the four and the per-seed
table below is the reason it is reported that way rather than summarised. The
predictor's contribution to encoder rank is real in the mean on this fixture and
is not established seed by seed; its contribution to alignment is both. The NLL
guardrail is likewise an aggregate criterion — seven of ten seeds have a
positive cost, the largest +0.020899 — and it is a bound on the cost being
small, not a claim that pretraining helps.

## Arms

Per-seed means over the 16 held-out batches and two oracle views each, then over
the ten seeds, from the terminal pretraining checkpoint in eval mode.

| Arm | Alignment | Encoder eff. rank | Projection eff. rank | Encoder row norm | Outcome NLL | Effect RMSE |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.960898 | 2.9780 | 5.6734 | 1.3456 | 1.212110 | 1.0154 |
| no_stop | 0.998194 | 1.9169 | 2.3682 | 1.7432 | 1.204018 | 0.9210 |
| no_predictor | 0.998694 | 2.7513 | 4.7271 | 1.8080 | 1.213399 | 1.0579 |
| no_pretrain | n/a | n/a | n/a | n/a | 1.202724 | 0.9741 |
| *(untrained encoder)* | n/a | 18.0658 | n/a | 0.6668 | n/a | n/a |

The mechanism runs in the direction the paper describes. Both ablations sit at
0.998 and 0.999 view alignment — the loss within two thousandths of its minimum
possible value of −1, which is paper Figure 2 (left)'s "the optimizer quickly
finds a degenerated solution" in this fixture's terms — while the full arm holds
at 0.961 and keeps more encoder rank than either.

The last row is why §6.4 bounds these contrasts in pairs and sets no absolute
rank floor. An untrained encoder already scores 18.07, so the full arm's 2.98 is
a **loss** of 15.09 (`encoder_rank_retention`, reported and deliberately
unbounded). That is what it should be: this fixture's invariant content under
§6.2's views is at most five-dimensional, so a random 4-layer projection of six
inputs spreading energy across eighteen directions is carrying noise, not
structure, and concentrating below it is the work. An absolute floor here would
have scored the random projection highest.

Downstream, the arms are close and pretraining is not free: `no_pretrain` has
the best outcome NLL of the four at 1.202724 against the full arm's 1.212110.
The guardrail asks that this cost be small and it is. Conditional-mean treatment
effect RMSE is reported for the audit §6.4 requires and card §2 excludes
treatment-effect recovery from the claim; these numbers, which favour `no_stop`
and `no_pretrain`, are consistent with that exclusion rather than evidence
against it.

## Per-seed required values

| Base | align (stop) | align (pred) | rank (stop) | rank (pred) | NLL cost |
|---|---|---|---|---|---|
| 520000 | +0.048608 | +0.048574 | +1.301014 | -0.349087 | +0.020899 |
| 520100 | +0.030521 | +0.031468 | +0.902190 | -0.031430 | +0.015749 |
| 520200 | +0.026889 | +0.026897 | +0.998300 | +0.282938 | -0.002662 |
| 520300 | +0.046455 | +0.046728 | +1.022301 | +0.348559 | +0.020820 |
| 520400 | +0.029782 | +0.030883 | +0.638397 | -0.093395 | +0.018767 |
| 520500 | +0.033338 | +0.034083 | +0.563555 | -0.260778 | -0.011901 |
| 520600 | +0.043538 | +0.044275 | +1.507636 | +0.891740 | +0.009572 |
| 520700 | +0.030991 | +0.031164 | +0.964998 | -0.316389 | +0.019436 |
| 520800 | +0.036891 | +0.037035 | +1.358848 | +0.778564 | +0.008566 |
| 520900 | +0.045947 | +0.046849 | +1.353372 | +1.016107 | -0.005390 |

## The retired instrument, on a tree where the mechanism reproduces

Card §6.4's audit note predicted that correcting the recipe would not rescue the
three bounds this run replaced. It did not, and this run is the sharpest
available test of that, because it is a tree on which the mechanism demonstrably
reproduces.

| Retired bound | Mean ± SE | Its old requirement | Would it have passed? |
|---|---:|---|---|
| `full_projection_spread` | 0.963268 ± 0.003126 | mean − SE ≥ 0.5 | yes, as it always would |
| `stop_gradient_spread_gap` | +0.004710 ± 0.007471 | mean − SE > 0 | **no** (-0.002762) |
| `predictor_spread_gap` | -0.021620 ± 0.002978 | mean − SE > 0 | **no** (-0.024598) |

So on the same ten seeds where all four replacement bounds pass, the old
instrument clears its vacuous absolute bound by a mile and fails both of its
attribution bounds — one of them still with the sign reversed. `S` is pinned at
0.963, 0.959 and 0.985 across the three arms while encoder effective rank moves
between 2.98, 1.92 and 2.75. `tests/invariants/test_simsiam_benchmark.py`
asserts this stays true of the recorded artifact, so the reason for the
replacement cannot quietly stop being checked.

## Validation

Pairing is executed rather than asserted, unchanged from the failed run and
re-checked here: hashed row/mask/value traces and hashed executor view draws
must agree across the three pretraining arms; one fitted population and one
realised MCAR mask are shared by every stage and arm; each stage starts a fresh
optimiser with empty state; the transferred checkpoint's parameters *and*
buffers are compared at the boundary; downstream heads start identically in all
four arms. The oracle views preserved every analytic DGP target to
2.384186e-07, against the imported 2e-5 tolerance,
and no arm produced a zero or near-zero embedding vector.

Tier 0 is 28 tests on this recipe plus 72 on the schedules, all passing, and the
audit's five additions are among them. Twelve named mutants are observed
failing, including the two this change added: the withdrawn encoder initialiser,
which breaks both BatchNorm-regime checks, and a `CosineAnneal` horizon that
does not match its stage's budget. Tier 1's three declared bases pass at the
reduced budget with no directional gate, which §6.3 still forbids.

`uv run ruff check .`, `uv run ruff format --check .` and `uv run mypy --strict`
are clean.
