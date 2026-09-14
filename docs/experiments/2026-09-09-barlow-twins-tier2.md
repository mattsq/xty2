# Barlow Twins Tier 2 evidence

`docs/recipes/barlow_twins.md` §6.2's ten-seed, four-arm paired study ran at
its declared bases and §4 budget on committed
[`eb47d8bcf810`](https://github.com/mattsq/xty2/commit/eb47d8bcf810), and all
four §6.4 bounds pass. The card moves to `reproduced`, §6.1 carries the row,
and the complete artifact — every per-seed value of every metric — is
[`results/barlow_twins-eb47d8bcf810.json`](results/barlow_twins-eb47d8bcf810.json).

This is the card's project-local mechanism claim on a fixed synthetic fixture
with explicitly supplied, target-preserving oracle views. §2's exclusions stand
unchanged: it is not an ImageNet reproduction, not a claim of superiority to
VICReg, not a general tabular augmentation policy, and not causal
identification.

## Protocol

Replicate `i` ran at `base = 390000 + 100 * i`, `i = 0..9` — a stream disjoint
from Tier 1's bases 419/523/631 — with 1,024 training rows at row offset 0, 40
observed treatments, 2,048 held-out rows at offset 10,000, and §4's
1,000 pretraining and 3,000 fine-tuning steps at batch 128. All four §6.2 arms
ran on every seed: `full`, `diagonal_only` (eq. (1)'s `lambda` set to exactly
zero with the objective still declared), `no_pretrain`, and the contextual
`vicreg` arm at its own reviewed (25, 25, 1). No seed, checkpoint, threshold,
weight, architecture or budget selection was performed, and no bound was
changed after seeing a number.

```bash
uv run --no-sync python -m xty2.evaluation.runner --recipe barlow_twins
```

Python 3.11, PyTorch 2.14.0+cpu, four worker processes each pinned to one
deterministic Torch CPU thread by `configure_worker`.

Pairing is checked rather than assumed, four ways: the executor's own hashed
row ids and corrupted view tensors over the whole of pretraining must agree
across the three pretraining arms; every logged objective value and diagnostic
at pretraining step 0 must agree bit for bit between `full` and
`diagonal_only`; the stage seeds must agree, which is what says `no_pretrain`
landed on stage index 1's stream through `STREAM_STRIDE`; and the row stream,
both view tensors and the treatment mask are materialised through the loader
entry points the executor itself calls and compared element by element. The
contextual arm receives the other arms' encoder and head tensors by copy,
because `BarlowTwinsProjector` and `VICRegExpander` consume construction RNG
differently.

Embeddings come from the terminal pretraining checkpoint, restored into the
graph after the downstream numbers are taken and evaluated with the hidden
BatchNorm in eval mode on its frozen training buffers, over §6.2's 16 disjoint
held-out batches of 128 rows and two independent oracle-symmetry draws each.
§3.1's stateless correlation is imported from `xty2.objectives.barlow_twins`
rather than transcribed a second time.

## Required results

`SE` is the sample standard deviation of the ten replicate values divided by
sqrt(10). A bound passes only when the mean satisfies it by at least one `SE`,
which is what `MetricResult.passed` enforces.

| §6.4 bound | Mean ± SE | Assessed | Required | Result |
|---|---:|---:|---:|---|
| 1. Full diagonal alignment `a` | 0.989238 ± 0.000688 | 0.988550 | ≥ 0.5 | pass |
| 2. Full active fraction `f` | 1.000000 ± 0.000000 | 1.000000 | ≥ 0.9 | pass |
| 3. Paired redundancy gap `r_diag_only − r_full` | 0.539975 ± 0.013896 | 0.526079 | ≥ 0.01 | pass |
| 4. Outcome transfer cost `NLL_full − NLL_no_pretrain` | −0.009835 ± 0.006073 nat/row | −0.003762 | ≤ 0.05 | pass |

Bounds 1–3 hold on every individual seed as well as in the mean: the lowest
per-seed alignment is 0.983751, the active fraction is exactly 1.0 on all ten,
and the smallest paired redundancy gap is 0.467041. Bound 4 is an aggregate
criterion and does not require every seed's cost to be negative; two are
positive, the largest being +0.022661, and both are retained above.

## Arms

Per-seed means, averaged over the 16 held-out batches and then over the ten
seeds. `a` is §6.4's mean diagonal, `r` its mean squared off-diagonal per
ordered pair, `f` its active-coordinate fraction, `var` the mean raw population
variance of branch A's embedding, and `top eig` the covariance's top eigenvalue
share.

| Arm | a | r | f | var | top eig | D/d | Treatment NLL | Outcome NLL |
|---|---|---|---|---|---|---|---|---|
| full | 0.989238 | 0.012524 | 1.000000 | 0.043624 | 0.034608 | 1.27e-04 | 1.141429 | 1.229854 |
| diagonal_only | 0.999667 | 0.552499 | 1.000000 | 0.951955 | 0.733971 | 1.41e-07 | 1.288219 | 1.242519 |
| no_pretrain | n/a | n/a | n/a | n/a | n/a | n/a | 1.246338 | 1.239689 |
| vicreg | 0.999089 | 0.016587 | 1.000000 | 0.594211 | 0.045337 | 9.46e-07 | 1.206175 | 1.230219 |

The mechanism runs in the direction the paper describes, and §6.4's first two
bounds are what stop that reading from being available to a collapsed
embedding. Adding the off-diagonal penalty costs about one point of diagonal
alignment and removes forty-four times the mean squared off-diagonal
correlation, while deactivating no coordinate at all: `f` is exactly 1.0 in
every arm on every seed, so the redundancy improvement is not bought by
switching coordinates off. Raw variance and eigenvalue concentration say the
same thing from the other side — the diagonal-only arm holds about twenty-two
times the raw variance while concentrating 73% of its covariance energy in one
direction, where the full arm holds 3.5%. Card §6.4's rank note is visible
here: at `B = 128` and `d = 512` a centred cross-product has rank at most 127,
so the full arm's residual redundancy is a floor of the protocol rather than a
failure of the objective, and no bound asks for zero.

Informational comparisons, none of which carries a bound:

| Diagnostic | Mean ± SE |
|---|---:|
| Pretraining treatment NLL cost | −0.104909 ± 0.071304 nat/row |
| Contextual VICReg redundancy gap `r_vicreg − r_full` | 0.004063 ± 0.000173 |
| Contextual VICReg alignment gap `a_full − a_vicreg` | −0.009850 ± 0.000689 |
| Contextual VICReg outcome NLL cost | −0.000365 ± 0.006638 nat/row |

The contextual arm reaches comparable redundancy by a different route at about
fourteen times the full arm's raw variance, and the two arms' outcome NLLs are
indistinguishable at this budget. No claim of superiority either way follows,
and §6.2 does not ask for one. Conditional-mean treatment-effect RMSE (0.908
full, 1.060 diagonal-only, 0.876 no-pretraining, 1.056 VICReg) and absolute ATE
error are reported for the audit §6.4 requires; card §2 excludes treatment
effect recovery from the claim, and these numbers are consistent with that
exclusion rather than evidence against it.

## View contract

Card §6.2's five checks on the held-out views run before any metric is read and
stop the study rather than reporting a number: maximum DGP target error against
the imported 2e-5 tolerance, nonidentity, independent branch draws, the
preserved fields the recipe itself declares, and a training-only donor pool and
fitted scale. Observed: maximum target error 2.62e-07, mean changed coordinates
3.031 of six, and a distinct-row fraction of 0.999902 between the two branches.

That last number is why the independence check is an average over the 16
batches and not a bound on the worst one. Two independent draws agree on a row
whenever they draw the same donor under the same two symmetry bits — about one
row in four thousand — so a per-batch threshold would have been a coin flip on
that tail rather than a statement about independence. A per-batch check that
the two branches are not the *same tensor* is kept, since that is what an
accidentally shared generator seed produces.

## Mutation evidence

Each mutation was applied on its own at a reduced 20/30-step budget, the
replicate was rerun, the named guard failed, and the mutation was reverted.

| Mutation | Guard that failed |
|---|---|
| `no_pretrain` keeps stage index 0's execution seed | the arms ran `joint_fit` at the same stage seed |
| The contextual arm keeps its own encoder and head draws | every arm starts from the same transferred tensors |
| The ablated arm replaces `lambda` with `lambda` | `full` and `diagonal_only` pretrained different encoders |
| One generator seed for both held-out branches | a batch's two branches are not the same tensor |
| The held-out views are the clean rows | every row of a held-out batch was transformed |
| The view donor pool is refitted on the held-out rows | the views were drawn from the fitted training rows |
| Pretraining also fits the outcome head | pretraining logged exactly the two declared objectives |
| ... with that objective-name guard disabled | the pretraining checkpoint holds no head tensor |
| Fine-tuning keeps the projector in its objectives and trainable set | the executor fed pretraining the same rows and views |
| ... with both stream guards disabled | fine-tuning left every projector tensor bit-identical |
| One held-out batch is dropped from the average | 16 held-out observations per seed |
| The ablated arm corrupts its second view differently | the executor fed pretraining the same rows and views |
| The off-diagonal mask keeps the diagonal | the mask selects `d * (d - 1)` ordered pairs |
| The off-diagonal mask keeps one triangle only | the mask selects `d * (d - 1)` ordered pairs |
| The diagnostics skip the checkpoint restore | the terminal pretraining checkpoint was restored |

Two mutations are caught first by a stream guard that notices any change to
what the executor fed pretraining. Those rows are listed twice: once as the
first guard that fires, and once with that guard deliberately disabled, so the
deeper assertion is seen to fail on its own rather than assumed to be live
behind it. Two further mutations — pretraining a head, and fine-tuning the
projector — are rejected by `compile` itself before the module is reached when
they are written as a bare `trainable` change, because `DESIGN.md` §8.4 refuses
a stage that trains a component no active objective depends on; the versions in
the table add the objective too, which is what gets past the compiler and into
the guards above.
