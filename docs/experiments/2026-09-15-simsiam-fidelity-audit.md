# SimSiam fidelity re-audit

Card §6.4 requires an audit of "gradients, topology, BN, initialisation, views
and optimiser fidelity" before anyone proposes a prospective amendment, and
`CLAUDE.md` requires treating a negative result as an implementation failure
until the method has been audited equation by equation. The
[ten-seed Tier 2 run](2026-09-14-simsiam-tier2.md) failed both attribution
bounds on every replicate. This is that audit.

It is a dated, non-normative diagnostic record. It changes no bound, no §4
value and no ledger row. It proposes two amendments and leaves both for review;
the card carries them at deviation 7 and in §6.4's audit note.

The audit returns three things:

1. **The transcribed method is faithful.** Equations (1)–(4), Algorithm 1's two
   halves, the target-only detach, the pinned builder's topology, the frozen
   projector bias, the two-forward BatchNorm discipline and both ablation
   constructions re-check clean. No equation, detach, pairing or optimiser-reset
   mismatch was found, which is also what the previous audit reported.
2. **The measurement cannot carry its bounds.** Card §6.4's `S` is a
   channel-balance statistic sitting downstream of a non-affine BatchNorm. It
   scores 0.99 on an embedding of exact rank one. Three of the four required
   bounds rest on it.
3. **Two declared departures jointly suppress the mechanism.** Under the card's
   own protocol the full arm's projection effective rank is *below* both
   ablations' in the recorded ten-seed run, and level with one of them in the
   audit's five-seed probe. Under the paper's own initialiser and optimiser it
   is two to three times *above* both, on every seed, with no overlap. Neither
   departure alone restores the ordering — and even the corrected cell still
   fails both spread bounds, which is finding 2 again from the other side.

## Provenance

Audited tree `140d193c67d2` on `claude/simsiam-fidelity-check-2d9rr1`; paper
arXiv:2011.10566v1 including supplement §A; reference implementation
`facebookresearch/simsiam@a7bc1772`. Probes ran on Python 3.11.15, PyTorch
2.14.0, NumPy 2.4.6, Linux x86-64, four CPUs, one deterministic Torch thread
per worker via `configure_worker`.

Probe runs are diagnostics on Tier 1 bases and a shortened budget, not Tier 2
evidence. They are reported as per-seed values with the paired structure the
card requires, and no bound is scored against them.

## 1. What re-checked clean

Re-read against the paper and the pinned `SimSiam.__init__` / `SimSiam.forward`
/ `main_simsiam.train`, with the compiled plan (`plan.render()`) beside card
§3.2 and §4:

- **Equation (4).** Two `CosineFeatureConsistency` terms at weight 0.5,
  `reduction="mean"`, `rows="all"`, forward `X_PRED@a` against `X_PROJ@b` and
  reverse `X_PRED@b` against `X_PROJ@a`. The source computes
  `-(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5`; the plan's two
  half-weighted row-means are the same arithmetic, and the mixer does not
  divide again. This is now checked by value rather than by reading: on the
  fixture's own §6.2 seed and training batch, the stage's mixed total and that
  expression evaluated through `nn.CosineSimilarity` — a different code path —
  agree to the last bit at `-0.0074165324`
  (`test_mixed_pretraining_loss_equals_the_reference_expression`). Deviation 6
  predicted the epsilon difference would be invisible at ordinary norms, and
  that is the measurement.
- **Stop-gradient.** `stop_grad="target"` detaches the target side only, so both
  prediction branches stay trainable and the encoder receives gradient through
  both views. The plan's `detaches` lines agree with the autograd behaviour, and
  `test_directional_values_and_gradients` checks both against an independently
  written scalar oracle at ordinary and near-zero norms.
- **Topology.** Projector `Linear(bias=False) → BN(affine) → ReLU` twice, then
  `Linear(bias=True)` with the bias frozen as a buffer, then
  `BatchNorm1d(affine=False)`. Predictor `Linear(256→64, bias=False) → BN →
  ReLU → Linear(64→256, bias=True)`, no output BatchNorm. Every BatchNorm at
  `eps=1e-5, momentum=0.1, track_running_stats=True`. This is the pinned
  builder, with the card's widths.
- **Forward ordering.** The compiled stage runs `view=corrupted_a` then
  `view=corrupted_b`, each through encoder → projector → predictor. The source
  runs `encoder(x1); encoder(x2); predictor(z1); predictor(z2)`. The
  interleaving differs, but every BatchNorm module still sees view a before
  view b and updates twice per step, so the running buffers are the same
  sequence. `test_two_cached_bn_updates_and_encoder_branch_gradients` recomputes
  both buffers by hand and the concatenated-view mutant is rejected.
- **Ablations.** `no_stop` sets `stop_grad="none"` on both terms and changes
  nothing else, which is the paper's "architectures and all hyper-parameters are
  kept unchanged, and stop-gradient is the only difference". `no_predictor`
  points both terms at `X_PROJ`, keeps the stop-gradient, and prunes the
  component from the graph and the trainable set after the common tensors are
  drawn — the paper's `h = identity`, whose symmetric detached gradient is half
  the undetached one, as `test_no_predictor_half_gradient` checks. Neither is
  approximated by a frozen or zero-learning-rate predictor, which the paper
  measures separately at Table 1(b).
- **Pairing and data policy.** Hashed row/mask/value traces, hashed executor
  view draws, one fitted population and one realised MCAR mask shared across
  arms and stages, fresh optimiser per stage with empty state, checkpoint
  parameters *and* buffers compared at transfer, identical initial downstream
  head states. All executed during the study rather than asserted afterwards.
- **Views.** `OracleSymmetry` preserves every analytic DGP target to
  4.77e-7 against the imported 2e-5 tolerance.

Nothing in this list changed as a result of the audit.

## 2. Card §6.4's `S` cannot express the attribution claim

`S(Z) = sqrt(d) · mean_j std_i(U_ij)` with `U` the row-normalised `Z`. Row
normalisation makes `S` invariant to the embedding's scale, so what it measures
is how evenly the channels share the direction — and the projector terminates in
a non-affine BatchNorm, which equalises exactly that.

Measured directly (`test_projection_spread_cannot_see_directional_collapse`,
`d = 256`, `B = 128`):

| Embedding | Effective rank | `S` |
|---|---:|---:|
| exact rank one, bare | 1.0000 | 0.789 |
| exact rank one, behind the non-affine output BatchNorm | 1.0000 | 0.993 |
| isotropic Gaussian | 98.01 | 0.993 |

An embedding of exact rank one — the strongest collapse there is — passes the
`full_projection_spread >= 0.5` bound on its own, and scores what a rank-98
embedding scores once the output BatchNorm has run. That bound cannot fail for
any embedding leaving a functioning output BatchNorm, and the two gap bounds are
differences of two numbers both pinned near one.

Card §6.4 says "Constant nonzero outputs have S=0", which is true, and paper
§4.1 reads the same statistic at zero for its collapsed run. Both require the
pre-BatchNorm output to be constant across rows. Measured by hooking the layer
during the study's own terminal evaluation, base 42 at the Tier 1 budget of 128
pretraining steps (the §6.2 budget reaches the same regime; this cell is the one
the hook was run on):

| Arm | pre-BN variance | `var/(var+eps)` | `S` | projection effective rank | alignment |
|---|---:|---:|---:|---:|---:|
| full | 5.30e-1 | 0.999972 | 0.948 | 2.818 | 0.955 |
| no_stop | 1.02e+0 | 0.999989 | 0.926 | 2.113 | 0.999 |
| no_predictor | 7.74e-1 | 0.999985 | 0.984 | 1.784 | 0.99985 |

Five orders of magnitude above `eps = 1e-5`. The card imported the paper's
diagnostic without the state that makes it move.

The residual variation it does score is row-norm heterogeneity. After a
functioning non-affine BatchNorm every channel has unit variance, so
`E[||z||^2] = d` and the root-mean-square row norm is fixed at 16 whatever the
embedding does; a *mean* row norm below 16 is dispersion, and dispersion is what
pulls `S` off one. The full arm's mean projection row norm is 13.04, against
14.25 and 14.67 for the two ablations, and it scores lowest of the three for
that reason. Over all thirty (arm, seed) points of the recorded run the
correlation between mean projection row norm and `S` is **r = 0.88**, and it
stays positive within every arm separately (0.86, 0.78, 0.56). The sign of both
gap bounds is set by that, not by collapse.

One diagnostic already logged as informational does separate the arms on the
recorded ten-seed run, in the paper's own direction and on every seed: view
alignment, 0.974 in the full arm against 0.9998 and 0.99996 in the two
ablations. That is Figure 2 (left)'s "the optimizer quickly finds a degenerated
solution and reaches the minimum possible loss of −1", in this fixture's terms —
the ablations sit at the trivial optimum and the full arm does not. Centred
covariance effective rank does *not* separate them on that run (1.566 full
against 1.911 and 1.632, the wrong way round), but it separates them by factors
once §4's protocol is corrected, which is §4 below. A replacement bound belongs
among these, declared prospectively on a seed stream disjoint from every seed
this audit has now seen.

## 3. The encoder initialiser departs from the paper's own instruction

Paper supplement §A, *Initialization*: the convolution and fc layers "follow the
default PyTorch initializers", `U(-sqrt(k), sqrt(k))` with `k = 1/fan_in`, and —
in the same paragraph — "Models with substantially different fc initializers
(e.g., a fixed std of 0.01) may not converge."

Card §4 binds `mlp_encoder: normal std=0.1/sqrt(fan_in)`, inherited from
`vicreg` and `barlow_twins`. At `fan_in = 256` that is `std = 0.00625`: the
constant the paper names, in the paragraph that names it. The projector and
predictor already take the torch defaults, so the encoder is the only component
that departs.

The consequence is not a scale detail, because the projector's first two
BatchNorm layers consume that scale. Across all ten §6.2 model seeds, on a
128-row training batch:

| Quantity | mean | min | max |
|---|---:|---:|---:|
| `X_REPR` mean row norm | 3.84e-4 | 3.55e-4 | 4.49e-4 |
| projector BN1 `var/(var+eps)` | 7.48e-6 | 6.09e-6 | 1.00e-5 |
| projector BN2 `var/(var+eps)` | 0.0781 | 0.0655 | 0.1007 |
| projector output BN `var/(var+eps)` | 0.9988 | 0.9985 | 0.9990 |

`var/(var+eps)` is the fraction of the declared normalisation each layer
actually performs. The source's projector reads a ResNet-50 pooled feature of
order one, where all three sit at one. Here two of the three are floored by
`eps` and act as fixed gains, so the realised head is between paper Table 3(a)
("no BN", 34.6%) and 3(b) ("hidden-only", 67.4%) rather than at its 3(c)
default (68.1%) — an axis the paper measures at 33 accuracy points.

Card §3.2 required a norm check on this objective before pointing it at new
ports, and one was performed. It read `X_PROJ`, where the terminal non-affine
BatchNorm pins `||z||` near `sqrt(d)` whatever the encoder does, and concluded
correctly that DoubleMatch's failure does not transfer. It did not read the port
the *projector* consumes. Both measurements are now executable in
`tests/invariants/test_simsiam.py`, and the committed
`test_initialisation_oracle_kills_the_normalising_projector` fails them the
moment the initialiser is corrected, so deviation 7 cannot be left standing
beside a fixed recipe.

## 4. Which departures suppress the mechanism

`MLPEncoder` already accepts `TORCH_LINEAR_INITIALISATION` — the paper's own
initialiser — so the correction needs no new component. To separate it from
deviation 3's optimiser, all four combinations ran the card's full paired study
at 1000 pretraining steps on Tier 1 bases 42/142/242/342/442, 512 training rows,
512 held-out rows, four evaluation batches, with everything else at §4's values.
Arms, seeds, view draws and initial tensors are paired inside each cell exactly
as §6.2 requires.

Projection effective rank, mean ± standard error over the five seeds, paired
within seed:

| optimiser | encoder initialiser | full | no_stop | no_predictor | full above both on every seed? |
|---|---|---:|---:|---:|---|
| Adam 1e-3, no decay (§4) | `normal std=0.1/sqrt(fan_in)` (§4) | 1.876 ± 0.221 | 1.876 ± 0.059 | 1.599 ± 0.048 | no |
| Adam 1e-3, no decay (§4) | torch default (paper) | 2.265 ± 0.157 | 1.891 ± 0.139 | 1.630 ± 0.042 | no |
| SGD 0.05, momentum 0.9, decay 1e-4 (paper) | `normal std=0.1/sqrt(fan_in)` (§4) | 2.938 ± 0.295 | 1.806 ± 0.105 | 2.708 ± 0.214 | no |
| SGD 0.05, momentum 0.9, decay 1e-4 (paper) | torch default (paper) | **6.229 ± 0.276** | 2.092 ± 0.110 | 3.356 ± 0.304 | **yes** |

Per seed, `full / no_stop / no_predictor`:

| cell | 42 | 142 | 242 | 342 | 442 |
|---|---|---|---|---|---|
| §4 optimiser, §4 initialiser | 1.63/1.95/1.48 | 1.53/1.99/1.64 | 2.72/1.65/1.49 | 1.58/1.87/1.65 | 1.93/1.93/1.73 |
| §4 optimiser, paper initialiser | 2.37/2.40/1.79 | 2.43/1.69/1.56 | 1.73/1.66/1.61 | 2.65/1.72/1.56 | 2.14/1.99/1.63 |
| paper optimiser, §4 initialiser | 2.34/1.90/2.26 | 2.34/1.76/2.37 | 3.28/1.53/2.47 | 3.88/1.69/3.30 | 2.85/2.15/3.14 |
| paper optimiser, paper initialiser | 6.66/2.38/4.33 | 6.14/2.30/3.43 | 6.62/1.83/2.43 | 5.19/1.88/3.15 | 6.54/2.06/3.44 |

Three readings:

- **Only the source-faithful cell reproduces the paper's ordering.** With the
  paper's own initialiser and optimiser the full arm holds three times the
  no-stop-gradient arm's projection rank and about twice the no-predictor arm's,
  on all five seeds and with no overlap. Encoder effective rank says the same
  (3.51 against 1.71 and 2.16).
- **Neither departure alone gets there.** Correcting only the initialiser leaves
  the full arm level with `no_stop`; correcting only the optimiser leaves it
  level with `no_predictor` and blows the encoder up. The card's own cell is the
  worst of the four, and its full-arm encoder effective rank of 1.14 is the
  lowest of its three arms — which is the recorded ten-seed run's 1.05, at a
  smaller fixture.
- **The card's cell reproduces the pathology supplement §A names.** With §4's
  initialiser under the source's SGD, the encoder row norm reaches 72.7 and the
  full arm's `S` falls to 0.246: the "may not converge" case, in this
  architecture, at these widths.

And the finding that matters most for §6.4: **the spread gaps stay negative in
all four cells**, including the one where the mechanism is plainly working.

| cell | `S` full | `S` no_stop | `S` no_pred | stop-gradient gap | predictor gap |
|---|---:|---:|---:|---:|---:|
| §4 optimiser, §4 initialiser | 0.914 | 0.961 | 0.972 | −0.047 | −0.059 |
| §4 optimiser, paper initialiser | 0.937 | 0.971 | 0.981 | −0.034 | −0.044 |
| paper optimiser, §4 initialiser | 0.246 | 0.898 | 0.938 | −0.652 | −0.692 |
| paper optimiser, paper initialiser | 0.952 | 0.970 | 0.985 | −0.018 | −0.032 |

The bottom row is the point. A cell in which the full arm's projection rank is
three times either ablation's on every seed still fails both attribution bounds,
by the same sign and roughly the same margin as the recorded run. Correcting the
recipe would not rescue these bounds; the instrument has to change too, which is
why §6 below proposes both and not just the first.

## 5. Is the test suite fair and achievable?

**Tier 0 is sound.** Independent scalar and autograd oracles at ordinary and
near-zero norms, source topology and frozen-bias checks, hand-recomputed
BatchNorm buffers, the half-gradient identity for the no-predictor arm, full
card-to-plan value reconciliation over more than seventy entries, and the named
mutants §6.3 lists, each committed and observed failing. None of its assertions
is vacuous and none is directional. The audit adds five tests and seven
observed mutants to it, and removes nothing:

| Added test | What it guards | Mutants observed failing |
|---|---|---|
| `test_mixed_pretraining_loss_equals_the_reference_expression` | both half-weights, both directions, the row mean and the absence of a second division, against `main_simsiam.train`'s own line | one half-weight set to 1.0; the forward term retargeted at its own view's projection |
| `test_projector_batchnorm_normalising_regime` | deviation 7's measured BatchNorm regime, at four seeds | the paper's initialiser (committed as `test_initialisation_oracle_kills_the_normalising_projector`); projector BatchNorm `eps` set to 1e-30 |
| `test_encoder_representation_norm_is_four_orders_below_the_source` | the port §3.2's norm check did not read | the paper's initialiser (same committed oracle) |
| `test_projection_spread_cannot_see_directional_collapse` | that `S` maps rank one and rank 98 to the same value, against a hand-written scalar oracle | `S` computed without card §6.4's row normalisation; the rank-one fixture given full rank; the scalar oracle switched to `correction=1` |
| `test_initialisation_oracle_kills_the_normalising_projector` | that correcting the recipe retires deviation 7 in the same change | — (it is the mutant) |

**Tier 1 is fair.** Bases 42/142/242 at a reduced budget assert finiteness,
intended gradients, actual parameter updates, exact pairing and transition
equality, and nothing directional — which is right, because the audit's probe
shows the direction is not stable at this budget. Card §6.3's rule that a new
directional Tier 1 assertion needs evidence on every declared seed and a
reviewed amendment should stay.

**Three of the four Tier 2 bounds are not achievable as written**, for the
reason in §2 above: they are built on an instrument that scores 0.99 on an
exactly collapsed embedding. `full_projection_spread >= 0.5` is the mirror
problem — it is vacuous, and it passed for that reason rather than because the
method worked, which is why it passed beside a full-arm encoder effective rank
of 1.05. The fourth bound, `pretraining_outcome_nll_cost <= 0.05`, is a genuine
guardrail on a quantity that moves, and it should stand.

The benchmark and result artifacts are otherwise in good order: every scalar in
§6's YAML is bound by value, the recorded JSON carries all ten per-seed values
of all 75 metrics, `test_saved_replicates_and_decisions_are_independently_recomputed`
recomputes every mean, standard error and pass decision from the values, and the
failed status is pinned rather than papered over.

## 6. Proposed, and not done here

Both proposals change values that card §4 and §6 bind, and a material recipe
change reruns §6, so both stop for review:

1. **Withdraw deviation 7**: give `mlp_encoder`
   `TORCH_LINEAR_INITIALISATION`, the paper's own initialiser, restoring the
   projector's BatchNorm layers to their normalising regime. This invalidates
   the recorded §6.1 row and requires a fresh ten-seed run.
2. **Replace §6.4's three spread-based bounds** with bounds on statistics that
   move — alignment and centred covariance effective rank are the candidates the
   recorded run already separates — declared prospectively, on a seed stream
   disjoint from bases 42/142/242/342/442 and 310000–310900, before any run.

Whether deviation 3's optimiser should also move toward the source is a separate
question this audit does not settle: the probe shows it participates, but the
card's Adam protocol is a deliberate local-budget choice with its own §5 row, and
changing two departures in one rerun would confound them.

Until both are reviewed, the card stays `deviating`, §6.1's failed row stands
unchanged, and no bound has been widened.

## 7. Outcome

Both proposals were reviewed and accepted the same day, and the repository owner
answered the open question in §6's last paragraph by directing that deviation 3
move to the source's optimiser in the same change. All three are implemented;
the rerun and its result are
[`2026-09-15-simsiam-corrected-tier2.md`](2026-09-15-simsiam-corrected-tier2.md).

Nothing above is retracted. Every measurement in §§1–4 was taken on the tree
this document names, `140d193c67d2`, and the three tables are the evidence the
amendments rest on. What changed after it:

- Deviation 7 is withdrawn; the encoder takes the paper's initialiser.
- Deviation 3 is narrowed to the batch and horizon: pretraining runs the
  source's SGD with momentum 0.9, its linear scaling rule, weight decay 1e-4 on
  every parameter layer, and `adjust_learning_rate`'s half cosine — which
  needed a new `CosineAnneal` schedule, since neither `CosineDecay` nor
  `WarmupCosine` is that curve.
- Card §6.4's three spread-based bounds are replaced by four paired ones, on
  view alignment and encoder effective rank, scored on a fresh seed stream.

Because the owner chose to move both deviations together, this rerun does not by
itself separate their contributions; §4's four cells are what does, and they
stay the evidence for that attribution.
