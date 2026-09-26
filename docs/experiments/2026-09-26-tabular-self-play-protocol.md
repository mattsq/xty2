# Tabular self-play: validation tuning and confirmation protocol

Non-normative experiment for
[`proposals/tabular-self-play-ssl.md`](../proposals/tabular-self-play-ssl.md)
§8. It is not a recipe, a Tier 2 reproduction, or a reproduction of the
source paper's generator. This document is committed after the validation
tuning stage and **before** any confirmation run. No test-split result of
this design had been computed when it was written.

## 1. What changed from the pilot

The pilot (proposal §8, reproducible at commit `2497a40`) evaluated its only
readout on the test split, had no validation split, compared against an
untuned heuristic, and trained for 64 updates. The driver now:

- draws 1,024 training, 512 validation and 512 test rows per fixture seed,
  with standardization fitted on training rows only;
- evaluates exactly one declared split per run (`Config.endpoint`);
- adds a no-pretraining arm (`none`: the paired initial encoder, zero
  updates) and a raw-feature probe (`raw_bce`, the same logistic probe on
  standardized `X`) as transfer references;
- records secondary endpoints on the same split: clean single-column masked
  MSE (full context, no corruption), mean encoder feature standard deviation,
  dead-unit fraction, and effective rank;
- summarizes probe traces: target/context selection frequencies, final
  normalized entropy, per-round rank correlation of reward with gradient norm
  (scale selection) and with current loss, and the share of positive signed
  alignments. The confirmation keeps per-round marginals for every run and
  full candidate-level traces for the first seed.

The fixed arms may skip their discarded probes during tuning; a Tier 0 test
shows that training is bit-identical either way. The confirmation computes
the probes in every arm except `none`, so gradient calls are matched.

## 2. Validation tuning (completed)

`python -m experiments.tabular_self_play_tuning` ran from the clean commit
`8cd5b58` with tuning seeds `320000 + 100*i`, `i = 0..4`. Outputs are in
[`results/tabular-self-play-tuning/`](results/tabular-self-play-tuning/).
Only validation BCE was computed; the test split was never evaluated.

**Stage 1, learner budget** (uniform policy, mean validation BCE over the
three fixtures and five seeds; lower is better):

| Updates | 64 | 256 | 1024 |
|---|---:|---:|---:|
| Validation BCE | 0.4146 | **0.3266** | 0.3274 |

By fixture, clean masked MSE on validation rows was 0.95/0.53/0.39
(dependent), 0.95/0.70/0.44 (interaction) and 1.02/1.01/1.01 (independent)
at 64/256/1024 updates. The pilot's 64-update learner therefore barely
learned its pretext task, which by itself makes the pilot's transfer table
uninformative about the controller. The rule chose 256 updates; 1,024 was
within 0.001 and better on both structured fixtures, and worse on the
independent fixture, where pretraining cannot find structure.

**Stage 2, fixed policy** at 256 updates. The grid has 12 policies: uniform,
the pilot heuristic, each context alone, and each target alone. Selected
(lowest mean validation BCE, ties to the earlier entry):

| Fixture | Uniform | Selected | Selected BCE | `none` | raw `X` |
|---|---:|---|---:|---:|---:|
| Dependent | 0.2198 | `fixed_dependent` | 0.2114 | 0.3936 | 0.2113 |
| Interaction | 0.3374 | `target:0` | 0.3185 | 0.5134 | 0.2407 |
| Independent | 0.4227 | `target:2` | 0.3643 | 0.3958 | 0.1791 |

Three facts shape the interpretation of the confirmation:

1. Pretraining helps the frozen encoder over its random initialization on
   the structured fixtures, but on the independent fixture uniform pretraining
   is *worse* than no pretraining.
2. The raw-feature probe matches or beats every frozen encoder. The primary
   endpoint therefore measures the information a 16-unit frozen encoder
   retains, not an improvement over the input.
3. The tuned policy was chosen from 12 candidates on five seeds, so its
   validation margin carries selection bias. Fresh confirmation seeds and
   data draws remove that bias from the confirmation estimate.

Adaptive-controller settings (EMA 0.2, probability floor 0.2, probes every
eight updates after an eight-update warm-up, clamp ±5) were not tuned. The
fixed control received a label-based tuning budget that the label-free
adaptive arm did not; this asymmetry favours the control.

## 3. Confirmation protocol (frozen)

- **Command:** `python -m experiments.tabular_self_play --output
  docs/experiments/results/tabular-self-play-confirmation --workers 4`, from
  the clean commit that adds this document. The script refuses a dirty tree
  or an existing output.
- **Configuration:** `selection.json` above: 256 updates, batch 64, probes
  every 8 after 8 warm-up updates, 64 labels, `endpoint = test`.
- **Seeds:** ten bases `330000 + 100*i`, `i = 0..9`, disjoint from tuning.
  Each base determines the fixture draw (`base+1`), initial weights
  (`base+2`), row order, probe batches and view noise; all arms are paired
  within a base.
- **Arms:** `none`, `uniform`, `fixed_dependent`, `tuned` (the per-fixture
  selection above), `alignment`, `current_loss`, `shuffled`.
- **Primary endpoint:** test BCE of the frozen-encoder logistic probe.
- **Primary contrasts:** per fixture, paired `alignment − uniform` and
  `alignment − tuned`, reported as the mean with a two-sided 95% Student t
  interval over the ten seeds (`t = 2.2622`, df 9). Secondary contrasts:
  `alignment − shuffled`, `alignment − current_loss`, `tuned − uniform`,
  `uniform − none`. `analyse` in the driver computes all of these.
- **Decision rule:** the adaptive arm is *supported* only if the upper
  bound of both primary intervals is below zero on **both** the dependent and
  interaction fixtures. The independent fixture is reported but is not
  required, because no transferable dependency structure exists there. Any
  other outcome is recorded as not supported, and a supported outcome still
  requires the trace checks below before any integration proposal.
- **Trace checks (reported, not gates):** mean `reward_vs_norm` and
  `reward_vs_loss` for the alignment arm, final selection entropy, maximum
  single-task frequency, and whether alignment concentrates on targets that
  the validation tuning found useful.
- **No changes after inspection:** no rerun with other seeds, budgets,
  controller settings, fixtures or endpoints. Any follow-up is a new,
  separately registered protocol.

## 4. Confirmation result (2026-09-26)

Run from the clean commit `bacdebe` that added §3, with the command stated
there. Outputs are in
[`results/tabular-self-play-confirmation/`](results/tabular-self-play-confirmation/)
(`analysis.json` holds the predeclared contrasts). Every arm except `none`
used 256 update gradients and 744 probe gradients per seed.

**Decision: not supported.** No primary interval lies below zero on either
structured fixture.

Mean test BCE over ten seeds (lower is better):

| Fixture | raw `X` | none | uniform | tuned | alignment | current loss | shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dependent | 0.2348 | 0.3339 | 0.2160 | 0.2136 | 0.2204 | 0.2237 | 0.2163 |
| Interaction | 0.2554 | 0.4518 | 0.3459 | 0.3683 | 0.3391 | 0.3633 | 0.3469 |
| Independent | 0.2035 | 0.4739 | 0.4869 | 0.4705 | 0.4884 | 0.4868 | 0.4846 |

On the dependent fixture, `tuned` is `fixed_dependent`, the same policy as
that arm, so the two rows are identical.

Paired contrasts, mean [95% t interval]:

| Contrast | Dependent | Interaction | Independent |
|---|---|---|---|
| alignment − uniform | +0.0044 [−0.0010, +0.0099] | −0.0069 [−0.0276, +0.0138] | +0.0015 [−0.0105, +0.0135] |
| alignment − tuned | +0.0068 [+0.0008, +0.0127] | −0.0292 [−0.0890, +0.0306] | +0.0179 [−0.0145, +0.0502] |
| alignment − shuffled | +0.0042 [−0.0039, +0.0122] | −0.0079 [−0.0278, +0.0121] | +0.0038 [−0.0198, +0.0274] |
| alignment − current loss | −0.0033 [−0.0106, +0.0040] | −0.0243 [−0.0483, −0.0003] | +0.0016 [−0.0136, +0.0168] |
| tuned − uniform | −0.0023 [−0.0044, −0.0002] | +0.0223 [−0.0436, +0.0882] | −0.0164 [−0.0538, +0.0210] |
| uniform − none | −0.1180 [−0.1600, −0.0760] | −0.1058 [−0.1690, −0.0426] | +0.0129 [−0.0289, +0.0547] |

Findings:

1. **Transfer.** Alignment was significantly worse than the tuned policy on
   the dependent fixture and indistinguishable from uniform and shuffled
   everywhere. Its only interval excluding zero in its favour is against the
   current-loss controller on the interaction fixture.
2. **Tuned control.** Validation selection transferred on the dependent
   fixture only. The interaction selection (`target:0`) had clean masked MSE
   0.94 versus 0.71 for uniform on fresh draws and lost transfer; the
   12-policy, five-seed search overfit, as §2 anticipated.
3. **Selection traces.** Alignment did select structured tasks. On the
   dependent fixture it gave 81% of updates to the four correlated targets
   (uniform: 67%) and 19% to the two noise columns. Current loss did the
   opposite (45% to the noise columns), because unpredictable columns keep
   high loss. On the interaction fixture alignment favoured column 3, the
   `tanh` interaction (35%). On the independent fixture it stayed near
   uniform. Final normalized entropy was 0.91–0.97 and the largest
   single-task frequency was at most 0.13, so there was no collapse onto one
   candidate.
4. **Reward diagnostics.** The alignment reward had mean per-round rank
   correlation 0.36–0.46 with gradient norm and −0.38 to +0.07 with current
   loss. The shuffled arm's raw rewards show the same norm correlation
   (0.38–0.58), so gradient scale is part of the reward itself; the
   controller is not a disguised loss controller.
5. **Representation.** Pretraining raised feature spread and lowered
   effective rank relative to `none`; no arm collapsed (dead-unit fraction at
   most 0.125). On the independent fixture pretraining did not beat the
   random encoder, and every frozen encoder remained well behind the raw
   probe on all fixtures.

Interpretation: at this scale, the Eq. 2 reward identifies the dependent
columns and avoids the noise targets that a current-loss reward prefers,
but that choice did not improve the frozen-encoder transfer endpoint over a
uniform task mixture. The result is limited to three six-column synthetic
fixtures, a 16-unit encoder, 256 updates and one controller setting. The
raw-feature probe beating every encoder means this endpoint rewards
information retention, which may not be what better task selection
improves. Per §3, no rerun or alternative setting is part of this result.
