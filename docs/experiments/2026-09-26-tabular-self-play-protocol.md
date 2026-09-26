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
