# VICReg paired view experiment: protocol before execution

Requested after review of PR #48 at `73a891b15a3ad3a7260ff72aa01bfbcb09f86992`.
The source tree of that commit is `b55d7109c1032f72ca4c5689ee86388766a9fb17`.
This is an authorised diagnostic extension, not a replacement of the recipe or
its failed result. No acceptance threshold is changed. No run of this protocol
has been inspected when this document is committed.

## Question and intervention

Does preserving the fixture's target information resolve the spread shortfall
with the same VICReg implementation, weights, architecture and optimisation?

Compare the existing two independent `FeatureCorruption(0.6, all)` views with
two independent draws of an oracle symmetry of the declared DGP. The latter
uses the fixture equations, not treatment/outcome observations, to preserve
both Bayes treatment probability and both conditional outcome means.

On original-scale features each oracle view:

1. With probability 1/2, reflects `(x0,x1,x2,x3)` across the plane orthogonal
   to `v=(3,5,0,-8)`: `x_signal -= 2*v*(x_signal @ v)/(v @ v)`.
2. Independently with probability 1/2, replaces `x4` by `-x4`.
3. Always replaces `x5` with a training-population marginal donor.

The first reflection preserves `sum(x0..x3)`, `0.5*x0-0.3*x1`, and `x2`.
The second preserves `x4**2`. Consequently the propensity, outcome baseline
`0.5*x0-0.3*x1+0.2*(x4**2-1)`, and effect `1+0.5*tanh(x2)` are unchanged.
The reflection is orthogonal to both cluster centres and preserves their
isotropic Gaussian noise law. The sign flip preserves x4's symmetric law;
x5 is independent of the targets. Train-only scaling is undone/reapplied.
This is not the identity: it changes three coordinates on average (1.5 from
the first reflection, 0.5 from the sign flip, and one donor coordinate), as
does the baseline's exact three-of-six replacement. Equal counts do not make
perturbation geometry equal; this tests a view policy, not a single scalar
notion of augmentation strength.

**Scope:** oracle access to DGP symmetries is a deliberate experimental
advantage. A success demonstrates mechanism under valid views; it does not
provide a deployable automatic view selector for arbitrary tabular data.
In particular x4 is outcome-relevant despite being cluster-independent.

## Fixed execution and evaluation

- Ten bases `190000+100*i`, i=0..9, with every data/model/execution offset from
  card section 6.2 unchanged. These repeat the original seeds to isolate views.
- For each policy run full, no-variance and no-covariance; share one
  no-pretraining arm because it has no augmentation. Seven fits per base.
- Full budgets: 1000 pretraining, 3000 downstream, batch 128. Fresh optimiser
  per stage, same initial tensors, training data/mask and sampled row streams.
- Within each policy, compare realised row/view streams across the three
  pretrained arms using the registered benchmark's checks. Across policies,
  rows, initial states and downstream streams are paired; views intentionally
  differ. Log actual training row/view digests for all steps as well.
- Use the benchmark's terminal pretraining checkpoint, frozen BN buffers,
  16 held-out batches, two independent views per batch, and clean downstream
  evaluation. Cross-evaluate every pretrained arm under BOTH held-out view
  policies to separate training-policy effects from evaluation-policy effects.
- Primary comparison uses each training policy's own evaluation views.
  Report the original four targets with the exact original bounds and SE
  rules: spread >=0.5, variance gap >=0.1, redundancy gap >=0.01, outcome NLL
  cost <=0.05. Also report all original diagnostics and paired policy changes.
- Record propensity and conditional-outcome changes, changed coordinate count,
  and feature displacement. Oracle preservation must hold within 2e-5 in
  float32; both views must actually change features and differ from each other.
- Record held-out invariance, variance and covariance losses and an output
  rescaling curve for full arms; these are diagnostic, not selection criteria.
- Save every per-seed result plus source commit, protocol SHA256, environment,
  seed identities and the compiled plan. No checkpoint/rate/width search.

## Decision rule

If the oracle policy meets the unchanged four bounds and spread increases
under both evaluation policies, this supports an augmentation-induced failure
rather than an arithmetic failure or a pure evaluation-distribution artifact.
If it fails, record the failure; do not choose another symmetry or relax a bound
after inspecting the result. A favourable result alone does not relabel the
unchanged all-feature-corruption recipe `reproduced`. Any new normative scope
must explicitly retain the old failure and declare the oracle-view advantage.

Run `python -m experiments.vicreg_views --output runs/vicreg-views --workers 6`.
The fixed ten-replicate output is separate from registered Tier 2 and its ledger.
