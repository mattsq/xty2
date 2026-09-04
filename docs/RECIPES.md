# Recipe cards

A card is the reviewed contract between a source method, its xty2 recipe, and
its evidence. The status line inside each card is authoritative.

| Card | Recipe | Primary role |
|---|---|---|
| [`tarnet.md`](recipes/tarnet.md) | `tarnet` | paper-faithful shared encoder and weighted factual-MSE outcome heads |
| [`tarnet_extension.md`](recipes/tarnet_extension.md) | `tarnet_extension` | propensity and exact missing-treatment marginalisation extension |
| [`cnflow.md`](recipes/cnflow.md) | `cnflow` | conditional density head under the same marginal objective |
| [`mean_teacher.md`](recipes/mean_teacher.md) | `mean_teacher` | EMA teacher, views, and consistency scheduling |
| [`cycle_dual.md`](recipes/cycle_dual.md) | `cycle_dual` | staged posterior labels and leakage controls |
| [`ssdml.md`](recipes/ssdml.md) | `ssdml` | array and cross-fit executors |
| [`fixmatch.md`](recipes/fixmatch.md) | `fixmatch` | quota sampling and confidence-gated pseudo-labels |
| [`uda.md`](recipes/uda.md) | `uda` | temperature-sharpened consistency, confidence masking, and training signal annealing |
| [`scarf.md`](recipes/scarf.md) | `scarf` | corruption-based contrastive pretraining |
| [`doublematch.md`](recipes/doublematch.md) | `doublematch` | FixMatch plus representation consistency |
| [`flexmatch.md`](recipes/flexmatch.md) | `flexmatch` | stateful class-adaptive confidence thresholds |
| [`freematch.md`](recipes/freematch.md) | `freematch` | shared self-adaptive thresholds and fairness |
| [`paws.md`](recipes/paws.md) | `paws` | class-stratified support sets and non-parametric soft pseudo-labels |
| [`comatch.md`](recipes/comatch.md) | `comatch` | memory-smoothed pseudo-labels and a pseudo-label-graph contrastive term |
| [`simmatch.md`](recipes/simmatch.md) | `simmatch` | labelled-instance memory and bidirectional semantic-instance pseudo-label propagation |
| [`variational_treatment.md`](recipes/variational_treatment.md) | `variational_treatment` | amortised treatment posterior and discrete-latent ELBO |
| [`meta_pseudo_labels.md`](recipes/meta_pseudo_labels.md) | `meta_pseudo_labels` | one-step labelled-feedback meta-gradient from student to an independently trained UDA teacher |
| [`softmatch.md`](recipes/softmatch.md) | `softmatch` | truncated-Gaussian per-row pseudo-label weights and uniform alignment |

A **Recipe** cell names a callable in `xty2/recipes/`; `none yet (draft)` marks
a card that is still stopped for review, so no such name exists.

Use [`_TEMPLATE.md`](recipes/_TEMPLATE.md) for a new method and stop for review
before implementation.

## Reading a card

- §1–§2: source, estimand, and limits of the claim.
- §3: equations mapped to code. This defines implementation scope.
- §4: machine-readable mechanics. Non-`n/a` values must match the plan.
- §5: departures and framework additions. Debt rows are CI-reconciled.
- §6: predeclared benchmark protocol and result ledger.
- §7: choices forced by source ambiguity.
- §8: review record.

For an ordinary implementation change, read §2–§5 first. Read the full card
when auditing fidelity or interpreting a benchmark.
