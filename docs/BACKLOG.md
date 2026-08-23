# xty2 — Post-P12 research backlog

**Status:** idea backlog, not a build plan
**Reads with:** `DESIGN.md`, `FIDELITY.md`, `PLAN.md`, `CLAUDE.md`

This document is a catalogue of methods and experiments that become possible once
P0–P12 are complete. It is deliberately broader than `PLAN.md`.

It is **not** permission to keep expanding the framework. The default assumption
at Gate 2 remains that xty2 is done: new work should arrive as a reviewed spec
card plus a declarative recipe assembled from existing components, objectives,
views and executors. A new framework abstraction is justified only when a second
real recipe needs it. If a method cannot be expressed cleanly, that is evidence
to discuss the framework boundary, not a reason to widen it opportunistically.

The intended post-P12 workflow is therefore:

1. Pick an approach because there is a real experiment to run.
2. Write `docs/recipes/<name>.md` first and stop for review.
3. Try to express it with the existing xty2 vocabulary.
4. Add the smallest missing component/objective/view if necessary.
5. Add a new framework concept only after the two-consumer rule is satisfied.
6. Require Tier 0, Tier 1 and an explicit Tier 2 target as usual.

The point of this backlog is increasingly **combinatorial** experimentation. It
should not turn into a project to port the other ~35 XTYLearner model classes.

---

## 1. High-priority first tranche

These are attractive early post-P12 implementations because each tests whether
xty2 can express a recognisable method mostly through recombination rather than
new framework machinery.

| Approach | What it exercises | Expected fit |
|---|---|---|
| FixMatch | weak/strong views, confidence pseudo-labels, consistency | mostly recipe/objective |
| FlexMatch | FixMatch plus class-specific curriculum thresholds | pseudo-label policy/objective |
| FreeMatch | self-adaptive confidence thresholding | objective/policy |
| SoftMatch | soft confidence weighting of pseudo-labels | objective/policy |
| UDA | weak/strong augmentation plus distribution consistency | mostly recipe |
| VAT | adversarial local consistency | adversarial view/objective |
| Pi-model | consistency across stochastic realisations | recipe |
| Temporal Ensembling | consistency to historical predictions | artifact/objective |
| Noisy Student | staged teacher -> pseudo-label -> noisy student | existing Program/artifacts |
| SCARF | tabular corruption plus representation contrast | view + contrastive objective |
| SubTab | feature-subset views plus reconstruction | view + reconstruction |
| VIME | corruption mask prediction plus reconstruction | likely one new semantic output |
| SimCLR-style tabular SSL | two views plus contrastive representation loss | objective |
| VICReg | invariance/variance/covariance regularisation | objective |
| Barlow Twins | redundancy-reduction representation learning | objective |
| S4L-style joint training | several SSL and supervised objectives jointly | composite recipe |
| MixMatch | augmentation averaging, sharpening, MixUp, pseudo-labels | deliberate boundary test |
| ReMixMatch | MixMatch plus distribution alignment and anchoring | deliberate boundary test |
| hard/soft EM | alternate latent-treatment inference and fitting | stages/objectives |
| variational latent-treatment model | q(t|x,y), p(t|x), p(y|x,t), ELBO | existing core quantities |
| DragonNet | propensity, outcome and targeted regularisation | components/objective |
| CFRNet | outcome learning plus representation balancing | balancing objective |

A useful initial sequence would be:

1. SCARF
2. FixMatch
3. VIME
4. CFRNet
5. DragonNet
6. MixMatch
7. FlexMatch
8. variational latent-treatment / ELBO
9. explicit treatment-observation/missingness model
10. an S4L-style composite XTY recipe

The tenth experiment is the intended payoff: combine several independently
validated pieces and use xty2's per-objective loss, coverage, gradient norm and
gradient cosine diagnostics to determine whether they actually cooperate.

---

## 2. Semi-supervised treatment-label methods

### 2.1 Pseudo-labelling and self-training

- Plain self-training from `p(t|x)`.
- EMA-teacher self-training.
- Out-of-fold self-training using P10 provenance checks.
- Posterior self-training from `q(t|x,y)` with leakage protection active.
- Ensembled pseudo-labels combining `p(t|x)` and `q(t|x,y)`.
- Agreement-only pseudo-labelling when propensity and posterior agree.
- Soft pseudo-labels rather than argmax labels.
- Temperature-sharpened pseudo-labels.
- Confidence-weighted pseudo-label loss.
- Uncertainty-weighted pseudo-label loss.
- Class-balanced pseudo-labelling.
- Treatment-frequency-aware pseudo-labelling.
- Distribution alignment of pseudo-label marginals.
- Curriculum pseudo-labelling from high to lower confidence.
- Per-class curriculum thresholds.
- Self-adaptive thresholds.
- Calibration-aware thresholds.
- Entropy-based abstention.
- Ensemble-disagreement abstention.
- Monte-Carlo uncertainty abstention.
- Pseudo-label refresh every epoch/stage.
- Frozen pseudo-label tables generated once between stages.
- Iterative pseudo-label/refit programs.
- Multiple treatment imputation rather than one hard pseudo-label.
- Posterior-sampled treatment imputation.

### 2.2 Consistency methods

- Weak/strong consistency.
- VAT.
- Pi-model consistency.
- Temporal Ensembling.
- Mean Teacher variants.
- Consistency under feature masking.
- Consistency under bounded jitter.
- Consistency under feature-subset views.
- Consistency under dropout/noise realisations.
- Consistency under manifold perturbation.
- Consistency between `p(t|x)` and `q(t|x,y)`.
- Symmetric KL between propensity and posterior.
- Jensen-Shannon propensity/posterior matching.
- Consistency of treatment-wise outcome means under permissible views.
- Consistency of CATE estimates across views.

### 2.3 Entropy, posterior and latent-variable objectives

- Entropy minimisation on missing-treatment rows.
- Entropy maximisation where uncertainty should be preserved.
- Posterior KL `KL[q(t|x,y) || p(t|x)]`.
- Reverse KL.
- Symmetric KL.
- Jensen-Shannon matching.
- Mutual-information maximisation between representation and treatment.
- Hard EM.
- Soft EM.
- Generalised EM with parameter-block stages.
- Variational EM.
- ELBO for latent treatment.
- Importance-weighted ELBO / IWAE-style objective.
- Posterior tempering.
- KL annealing.
- Free-bits style posterior regularisation if collapse appears.
- Importance-weighted missing-treatment marginal likelihood.

### 2.4 Multi-model SSL

- Co-training with distinct encoders/views.
- Tri-training.
- Deep mutual learning.
- Student-student consistency.
- Teacher ensembles.
- Cross-pseudo-supervision.
- Co-regularised propensity models.
- Co-regularised outcome heads.

---

## 3. Explicit modelling of treatment-label availability

The v1 framework distinguishes observed and missing `t` but does not model why
`t` is observed. This is a deliberately deferred direction and may eventually
justify a new semantic quantity only after two concrete recipes need it.

Candidate approaches:

- MAR label-selection model `p(r_t=1 | x)`.
- Outcome-dependent selection `p(r_t=1 | x,y)`.
- MNAR selection depending on latent treatment.
- Selection depending on treatment uncertainty.
- Selection depending on acquisition process/site/source.
- Inverse-probability weighting for label availability.
- Doubly robust correction for treatment-label missingness.
- Joint propensity + label-observation model.
- Joint posterior + label-observation model.
- Sensitivity analysis over unverifiable missingness assumptions.
- Pattern-mixture formulations.
- Selection-model formulations.

Do not add a `LABEL_OBSERVATION_PROB`-like port just because one recipe asks for
it. Specify at least two real consumers first.

---

## 4. Self-supervised representation learning

### 4.1 Contrastive and redundancy-reduction methods

- SCARF.
- SimCLR-style contrastive pretraining.
- Supervised contrastive learning on observed-treatment rows.
- Semi-supervised contrastive learning using pseudo-labels.
- VICReg.
- Barlow Twins.
- BYOL-style representation prediction.
- SimSiam.
- DINO-style self-distillation.
- DeepCluster-style clustering.
- Prototype consistency.
- Neighbour consistency.
- Contrastive predictive coding adapted to tabular views where meaningful.

### 4.2 Reconstruction and corruption methods

- Denoising autoencoder.
- Masked feature reconstruction.
- SCARF-style feature replacement without contrastive loss.
- VIME.
- SubTab.
- Feature dropout reconstruction.
- Block masking of related feature groups.
- Corruption-mask prediction.
- Missingness-pattern reconstruction.
- Multi-task reconstruction of raw and derived quantities.
- Treatment-conditioned reconstruction where causally appropriate.

### 4.3 Composite pretraining recipes

- SCARF pretrain -> joint marginal-likelihood fit.
- Masked reconstruction -> TARNet/propensity fit.
- SCARF -> FixMatch.
- SCARF -> Mean Teacher.
- Reconstruction + VAT + missing-treatment marginal likelihood.
- Contrastive learning + propensity supervision.
- Contrastive learning + outcome supervision.
- Contrastive learning + propensity + outcome + marginalisation jointly.
- Masked reconstruction + weak/strong consistency + pseudo-labelling.
- Contrastive pretrain -> Mean Teacher -> OOF pseudo-label refit.
- Multi-pretext S4L-style training where several SSL objectives run together.

---

## 5. Views and augmentation strategies

These should remain schema-aware and functional. Physical/tabular constraints
matter more than reproducing image-augmentation APIs.

- Independent bounded jitter.
- Feature masking/dropout.
- Random feature replacement.
- Feature-subset views.
- Group/block masking of related variables.
- Quantisation noise.
- Gaussian noise scaled by declared measurement uncertainty.
- Noise scaled by `FeatureSpec.perturbation_scale`.
- Empirical resampling from feature marginals.
- Conditional resampling given related features.
- Copula/conditional-density resampling.
- Correlated jitter using empirical covariance.
- Manifold perturbation in `X_REPR`.
- Adversarial perturbation.
- Weak/strong augmentation pairs.
- Random compositions of valid transforms, analogous to tabular RandAugment.
- Primitive-feature perturbation followed by recomputation of derived columns.
- Domain-informed perturbations preserving physical identities.
- Simulated missing-feature patterns.
- Noise based on known sensor resolution/error.
- Feature permutation where scientifically defensible.
- Counterfactual-valid transformations that preserve treatment/outcome semantics.

### 5.1 Row-mixing methods as an explicit boundary test

- MixUp.
- Manifold MixUp.
- CutMix-like feature-block interpolation if meaningful for tabular data.
- MixMatch.
- ReMixMatch.

These may not belong in `ViewSpec`: they synthesize rows and often targets,
rather than creating another realisation of one row while preserving declared
semantics. Do not force them into the existing view abstraction. If two real
recipes need row synthesis, design that concept explicitly.

---

## 6. Causal representation and treatment-effect methods

### 6.1 Neural causal representation learners

- TARNet variants.
- CFRNet with MMD balancing.
- CFRNet with Wasserstein balancing.
- DragonNet.
- DragonNet with targeted regularisation.
- DragonNet + exact missing-treatment marginalisation.
- TARNet + representation balance.
- TARNet + propensity regularisation.
- TARNet + self-supervised representation pretraining.
- Orthogonal neural networks / R-loss neural models.
- CEVAE.
- GANITE.
- Balancing-neural-network variants.
- Treatment-specific adapters or experts.

### 6.2 Meta-learners and orthogonal estimators

- S-learner.
- T-learner.
- X-learner.
- R-learner.
- DR-learner.
- AIPW.
- TMLE.
- Double/debiased ML.
- Orthogonal/R-loss estimators.
- Partialling-out estimators.
- Doubly robust pseudo-outcome regression.
- Multi-treatment extensions where compatible with categorical small-K v1.

### 6.3 Array/cross-fit estimators

Good candidates for the existing `array_fit` and `cross_fit` executors:

- Causal forests.
- Generalised random forests.
- BART-based causal models.
- Linear/lasso DML nuisance models.
- Gradient-boosted nuisance models.
- Random-forest nuisance models.
- Stacked/ensemble nuisance learners.
- Entropy balancing.
- Covariate balancing propensity scores.
- Overlap weighting.
- Propensity truncation strategies.

### 6.4 Semi-supervised causal combinations

- DragonNet + missing-treatment marginal likelihood.
- CFRNet + missing-treatment marginal likelihood.
- R-learner with representation pretraining.
- DR-learner with nuisance models trained using all eligible XTY information.
- OOF `q(t|x,y)` treatment imputation -> orthogonal causal estimator.
- Multiple-imputation treatment labels -> causal estimator pooling.
- Propensity/posterior agreement gating before causal fitting.
- SSL-pretrained encoder -> causal forest on learned representation.
- Mean Teacher propensity -> cross-fit causal nuisance pipeline.

---

## 7. Alternative outcome parameterisations

Once `cnflow` proves the `OutcomeDistribution` protocol, these should mostly be
component swaps. `MissingTreatmentMarginalNLL` should remain untouched.

- Homoskedastic Gaussian.
- Heteroskedastic Gaussian.
- Student-t.
- Laplace.
- Asymmetric Laplace / quantile model.
- Gaussian mixture / mixture-density network.
- Mixture of experts.
- Categorical outcome model.
- Ordinal outcome model.
- Poisson / negative-binomial count model where appropriate.
- Hurdle distributions.
- Zero-inflated distributions.
- Conditional RealNVP.
- Conditional MAF.
- Neural spline flows.
- Mixtures of flows.
- Treatment-specific flows.
- Shared flow with treatment context.
- Energy-based outcome model.
- Joint-energy models over `(x,t,y)` where a concrete recipe justifies them.

A standing regression test for this family should remain: swapping the outcome
parameterisation must not require editing the source of generic marginalisation
objectives.

---

## 8. Encoder and parameterisation swaps

These are components, not new framework concepts.

- Plain MLP.
- Residual MLP.
- Gated residual MLP.
- SELU/self-normalising MLP.
- Wide & Deep.
- Deep & Cross Network.
- FT-Transformer.
- TabTransformer.
- SAINT.
- TabNet.
- NODE.
- Feature-token transformers.
- Periodic embeddings for continuous variables.
- Piecewise-linear numerical embeddings.
- Learned spline embeddings.
- Mixture-of-experts encoder.
- Shared trunk + treatment-specific heads.
- Treatment-conditioned FiLM.
- Treatment-specific adapters.
- Low-rank treatment interactions.
- Monotonic/constrained subnetworks where domain knowledge warrants them.
- Sparse/feature-selecting encoders.

---

## 9. Multi-objective optimisation and scheduling

P3's raw-loss, coverage, gradient-norm and gradient-cosine diagnostics make this
an especially useful experimental axis.

- Fixed objective weights.
- Linear ramps.
- Sigmoid ramps.
- Warm-up periods.
- Step schedules.
- Curriculum introduction of objectives.
- Alternating objectives.
- Objective-specific update frequencies.
- Alternating parameter blocks.
- Freeze/unfreeze curricula.
- Loss normalisation by moving scale.
- Uncertainty weighting.
- GradNorm.
- PCGrad.
- CAGrad.
- MGDA.
- GradVac.
- Dynamic Weight Averaging.
- Nash-MTL.
- Gradient clipping per objective.
- Gradient clipping per component group.

A particularly useful experiment is PCGrad versus a conventional weighted sum
on an intentionally crowded S4L-style recipe, with gradient-cosine traces used
to test whether conflict resolution is solving a real observed problem.

---

## 10. Distillation, ensembling and uncertainty

### 10.1 Distillation

- Ordinary knowledge distillation.
- Self-distillation.
- Mean Teacher variants.
- Born-Again Networks.
- Noisy Student.
- Snapshot distillation.
- Propensity -> posterior distillation.
- Posterior -> propensity distillation where leakage semantics permit it.
- Mutual propensity/posterior distillation.
- Representation distillation.
- Outcome-distribution distillation.

### 10.2 Ensembles and approximate posterior uncertainty

- Deep ensembles.
- Bootstrap ensembles.
- Snapshot ensembles.
- MC dropout.
- SWA.
- SWAG.
- Laplace approximation.
- Heteroskedastic predictive uncertainty.
- Ensemble disagreement as pseudo-label uncertainty.

Do not immediately widen `Realisation.params` into an arbitrary ensemble axis.
Independent recipe fits outside one compiled program may be the correct solution
until two genuine recipes need ensemble members inside the framework.

### 10.3 Calibration and abstention

- Temperature scaling.
- Isotonic regression.
- Platt-style calibration where applicable.
- Vector scaling.
- Dirichlet calibration.
- Expected calibration error diagnostics.
- Brier score calibration diagnostics.
- Calibration-aware pseudo-label thresholds.
- Conformal treatment prediction sets.
- Conformal outcome intervals.
- Uncertainty-gated pseudo-labelling.
- Disagreement-gated pseudo-labelling.

---

## 11. Domain robustness and adaptation

These may require explicit domain/group metadata not present in v1, so keep them
below the abstraction line until concrete experiments justify that data concept.

- CORAL.
- MMD domain alignment.
- DANN / gradient-reversal domain adversary.
- GroupDRO.
- IRM.
- VREx.
- Domain-specific normalisation.
- Domain-conditioned heads.
- Domain-specific adapters.
- Invariant representation learning across aircraft/fleet/site/time domains.
- Test-time entropy minimisation.
- Test-time normalisation adaptation.
- Source-free adaptation.
- Domain-generalised augmentation.
- Importance weighting under covariate shift.
- Label-shift correction.
- Treatment-prior shift correction.

---

## 12. Active learning and label acquisition

These close the loop with an external labelling process and therefore may sit
outside `Program` rather than forcing acquisition into the training abstraction.

- Entropy sampling.
- Margin sampling.
- Least-confidence sampling.
- BALD.
- Query-by-committee.
- Ensemble disagreement.
- Core-set selection.
- BADGE.
- Expected gradient length.
- Uncertainty x diversity acquisition.
- Class-balanced acquisition.
- Treatment-stratified acquisition.
- Propensity/posterior disagreement acquisition.
- Expected model-change acquisition.
- Expected reduction in causal-estimand uncertainty.
- Acquisition targeted at overlap/positivity gaps.

---

## 13. Composite experiments enabled by xty2

These are more important than faithfully porting every named method. They test
the claim that components, objectives, views and stages can be mixed without
rewriting monolithic model classes.

### Representation + semi-supervision

- SCARF + FixMatch.
- SCARF + Mean Teacher.
- SCARF + VAT.
- VIME + exact treatment marginalisation.
- SubTab + Mean Teacher.
- Masked reconstruction + FixMatch.
- VICReg + pseudo-labelling.
- Contrastive pretraining + posterior/propensity matching.

### Semi-supervision + causal regularisation

- FixMatch + CFRNet.
- Mean Teacher + CFRNet.
- FixMatch + DragonNet.
- Mean Teacher + DragonNet targeted loss.
- Exact marginalisation + representation balancing.
- Exact marginalisation + orthogonal/R-loss objective.
- Posterior/propensity agreement + targeted regularisation.

### Staged programs

- SSL pretrain -> joint XTY fit -> EMA teacher -> OOF pseudo-labels -> refit.
- SSL pretrain -> marginal-likelihood fit -> cross-fit causal estimator.
- Joint XTY fit -> posterior pseudo-labels -> OOF refit -> targeted causal fit.
- Mean Teacher -> calibrated pseudo-label table -> DR-learner.
- Contrastive pretrain -> FixMatch -> cross-fit DML.
- VIME pretrain -> DragonNet -> TMLE/AIPW post-fit correction.

### Intentionally crowded S4L-style recipes

- Reconstruction + treatment NLL + outcome NLL + missing-treatment marginal NLL.
- Contrastive representation loss + supervised treatment loss + marginal NLL.
- Reconstruction + VAT + weak/strong consistency + marginal NLL.
- Contrastive + reconstruction + Mean Teacher + pseudo-label loss.
- Posterior KL + marginal NLL + supervised treatment NLL + outcome NLL.
- One of the above with PCGrad/GradNorm to test measured objective conflict.

---

## 14. Deliberate framework stress tests

Some recipes are valuable specifically because they may reveal that a v1
abstraction is too narrow. Treat these as experiments, not automatic feature
requests.

### MixUp / synthetic-row semantics

If MixMatch and another real recipe both require row interpolation, consider a
first-class synthetic-row mechanism rather than pretending row mixing is a
normal `ViewSpec`.

### Explicit missingness mechanism

If two recipes need to model why `t` is observed, consider an explicit typed
statistical quantity for label availability and corresponding provenance rules.

### Ensemble realisations

If two recipes need simultaneous in-program ensemble members, reconsider the
current student/teacher parameter realisation axis. Until then, train ensembles
as independent recipes/runs.

### Domain/group metadata

If two domain-robust recipes need a stable domain variable, extend the batch or
introduce another typed mechanism only after specifying exactly what is observed,
where it is valid and how leakage should be checked.

### Continuous/dose-response treatment

This is explicitly outside v1. It would invalidate assumptions behind exact
small-K marginalisation, candidate-treatment contracts and some estimator APIs.
Treat it as a versioned design exercise rather than another recipe.

### Sequence/time-series inputs

Also outside v1. A sequence encoder is not merely another MLP if the input
contract, views and reconstruction semantics all change. Require concrete
sequence recipes before widening the batch/schema abstractions.

---

## 15. Backlog triage rules

When choosing from this file, prefer an item that does at least one of the
following:

- reuses an existing objective with a genuinely different parameterisation;
- reuses an existing component in a genuinely different training program;
- composes two already validated methods without framework changes;
- exposes a suspected objective conflict that P3 diagnostics can test;
- exercises leakage/provenance rules on a real method;
- forces a meaningful boundary decision backed by two consumers;
- has a published result suitable for a credible Tier 2 target;
- addresses a real XTY experiment rather than filling out a taxonomy.

Deprioritise items that are merely another architecture swap, have no credible
reproduction target, or require large framework expansion before they answer a
research question.

---

## 16. What success looks like after P12

The desired trajectory is not:

> five recipes -> forty recipes -> another monolithic registry.

It is:

> five validated recipes -> a growing library of independently validated
> components/objectives/views -> increasingly ambitious composite experiments.

The strongest evidence that xty2 worked would be an experiment that would have
required a new model class in XTYLearner but becomes, in xty2, a small reviewed
card and a declarative recipe assembling pieces already known to work.
