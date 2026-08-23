# xty2 — Prior art and architecture notes

**Status:** non-normative research notes
**Reads with:** `DESIGN.md`, `FIDELITY.md`, `BACKLOG.md`

This document records observations from other research/codebases that are useful
when reviewing xty2's architecture. It is deliberately **non-binding**: design
rules live in `DESIGN.md`, fidelity requirements live in `FIDELITY.md`, and
future methods live in `BACKLOG.md`.

The purpose here is narrower: preserve evidence about how other systems have
solved similar decomposition problems, where those abstractions have worked, and
where they have started to leak. A prior-art observation should not become an
xty2 abstraction merely because another framework has one. The two-consumer rule
in `DESIGN.md` still applies.

---

## 1. SemiLearn / USB

Repository: https://github.com/microsoft/Semi-supervised-learning

Paper: Wang et al., *USB: A Unified Semi-supervised Learning Benchmark*,
https://arxiv.org/abs/2208.07204

SemiLearn is useful prior art because it supports a large family of SSL methods
through one common training framework and has already encountered many of the
same repeated mechanisms that xty2 expects to meet after P12: pseudo-labels,
confidence masking, distribution alignment, EMA teachers, moving statistics,
memory banks, multiple views and algorithm-specific auxiliary losses.

The important conclusion from reading the implementation is that SemiLearn is
**modular at a different level from xty2**. It successfully factors repeated
procedural machinery out of individual papers, but the named algorithm class and
its imperative `train_step()` remain the primary unit of composition.

### 1.1 Core execution model

`semilearn/core/algorithmbase.py` defines `AlgorithmBase`. It owns most of the
experiment lifecycle:

- dataset construction;
- labeled and unlabeled dataloaders;
- model and EMA model construction;
- optimizer and scheduler;
- common supervised and consistency losses;
- the epoch/step training loop;
- evaluation;
- checkpointing;
- registration and invocation of hooks.

Source:
https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/core/algorithmbase.py

The main loop is approximately:

```text
before_train_step hooks
        ↓
algorithm.train_step(batch)
        ↓
after_train_step hooks
```

The standard `ParamUpdateHook` takes the total loss returned by `train_step`,
performs backward/gradient clipping/optimizer step/scheduler step, and clears the
gradients. EMA, evaluation, checkpointing, logging and timing are similarly
implemented as lifecycle hooks.

Sources:

- https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/core/hooks/param_update.py
- https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/core/hooks/ema.py

This lifecycle-hook design is strong prior art. Backward/step, EMA maintenance,
checkpointing, evaluation and logging are genuinely orthogonal to the
statistical definition of an SSL method and do not belong duplicated inside
each algorithm.

### 1.2 SemiLearn uses hooks for two different jobs

The same hook mechanism is also used *inside* algorithm `train_step()` methods
for reusable SSL mechanisms. An algorithm can explicitly invoke a named hook,
for example:

```python
self.call_hook("dist_align", "DistAlignHook", ...)
self.call_hook("masking", "MaskingHook", ...)
self.call_hook("gen_ulb_targets", "PseudoLabelingHook", ...)
```

So there are effectively two roles:

| Hook role | Examples |
|---|---|
| lifecycle / infrastructure | parameter update, EMA, evaluation, checkpointing |
| reusable SSL mechanism | pseudo-label generation, masking/weighting, distribution alignment |

Source:
https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/core/algorithmbase.py

The second role is the more interesting one for xty2. It demonstrates that
several things repeatedly appear between model predictions and losses and are
worth reusing independently of the named paper.

### 1.3 FixMatch is the cleanest example of the intended decomposition

SemiLearn's FixMatch registers:

- `PseudoLabelingHook`;
- `FixedThresholdingHook`.

Its `train_step()` still explicitly performs the weak/strong forwards, computes
supervised CE, obtains detached weak-view probabilities, optionally runs
distribution alignment, computes a confidence mask, generates pseudo-labels,
computes strong-view consistency loss and combines the losses.

Source:
https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/fixmatch/fixmatch.py

The extracted mechanisms are genuinely reusable:

- `PseudoLabelingHook` converts logits/probabilities into hard or soft targets
  with optional temperature;
- `FixedThresholdingHook` converts confidence into a per-row weight/mask.

Sources:

- https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/hooks/pseudo_label.py
- https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/hooks/masking.py

The resulting architecture is approximately:

```text
                     ┌─ PseudoLabelingHook
FixMatch.train_step ─┼─ MaskingHook
                     ├─ optional DistAlignHook
                     └─ generic CE / consistency losses
```

The important distinction from xty2 is that **FixMatch remains the
orchestrator**. Those pieces cannot simply be declared in configuration and
compiled into FixMatch. The Python statements in `FixMatch.train_step()` still
define the execution semantics.

### 1.4 Stateful prediction mediation is a real recurring concept

SemiLearn's strongest architectural lesson for xty2 is not the generic hook
system. It is the repeated existence of mechanisms of the form:

```text
predictions
    ↓
stateful transform
    ↓
new predictions / targets / row weights / relationships
    ↓
objective
```

Distribution alignment is implemented with reusable stateful hooks. For
example, `DistAlignEMAHook` maintains moving estimates of the model and target
class distributions; `DistAlignQueueHook` maintains queue-based estimates.

Source:
https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/hooks/dist_align.py

FreeMatch goes further. Its custom thresholding hook maintains:

```text
p_model
label_hist
time_p
```

and updates them from the stream of unlabeled predictions before computing
class-specific thresholds.

Source:
https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/freematch/utils.py

This is important evidence for the post-P12 backlog. Not every reusable SSL
mechanism is naturally a model `Component`, a stateless `Objective`, a `ViewSpec`
or a simple scalar `Schedule`. Some mechanisms consume predictions, maintain
training-time state and determine what another objective sees.

**xty2 implication:** do not add a generic `Policy`, `Mediator` or `Hook`
abstraction in advance. But when FreeMatch, SoftMatch, ReMixMatch, CoMatch or
similar recipes are implemented, watch specifically for repeated stateful
prediction-mediation semantics. If two real recipes require the same lifecycle,
state ownership and input/output contract, that is concrete evidence for a new
abstraction under the two-consumer rule.

### 1.5 State ownership and checkpointing leak through in complex methods

The limits of SemiLearn's hook abstraction become visible in FreeMatch and
SoftMatch. Algorithm-specific hook state has to be manually added to
`get_save_dict()` and manually restored in `load_model()`.

Sources:

- https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/freematch/freematch.py
- https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/softmatch/softmatch.py

FreeMatch's extra entropy/fairness loss is also defined locally in the algorithm
module rather than as an independently reusable objective. The stateful masking
hook writes state back onto the parent algorithm object because the local loss
needs it.

This is a useful warning for xty2: if a future reusable stateful mechanism exists,
its lifecycle should probably include generic checkpoint/provenance semantics.
But that requirement should be established by multiple concrete consumers, not
pre-built.

### 1.6 CoMatch shows where imperative algorithm classes start to dominate

SemiLearn's CoMatch uses reusable distribution-alignment and threshold hooks, but
most of the method remains bespoke:

- a network wrapper adds projection heads;
- a local contrastive loss is defined in the algorithm module;
- feature and probability memory banks live directly on the algorithm object;
- the algorithm manually updates those banks;
- representation similarity smooths pseudo-label probabilities;
- pseudo-label probabilities construct the graph used by the contrastive loss;
- checkpoint save/load is manually extended for the banks and hook state.

Source:
https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/comatch/comatch.py

Conceptually:

```text
                        DistAlignHook
                             ↓
weak prediction → smoothed prediction → MaskingHook
       ↑                  │
       │                  ↓
 memory bank ← representations → pseudo-label graph
       │                           ↓
       └────────────────→ contrastive objective
```

SemiLearn can implement this faithfully, but the relationship is not represented
structurally. It exists because the statements inside `CoMatch.train_step()`
happen in that order.

**xty2 implication:** CoMatch remains one of the best post-P12 tests of whether
semantic quantities can shape one another while the compiled execution plan
remains understandable. It should not be reduced to "three losses added
together" if the paper's actual dependency structure is richer than that.

### 1.7 ReMixMatch is an extreme orchestration case

SemiLearn's ReMixMatch implementation combines, inside one imperative
`train_step()`:

- weak-view inference;
- distribution alignment;
- sharpening;
- multiple strong views;
- MixUp or manifold MixUp;
- BatchNorm freeze/unfreeze handling;
- mixed labeled and unlabeled losses;
- an additional strong-view loss;
- unsupervised warmup;
- an optional rotation head and rotation objective.

The model is wrapped to add the rotation classifier, while the algorithm class
owns the loss composition and control flow.

Source:
https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/remixmatch/remixmatch.py

This is exactly why ReMixMatch is a useful xty2 stress test. If xty2's recipe
turns into an imperative mini-program to reproduce it, the architecture has not
actually achieved the intended decomposition.

### 1.8 The data layer is explicitly algorithm-aware

The strongest contrast with xty2 is in SemiLearn's dataset layer.

`BasicDataset.__getitem__()` switches on the algorithm name to determine which
views to return. Examples include:

- Pi-model / Mean Teacher / MixMatch: two weak-style views;
- SequenceMatch: weak, medium and strong views;
- CoMatch: weak plus two strong views;
- ReMixMatch: weak, two strong views, a rotated strong view and a rotation
  target.

Source:
https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/datasets/cv_datasets/datasetbase.py

The effective chain is:

```text
algorithm name
    ↓
dataset decides which views exist
    ↓
train_step signature declares which batch keys it expects
    ↓
AlgorithmBase introspects that signature
    ↓
matching fields are moved to device and passed in
```

The signature introspection is pragmatic and keeps the trainer generic, but it
means view requirements are implicit in Python signatures and dataset branches.

**xty2 implication:** this is strong evidence for keeping views as first-class,
algorithm-independent declarations. `ViewSpec`/`Realisation` should prevent the
data source from knowing which named recipe is being run.

### 1.9 The registry remains at the named-algorithm level

SemiLearn's main registry is `ALGORITHMS`, populated with complete classes such
as FixMatch, CoMatch, Mean Teacher, ReMixMatch, SimMatch, UDA, SoftMatch and
FreeMatch.

Source:
https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/core/utils/registry.py

Its ontology is therefore approximately:

```text
algorithm
    has model
    has train_step
    uses reusable hooks
```

xty2 deliberately aims for a deeper decomposition:

```text
recipe
    assembles components
    assembles objectives
    requests views/realisations
    mixes objectives
    sequences stages
```

SemiLearn therefore should not be read as a model for moving xty2 back toward an
algorithm registry. It is better read as evidence that **moving reuse below the
algorithm class is worthwhile**, and as a catalogue of repeated mechanisms that
may eventually deserve independently typed xty2 concepts.

### 1.10 What SemiLearn gets right and should influence reviews

The following ideas have strong empirical support from SemiLearn's implementation:

1. **Lifecycle concerns belong outside methods.** Parameter updates, EMA,
   checkpointing, evaluation and logging should remain generic infrastructure.
2. **Pseudo-label generation is independently reusable.** It should not need to
   be reimplemented by every pseudo-labeling recipe.
3. **Eligibility/weighting is independently reusable.** Fixed thresholds,
   adaptive thresholds and continuous confidence weights are conceptually
   separable from the loss that consumes them.
4. **Distribution alignment is independently reusable and often stateful.**
5. **Some SSL mechanisms maintain streaming state.** Moving class distributions,
   thresholds, queues and memory banks are not edge cases.
6. **Complex methods expose state-lifecycle requirements.** If xty2 eventually
   abstracts such mechanisms, checkpointing/provenance should be part of the
   contract rather than bespoke recipe code.

These observations are evidence, not pre-approval for framework changes.

### 1.11 What xty2 should deliberately avoid copying

| SemiLearn pattern | xty2 direction |
|---|---|
| algorithm-specific `train_step()` | independent objectives + compiled execution plan |
| algorithm-name branches in dataset | named algorithm-independent views |
| ad-hoc model wrappers for auxiliary heads | components producing semantic ports |
| losses local to algorithm modules | independently testable objectives where semantics recur |
| hooks invoked dynamically by string | statically resolvable dependencies where possible |
| mutable hook ↔ algorithm cross-state | explicit ownership and lifecycle for state |
| algorithm manually extends save/load | generic artifact/state persistence if justified |
| execution semantics live in Python statement order | printable compiled execution plan |
| named-algorithm registry is the primary abstraction | thin recipes over reusable pieces |
| framework-wide method conventions | paper-governed choices explicit in the recipe/card |

### 1.12 Fidelity lesson

The current SemiLearn `main` snapshot also illustrates why xty2's fidelity tiers
are worth keeping separate from "the code runs".

In `meanteacher.py`, the code performs a forward pass for `x_ulb_s`, but the
following assignments read `outs_x_ulb_w` rather than `outs_x_ulb_s`:

```python
outs_x_ulb_s = self.model(x_ulb_s)
logits_x_ulb_s = outs_x_ulb_w['logits']
feats_x_ulb_s = outs_x_ulb_w['feat']
```

Source:
https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/meanteacher/meanteacher.py

This appears to be a straightforward implementation error: tensor shapes remain
valid, a scalar loss exists, training can proceed, yet the intended second-view
signal is not the one being consumed.

The lesson is not that SemiLearn is unreliable. Research code inevitably
contains defects. The architectural lesson for xty2 is that **shape-correct
execution is weak evidence**. Directional Tier 1 invariants and Tier 2 published
reproduction targets are specifically valuable for catching mistakes that type,
shape and "loss decreases" checks cannot.

---

## 2. Questions to revisit after P12

The SemiLearn study gives several concrete questions to ask when implementing
post-P12 methods:

1. Do FreeMatch and SoftMatch require the same kind of stateful prediction
   weighting/threshold mechanism?
2. Does distribution alignment need the same lifecycle/state contract as those
   threshold mechanisms, or is it statistically distinct enough to remain its
   own objective-local helper?
3. Do SimMatch, CoMatch and Temporal Ensembling establish a shared lifecycle for
   persistent in-training memory/state?
4. Can CoMatch's prediction ↔ representation interaction be expressed with
   ordinary semantic quantities and objectives, or does it reveal a missing
   mediator concept?
5. Can ReMixMatch remain a declarative recipe once synthetic-row MixUp semantics
   are handled honestly?
6. Does any reusable stateful mechanism need generic checkpoint/provenance
   support, and does a second real recipe require the same thing?
7. Are lifecycle hooks useful internally for infrastructure without becoming a
   public recipe abstraction?

Record answers here as evidence accumulates. Promote an answer into
`DESIGN.md` only when it becomes a binding architectural decision.
