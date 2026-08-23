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

---

## 3. Broader framework prior art

SemiLearn is not the only attempt to move reuse below a named method. Other
frameworks attack different parts of the same decomposition problem. None is a
direct template for xty2; the useful question is which boundary each project
chooses and where it eventually falls back to imperative method code.

### 3.1 VISSL

Repository: https://github.com/facebookresearch/vissl

VISSL was FAIR's modular framework for self-supervised learning. Its task
configuration independently selected model trunks, heads, losses, transforms,
optimizers/schedulers and lifecycle hooks. The explicit design goal was that
components developed for one SSL task could be reused in another.

Architecturally, this is much closer to xty2 than SemiLearn because the method is
assembled from lower-level pieces rather than represented only by an algorithm
subclass. VISSL also exposes fine-grained lifecycle hook points around forward,
loss, backward and parameter update.

**xty2 lesson:** study VISSL as prior art for config/recipe composition and
lifecycle separation, but retain xty2's stronger semantic contracts. VISSL's
interfaces are mostly software/architectural contracts rather than typed
statistical quantities. The project was archived in 2024, so treat it as design
history rather than a dependency candidate.

### 3.2 OpenMixup / OpenMMLab-style composition

Repository: https://github.com/Westlake-AI/openmixup

OpenMixup supports supervised, self-supervised and semi-supervised learning using
OpenMMLab registries and configuration. Configs separately name model type,
backbone, neck, head, loss, augmentation and runtime/schedule pieces.

This is strong precedent for making a named method mostly configuration over
registered components. It also shows the limits of that pattern: the reusable
vocabulary is primarily architectural (`backbone`, `neck`, `head`) rather than
statistical (`p(t|x)`, `p(y|x,t)`, candidate-treatment distributions, row
eligibility).

**xty2 lesson:** keep the good part — thin recipes and resolved registries —
without collapsing ports into generic neural-network module roles.

### 3.3 LightlySSL

Repository: https://github.com/lightly-ai/lightly

LightlySSL is a useful "box of Lego" precedent. It exposes transforms, view/data
utilities, projection/prediction heads, SSL losses, momentum-model helpers and
memory-bank machinery as independent PyTorch pieces. Full methods are then wired
in ordinary training code.

A particularly relevant pattern is reusable memory-bank state. The same loss or
neighbour mechanism can be paired with or without a memory bank rather than
requiring a completely different monolithic algorithm class.

**xty2 lesson:** this supports keeping objectives and state mechanisms
independent where their contracts genuinely recur. Lightly does not solve the
execution-plan problem: composition ultimately lives in imperative
`training_step` code, which is exactly the boundary xty2 is trying to push past.

### 3.4 solo-learn and Dassl

Repositories:

- https://github.com/vturrisi/solo-learn
- https://github.com/KaiyangZhou/Dassl.pytorch

These are useful comparison cases because they converge on a pattern similar to
SemiLearn: substantial shared infrastructure plus a method-specific subclass.
solo-learn separates losses, augmentations and momentum utilities but full
methods still inherit from `BaseMethod`/`BaseMomentumMethod` and override the
training step. Dassl similarly provides registries and common trainers, while
SSL/domain methods implement their own `forward_backward` logic.

**xty2 lesson:** several independent framework efforts have found this a useful
engineering equilibrium, but it is also a clear composability ceiling. Shared
trainers remove repetition without making the learning programme itself a
first-class inspectable object.

### 3.5 MosaicML Composer

Repository: https://github.com/mosaicml/composer

Composer is not an SSL framework, but it is important execution prior art. Its
core idea is that training interventions should be independently composable
algorithms attached to explicit events. An algorithm effectively answers:

```text
match(event, state) -> bool
apply(event, state)
```

with events around dataloading, forward, loss, backward and parameter updates.
Multiple interventions can therefore modify batches, models, losses or training
dynamics without being hard-coded into one trainer.

This is the strongest prior art found so far for xty2's unresolved post-P12
"stateful mediator/policy" question. A mechanism such as distribution alignment,
adaptive thresholding or an in-training state update can be understood as a
state transition at a declared point in execution.

**xty2 lesson:** do not copy a generic event bus into the public recipe API.
Composer gains flexibility by weakening semantic typing and allowing broad state
mutation. But its event/state model is worth studying if two xty2 recipes force
a reusable mechanism that cannot honestly be represented as a component,
objective, view, schedule or stage.

### 3.6 PyTorch Metric Learning

Repository: https://github.com/KevinMusgrave/pytorch-metric-learning

PyTorch Metric Learning is unusually relevant because its decomposition is
semantic rather than merely architectural. It separates concepts such as:

```text
sampler -> miner -> distance -> loss -> reducer
```

and provides explicit conversion/compatibility semantics between the structures
these pieces exchange. A miner and a loss need not have been designed as one
named algorithm to compose correctly.

This is close to the philosophical target for xty2: identify meaningful
intermediate objects, give them contracts, and let independent mechanisms
consume them.

A possible future analogy, if real recipes justify it, is:

```text
TreatmentDistribution
    -> alignment/calibration
    -> eligibility/weighting
    -> target construction
    -> objective
    -> reduction
```

**xty2 lesson:** this is perhaps the strongest precedent for semantic interface
design. It argues for resisting generic `Tensor -> Tensor` helpers when a stable
statistical object can be named and checked instead.

### 3.7 WRENCH and weak-supervision frameworks

Repository: https://github.com/JieyuZ2/wrench

WRENCH decomposes weak-supervision systems into supervision/label models,
downstream end models and joint models. It is useful prior art if xty2 expands
beyond exact-vs-missing treatment labels into labeling functions, partial labels
or other rich supervision.

**xty2 lesson:** supervision itself may eventually deserve typed structure rather
than being collapsed into a target tensor. Do not import that abstraction until
concrete rich-supervision recipes satisfy the two-consumer rule.

### 3.8 Harmony and compositional SSL research

Harmony (TMLR 2025) is scientifically important even though it is not a general
software framework. It deliberately combines weak supervision, discriminative
self-supervision, generative self-supervision and EMA soft targets, training
multiple objectives simultaneously and allowing subsets to be enabled for
ablation.

Reference: https://openreview.net/forum?id=IcOBCufqFO

This is close to the scientific use case xty2 is intended to make cheap:
construct a crowded but inspectable recipe from independently meaningful
signals, then study which combinations help and which interfere.

### 3.9 Composable Interventions and interaction science

The ICLR 2025 work on *Composable Interventions for Language Models* is adjacent
rather than SSL-specific, but it addresses an important downstream question:
once interventions become composable, their order and interaction effects become
scientific objects in their own right. Independently useful methods can
interfere, and composition can be non-commutative.

Reference:
https://proceedings.iclr.cc/paper_files/paper/2025/hash/7f5f9a88c6516469c83d074c6f2976fb-Abstract-Conference.html

**xty2 lesson:** objective-level raw losses, coverage, gradient norms and gradient
cosines are not merely debugging conveniences. They provide the beginnings of an
experimental language for asking what happens when independently validated
mechanisms are combined.

### 3.10 Comparison matrix

| Concern | Strong prior art | Typical boundary |
|---|---|---|
| shared SSL trainer + reusable utilities | SemiLearn, solo-learn, Dassl | method subclass still owns training logic |
| composable architecture/head/loss/view pieces | VISSL, OpenMixup | software contracts more than statistical semantics |
| reusable SSL primitives and memory state | LightlySSL | full method remains imperative |
| event-driven training interventions | Composer | very flexible shared-state mutation |
| semantic intermediate contracts | PyTorch Metric Learning | domain-specific but strongly typed composition |
| weak-supervision pipeline decomposition | WRENCH | label/end-model pipeline rather than XTY programme |
| multi-objective SSL composition | Harmony | research recipe, not general compiler |
| interaction/order effects of composed methods | Composable Interventions | analysis framework in a different domain |

No framework found so far combines all of xty2's intended pieces: semantic
statistical ports, explicit views and row populations, independently reusable
objectives, staged programmes, leakage/provenance checks, a printable compiled
execution plan, and card/reproduction-based fidelity. That does not establish
novelty, but it does identify the specific combination for which existing prior
art is currently thin.

### 3.11 Source-study priorities

If further source-code study is worth the time, prioritise:

1. **PyTorch Metric Learning** — semantic intermediate contracts and conversion
   rules.
2. **Composer** — state/event lifecycle and composition ordering.
3. **VISSL** — task/config decomposition across model, losses, transforms and
   lifecycle hooks.
4. **OpenMixup** — registry/config composition at larger research-framework
   scale.
5. **LightlySSL** — reusable stateful SSL primitives such as memory banks.

As with SemiLearn, record observations here first. Promote them into
`DESIGN.md` only when a real xty2 implementation decision makes them binding.
