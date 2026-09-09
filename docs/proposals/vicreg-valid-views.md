# Proposed VICReg valid-view contract amendment

**State: approved by the repository owner on 2026-09-09; implementation and prospective confirmation authorised.**

The [completed paired experiment](../experiments/2026-09-08-vicreg-views.md)
passes all four unchanged targets under a predeclared, nontrivial,
target-preserving view policy. Marginal corruption still fails spread.
`CLAUDE.md` rule 1 and `FIDELITY.md` section 4 require an amended card to be
reviewed before changing the recipe's normative mechanics. The current card
remains `deviating` until the authorised fresh confirmation is complete.
The owner approved this exact scope; no further mechanics review is pending.

## Recommended claim

The reproduced object should be VICReg supplied with explicitly declared,
domain-appropriate views. The claim should not be that a fixed 0.6 marginal
corruption is a universally suitable tabular VICReg transformation. The original
paper's transformations are part of its method, and the tested replacement was
a consequential adaptation rather than a neutral framework constant.

Proposed replacement for card section 2's mechanism claim:

> On the fixed XTY fixture, with the target-preserving transformations specified
> in section 6, variance regularisation prevents collapse, covariance
> regularisation reduces redundant dimensions, and transferring the encoder
> does not materially harm factual outcome fit. This is a fixture-specific
> mechanism reproduction, not ImageNet reproduction, a generally applicable
> augmentation selector, or evidence of treatment-effect identification.

## Exact amendments to review

1. **Sections 3-4: make the two view transformations explicit method inputs.**
   Keep the two cached branches, shared encoder/expander, all three loss
   reductions and coefficients, 1000/3000 steps, batch 128, Adam and downstream
   objectives. Assembly should accept caller-supplied transformations without
   conditionals or DGP logic in `xty2/recipes/`. The plan must render both.
2. **Section 5, deviation 3: withdraw marginal corruption as the canonical
   benchmark view, retaining its full failure record.** Add a judgement row
   for the analytic DGP symmetries, including their privileged knowledge of
   the fixture. Do not describe this as restoring the paper's image views.
3. **Section 6: use the exact predeclared symmetry from the experiment.**
   Preserve sum(x0..x3), 0.5*x0-0.3*x1, x2 and x4 squared through the reflection
   and sign flip; redraw only x5. Preserve all data/mask/initialisation and
   stage pairing, held-out evaluation and the four original numeric bounds.
   Target preservation, non-identity, independent draws and train-only scaling
   become executable benchmark/invariant requirements.
4. **Separate prospective confirmation from the completed diagnostic.**
   After the amended card is reviewed and implemented, run ten fresh bases
   `290000+100*i`, i=0..9, with all original offsets and four arms. This is a
   declared new seed stream, not a transcription mistake or a search over
   favourable seeds. Run from a committed implementation. Register the revised
   benchmark and record its complete result/ledger together. Advance to
   `reproduced` only if it meets every unchanged one-SE bound.
5. **Keep the old evidence legible.** Retain the original `a9fec5cc9623` ledger
   row, original corruption settings, its `deviating` outcome, the paired view
   experiment and this review decision. The final status must name which
   version of the view contract it validates.

## Why this is a substantive choice

The experiment establishes that view design matters; it does not turn oracle
DGP symmetries into a ready-made augmentation for arbitrary tabular problems.
Adopting this amendment means the card demonstrates VICReg's mechanism with
valid, explicitly supplied views. If the intended claim instead requires the
current all-feature marginal-corruption recipe to pass, this amendment is not
the answer: that recipe remains a failed adaptation, requiring a separately
predeclared investigation of its view/weight choices. Lowering 0.5 just to
clear the observed 0.27 is not proposed.
