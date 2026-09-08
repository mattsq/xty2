# VICReg review and Tier 1 evidence

Reviewed Claude's implementation at `ffa0b247278dfbf63b5334ccc87529f227d3edb4`.
This follow-up fixes covariance arithmetic and completes card §6.3's smoke
packet. It does not register or claim the full Tier 2 study.

## Findings

The three reductions, symmetric gradients and expander topology agree with the
[pinned author source](https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/main_vicreg.py).
Tabular departures are explicit in card §5. Claude's mixture-posterior correction
answers the original review comment. No additional recipe mechanic was needed.

One numerical defect was fixed: subtracting diagonal energy from total energy
can lose the off-diagonal penalty in float32. The regression fixture returns
zero under the old implementation instead of approximately 177,777,778.
Selecting off-diagonal entries before summing follows the author implementation;
the new test compares value and gradient with a float64 oracle. The expander
commentary also incorrectly said its shared output bias affects invariance.
It cancels from the branch difference; the comment is corrected.

## Protocol and results

All four arms ran on bases 191, 293 and 397 with the explicit 200/300-step Tier 1
overrides, 1,024 train rows, 40 observed treatments and 2,048 held-out rows.
CPU float32, PyTorch 2.14.0+cpu, one thread. Widths, optimiser, weights, ramp and
batch size are unchanged. Embeddings use the terminal pretraining checkpoint,
16 held-out batches with two views each and frozen training BN buffers.
NLL uses training-standardised outcomes. Exact values are in the
[JSON result](2026-09-08-vicreg-smoke.json).

| Base | Arm | Spread | Redundancy | Treatment NLL | Outcome NLL |
|---|---|---|---|---|---|
| 191 | full | 0.269005 | 41.737839 | 0.551175 | 1.155537 |
| 191 | no_covariance | 0.320478 | 502.870405 | 0.615565 | 1.141807 |
| 191 | no_pretrain | n/a | n/a | 0.550669 | 1.197259 |
| 191 | no_variance | 0.010020 | 130.518936 | 0.451223 | 1.184438 |
| 293 | full | 0.264449 | 36.859358 | 1.318516 | 1.099566 |
| 293 | no_covariance | 0.448880 | 508.394004 | 1.160496 | 1.107891 |
| 293 | no_pretrain | n/a | n/a | 1.300683 | 1.121098 |
| 293 | no_variance | 0.010002 | 136.056773 | 1.019401 | 1.133219 |
| 397 | full | 0.273517 | 40.134595 | 1.373961 | 1.123107 |
| 397 | no_covariance | 0.482965 | 509.954574 | 1.102811 | 1.168649 |
| 397 | no_pretrain | n/a | n/a | 1.104593 | 1.276282 |
| 397 | no_variance | 0.010016 | 111.948665 | 0.585698 | 1.301041 |

All three smoke tests pass. They check actual sampled row IDs and corruption
outputs across arms, two cached draws per step, training-only donors/scaling,
identical initial tensors/heads, fresh empty optimiser states, checkpoint
parameter transfer, changed encoder parameters, head isolation in pretraining,
no expander execution/update in fine-tuning, frozen evaluation buffers, finite
losses/gradient norms and valid class probabilities.

Variance removal nearly collapses the embeddings on all seeds. Covariance
removal increases redundancy on all seeds. Full-arm outcome NLL is lower than
no-pretraining on all three, but treatment NLL is worse on two. No directional
smoke gate was added. Full-arm spread is below the **Tier 2** 0.5 target at
this shorter budget. These observations do not assess the unrun full-budget
study. The next packet is the committed ten-seed Tier 2 run, all §6.4 diagnostics,
and a fidelity audit if its targets fail. No thresholds were changed.

## Mutation evidence

Validation: 1,508 invariant tests passed; the installed-package version check
was deselected after it failed because this checkout was not installed.
An editable installation could not finish because its build dependency download
was blocked. The three VICReg smoke tests passed in 93 seconds; Ruff lint and
format, strict mypy (204 source files), and `git diff --check` passed. The
other recipes' smoke tests were not rerun for this isolated change.

Each mutation was injected separately in a temporary checkout. Each named test
failed, then the mutation was reverted.

| Mutation | Test detecting it |
|---|---|
| Covariance denominator n-1 → n | `test_both_statistics_use_the_sample_correction` |
| MSE dimension sum instead of mean | `test_the_invariance_term_divides_by_the_embedding_dimension` |
| Remove variance half factor | `test_the_variance_term_averages_the_branches_and_the_covariance_sums_them` |
| Detach second branch | `test_each_terms_input_gradients_are_the_author_forwards` |
| Include covariance diagonal | `test_the_covariance_excludes_its_own_diagonal` |
| Redraw each view three times with different keys | `test_paired_mechanism_study[191]`: 1,200 draws vs 400 |
| L2-normalise output | `test_the_expander_output_is_unconstrained` |
| Add expander to fine-tuning | `test_the_expander_is_discarded_by_the_fitting_stage`: compile rejection |
| Restore diagonal-energy subtraction | `test_small_off_diagonal_energy_survives_large_diagonal_energy` |
