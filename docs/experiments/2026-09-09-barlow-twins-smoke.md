# Barlow Twins Tier 1 evidence

Ran card §6.3's Tier 1 packet on `84a10d8`, the head at which
`docs/recipes/barlow_twins.md` was `implemented` with no fit behind it. This
adds `tests/smoke/test_barlow_twins.py` and the numbers below. It does **not**
register the Tier 2 benchmark, fill §6.1's ledger row, or assess §6.4's
bounds: those are ten-seed statements at the §4 budget, and this is three
seeds at a fifth of the pretraining steps.

## Protocol

All four §6.2 arms — `full`, `diagonal_only` (`lambda` set to exactly zero
with the objective still declared), `no_pretrain`, and contextual `vicreg` —
ran on bases 419, 523 and 631 with card §6.3's explicit 200/300-step
overrides, the unchanged 1,000-step marginal ramp, 1,024 training rows, 40
observed treatments and 2,048 held-out rows. Widths, optimiser, weights and
batch size are §4's.
CPU float32, PyTorch 2.14.0, Python 3.11.15, one thread.

Pairing is checked rather than assumed: the arms share initial encoder and head
tensors, actual sampled row IDs per stage, and the cached view draws. The
`no_pretrain` arm shifts its execution seed by `STREAM_STRIDE` so its downstream
stream is stage index 1's. The contextual arm receives the same encoder and head
tensors by copy, because `BarlowTwinsProjector` and `VICRegExpander` consume
construction RNG differently.

Embeddings come from the terminal pretraining checkpoint, evaluated before
fine-tuning with the hidden BatchNorm in eval mode on its frozen training
buffers, over §6.2's 16 disjoint held-out batches of 128 rows and two
independent oracle-symmetry draws each. §3.1's stateless correlation is
imported from `xty2.objectives.barlow_twins`, not transcribed again. Outcome
and treatment NLL are the fine-tuned graph's, on clean inputs and the
training-standardised outcome scale. Exact values are in the
[JSON result](2026-09-09-barlow-twins-smoke.json).

## Results

`a` is §6.4's mean diagonal, `r` its mean squared off-diagonal per ordered
pair, `f` its active-coordinate fraction, `var` the mean raw population
variance of branch A's embedding, and `top eig` the covariance's top
eigenvalue share.

| Base | Arm | a | r | f | var | top eig | Treatment NLL | Outcome NLL |
|---|---|---|---|---|---|---|---|---|
| 419 | full | 0.972560 | 0.021807 | 1.000000 | 0.048047 | 0.064417 | 1.962519 | 1.130789 |
| 419 | diagonal_only | 0.999086 | 0.643086 | 1.000000 | 0.952294 | 0.836494 | 2.578963 | 1.126193 |
| 419 | no_pretrain | n/a | n/a | n/a | n/a | n/a | 2.079429 | 1.172241 |
| 419 | vicreg | 0.997762 | 0.028966 | 1.000000 | 0.392680 | 0.073578 | 2.454518 | 1.125353 |
| 523 | full | 0.971424 | 0.022434 | 1.000000 | 0.043807 | 0.065803 | 0.580382 | 1.117731 |
| 523 | diagonal_only | 0.998744 | 0.435488 | 1.000000 | 0.741601 | 0.643506 | 0.553907 | 1.114511 |
| 523 | no_pretrain | n/a | n/a | n/a | n/a | n/a | 0.571038 | 1.127378 |
| 523 | vicreg | 0.998437 | 0.031839 | 1.000000 | 0.388498 | 0.086125 | 0.760651 | 1.111667 |
| 631 | full | 0.978812 | 0.021041 | 1.000000 | 0.041432 | 0.059125 | 0.810657 | 1.180753 |
| 631 | diagonal_only | 0.999496 | 0.519801 | 1.000000 | 0.947455 | 0.682590 | 0.761846 | 1.172853 |
| 631 | no_pretrain | n/a | n/a | n/a | n/a | n/a | 0.760498 | 1.170020 |
| 631 | vicreg | 0.997980 | 0.030384 | 1.000000 | 0.390823 | 0.077611 | 0.796247 | 1.145027 |

Paired `r_diagonal_only - r_full` is 0.621279, 0.413054 and 0.498759; paired
`NLL_full - NLL_no_pretrain` is -0.041452, -0.009648 and +0.010733. Both are
reported and neither is asserted. Three seeds at 200 pretraining steps cannot
carry §6.4's ten-seed standard errors, and the third bound in particular is a
full-budget claim about a quantity that is still moving at this one.

The mechanism is connected in the direction the paper describes: at three
seeds the off-diagonal term costs under three points of diagonal alignment and
buys a nineteen- to thirtyfold reduction in mean squared off-diagonal
correlation, and it does so without deactivating coordinates — `f` is exactly
1.0 in every arm and seed, so the redundancy improvement is not the collapse
§6.4's first two bounds exist to exclude. Raw variance and eigenvalue
concentration say the same thing from the other side: the diagonal-only arm
keeps roughly twenty times the raw variance while concentrating 64-84% of its
covariance energy in one direction, where the full arm holds 6%. The contextual
VICReg arm lands within half an order of magnitude of the full arm's redundancy
by a different route, at roughly nine times its raw variance. No claim of
superiority either way follows from three seeds, and §6.2 does not ask for one.

## Mutation evidence

Each mutation was applied on its own, the study was rerun on base 419, the
named assertion failed, and the mutation was reverted. Two are Tier 0's
mechanics seen from the study's side; the other six perturb the §6.2 protocol
and the arm construction this packet is responsible for.

| Mutation | Assertion that failed |
|---|---|
| Drop `initialise_from="pretrain"` from `joint_fit` | pretraining checkpoint equals the fitting stage's starting tensors |
| Pretrain the projector only, freezing the encoder | pretraining moved at least one encoder parameter |
| A fresh draw per objective (`draws=2`, off-diagonal at `draw=1`) | one cached pair of draws per step: 800 view applications against 400 |
| `no_pretrain` keeps stage index 0's execution seed | the arms sample the same `joint_fit` row IDs |
| The contextual arm keeps its own encoder and head draws | every arm starts from the same transferred tensors |
| One generator seed for both held-out branches | the two branches of an evaluation batch are independent draws |
| A no-op weight replacement in the ablated arm | `full` and `diagonal_only` pretrain different encoders |
| A misspelt objective name in the ablated arm | exactly one declared term is zeroed |

## Validation

`pytest tests/invariants tests/smoke` passes whole, 1,694 tests in 55 minutes,
and `ruff check`, `ruff format --check` and `mypy --strict` (213 files) pass on
the committed tree. The three Barlow Twins Tier 1 fits take 127 seconds of
that.

The card moves to `smoke-passing`, and Tier 0's status assertion moves with it:
`test_the_card_records_the_implementation_it_now_has` now pins the new status,
the new §8 review row and the existence of the Tier 1 module. Both new halves
were seen to fail — once with the review row deleted from the card, once with
the smoke module moved aside. §6.1's ledger row is untouched and no `RECIPES`
entry or benchmark module is added: `CLAUDE.md` requires those to land with a
Tier 2 result rather than ahead of one.
