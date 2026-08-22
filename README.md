# xty2

XTYLearner 2 — semi-supervised learning in tabular settings using a causal
framework, building on Lucas Beyer's approaches in MOAM etc.

## Design documents

| Document | What it covers |
|---|---|
| [`SEED.md`](SEED.md) | The original architectural proposal this design derives from |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Architecture: ports, components, objectives, views, loss mixer, program, recipes, compiler |
| [`docs/FIDELITY.md`](docs/FIDELITY.md) | How reimplementations are kept honest: spec cards and three test tiers |
| [`docs/PLAN.md`](docs/PLAN.md) | Twelve work packets, two review gates, risk register |
| [`docs/recipes/_TEMPLATE.md`](docs/recipes/_TEMPLATE.md) | The spec card every recipe must have before it is implemented |

## The one-paragraph version

XTYLearner registered ~40 monolithic model classes, so nothing composed and a bad
result could not be attributed to a cause. xty2 splits five collapsed questions —
what quantities are represented, how they are parameterised, which losses train
them, which data views and row subsets each loss uses, and in what order — into
separate, independently testable layers. Named methods become **recipes** that
compose registered components, objectives and views.

Correctness is not left to review. Every recipe has a **spec card** written and
reviewed *before* any code: paper provenance, transcribed equations, a mechanics
checklist covering the details that get silently dropped, declared deviations,
declared unknowns, and a pre-declared reproduction target. Three test tiers back
it — invariants and a synthetic smoke fit on every PR, published-number
reproduction nightly. A recipe is done when it reproduces its target or explains
in writing why it does not.

**Scope of v1:** categorical treatment with small K, exact marginalisation over
missing treatments, a linear list of training stages, Python-first recipes, and
five ported models. See `docs/DESIGN.md` §11 for what is deliberately not built.
