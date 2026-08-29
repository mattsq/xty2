# Documentation map

Use this page as the entry point. Most changes require this page plus one card,
one design section, or one test file. Reading the whole suite first is usually
wasteful and makes historical rationale look like current scope.

## Task routing

| Task | Read first | Then inspect |
|---|---|---|
| Fix or extend an existing recipe | `recipes/<name>.md` §2–§5 | `xty2/recipes/<name>.py` and matching Tier 0/Tier 1 tests |
| Investigate a recipe result | the card §4–§7 | benchmark runner, result artifact, source paper/code |
| Add a recipe | `recipes/_TEMPLATE.md`, then `FIDELITY.md` §1–§2 | stop after drafting the card for review |
| Change a core contract | the relevant `DESIGN.md` section | owning module and invariant tests |
| Change loading or execution | `DESIGN.md` §7–§8 | `proposals/loader.md` only for the accepted data-boundary decision |
| Plan research | `BACKLOG.md` | `PRIOR_ART.md` for comparative notes |
| Understand project history | `PLAN.md` | Git and PR history |

Do not read `PLAN.md`, `BACKLOG.md`, or `PRIOR_ART.md` to implement an already
specified recipe unless its card names a specific dependency there.

## Authority

Intended fidelity comes from the method source and reviewed specification:

1. The cited paper version and pinned reference implementation define the
   published method.
2. The reviewed recipe card defines xty2's method-specific mapping, declared
   departures, and evidence contract.
3. `DESIGN.md` and `FIDELITY.md` define repository-wide contracts;
   `proposals/loader.md` records one accepted boundary decision.

Code and tests describe observed repository behaviour. They are evidence of
agreement with the card, not higher authority over the source method or reviewed
specification. A disagreement triggers an audit: fix the implementation or amend
and re-review the card and normative documents. `PLAN.md`, `BACKLOG.md`, and
`PRIOR_ART.md` remain historical or exploratory.

## Document roles

| Document | Role | Normative? |
|---|---|---|
| [`DESIGN.md`](DESIGN.md) | architecture and compiler contracts | yes |
| [`FIDELITY.md`](FIDELITY.md) | cards, test tiers, and deviation debt | yes |
| [`recipes/`](RECIPES.md) | per-method source and reproduction contracts | yes, per recipe |
| [`proposals/loader.md`](proposals/loader.md) | accepted data/loading decision | only for that boundary |
| [`PLAN.md`](PLAN.md) | completed P0–P12 build history | no |
| [`BACKLOG.md`](BACKLOG.md) | candidate research and framework stress tests | no |
| [`PRIOR_ART.md`](PRIOR_ART.md) | comparative framework notes | no |

`CLAUDE.md` is the short operating policy for coding agents. It routes here
rather than duplicating these documents.
