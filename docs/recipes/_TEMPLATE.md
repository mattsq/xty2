# Recipe spec card: <recipe_name>

> Copy this file to `docs/recipes/<recipe_name>.md`. Fill it in **before**
> writing any code, and get it reviewed. See `docs/FIDELITY.md` §1.
> Delete the `>` guidance lines as you go; keep every heading.

**Status:** `draft`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | |
| Authors, year | |
| DOI / arXiv | |
| Version used | <!-- v3, 2018-05-14 — papers change between versions --> |
| Reference implementation | <!-- URL @ commit sha, or "none available" --> |
| Reference impl. runnable? | <!-- yes / no / not attempted --> |

## 2. Estimand and claim

> What quantity does this method estimate? What does the paper claim about it,
> and on what evidence? What does it explicitly *not* claim?

- **Estimand:**
- **Claim:**
- **Not claimed:**

## 3. Equations and mapping

> Transcribe the losses in the paper's own notation, with its equation numbers.
> Do not paraphrase into our notation here — that translation is what §3.2 is
> for, and doing it inline is where terms get dropped.

### 3.1 As published

> Eq. (n): ...

### 3.2 Mapping to xty2

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| | | | |

> Every component, objective and view the implementation creates must appear in
> this table. If it is not here, it is out of scope.

## 4. Mechanics checklist

> Every key: a value with a paper citation (`§4.2` / `Eq. 7` / `ref impl:
> train.py:88`), or `n/a`, or `unspecified` (and then it belongs in §7 too).
> CI asserts the recipe explicitly sets every key named here.

```yaml
gradients:
  stop_gradients:
  detached_targets:
  gradient_clipping:
  marginal_nll_grad_path:

teacher:
  ema_decay:
  ema_applies_to_buffers:
  teacher_in_train_mode:
  teacher_requires_grad: false

losses:
  reduction:             # mean | sum | population
  eligible_rows:
  weights:
  schedules:
  temperature:
  sharpening:
  confidence_threshold:

optimisation:
  optimiser:
  lr:
  lr_schedule:
  weight_decay:
  batch_size:
  labelled_unlabelled_ratio:
  total_steps_or_epochs:

architecture:
  widths_depths:
  activation:
  normalisation:
  dropout:
  initialisation:
  output_parameterisation:

data:
  standardisation:
  outcome_scaling:
  treatment_encoding:
  split_protocol:
  missingness_mechanism:
```

## 5. Deviations from the paper

> "None." is a valid entry, but it must be written. An empty table is treated as
> an unanswered section, not as an assertion of fidelity.

| # | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|
| | | | |

## 6. Reproduction target

```yaml
reproduction:
  dataset:
  variant:
  split:
  metric:
  published:
  published_source:      # paper, table/figure number
  tolerance:
  seeds:
  report: mean_and_stderr
```

### 6.1 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| | | | | |

## 7. Unknowns

> Things the paper does not specify, and what we chose. This section is
> mandatory and is rarely empty. Each entry: what is missing, what we chose, and
> how we chose it (reference implementation / convention / guess).

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| | | |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | | |
| Plan diffed against §3.2 and §4 | | |
