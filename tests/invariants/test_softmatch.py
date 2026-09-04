"""Tier 0 — SoftMatch's Gaussian state, weighting arithmetic, and recipe plan."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from xty2.core import (
    CardKeyError,
    CategoricalTreatment,
    FeatureSpec,
    LossError,
    OutcomeSpec,
    Port,
    Realisation,
    Schema,
    State,
    TrainContext,
    compile,
)
from xty2.objectives import (
    ConfidenceGaussian,
    PseudoLabelTreatmentNLL,
    SoftWeightedTreatmentNLL,
    TruncatedGaussianWeighting,
)
from xty2.recipes import softmatch
from xty2.recipes.fixmatch import (
    STRONG_X,
    WEAK_MASK_RATE,
    WEAK_X,
    WEAK_X_LABELLED,
)
from xty2.recipes.flexmatch import STRONG_MASK_RATE
from xty2.recipes.softmatch import SOFTMATCH_TERM, SOFTMATCH_WEIGHTING

from tests.invariants.conftest import (
    BATCH_SIZE,
    NUM_TREATMENTS,
    make_batch,
    make_schema,
)

CARD = Path(__file__).resolve().parents[2] / "docs" / "recipes" / "softmatch.md"
TARGET = Realisation(view="weak_x")
PREDICTION = Realisation(view="strong_x")


def _policy(
    *, decay: float = 0.9, n_sigma: float = 2, alignment: str = "uniform"
) -> TruncatedGaussianWeighting:
    return TruncatedGaussianWeighting(
        decay=decay,
        n_sigma=n_sigma,
        alignment=alignment,  # type: ignore[arg-type]
    )


def _objective(**overrides: object) -> SoftWeightedTreatmentNLL:
    defaults: dict[str, object] = {
        "port": Port.T_GIVEN_X,
        "target": TARGET,
        "prediction": PREDICTION,
        "num_treatments": NUM_TREATMENTS,
        "weighting": _policy(),
        "sharpening": "hard",
        "stop_grad": "target",
        "rows": "all",
    }
    return SoftWeightedTreatmentNLL(**(defaults | overrides))  # type: ignore[arg-type]


def _probabilities() -> torch.Tensor:
    return torch.tensor(
        [
            [0.80, 0.15, 0.05],
            [0.20, 0.70, 0.10],
            [0.10, 0.20, 0.70],
            [0.60, 0.25, 0.15],
            [0.15, 0.55, 0.30],
            [0.20, 0.35, 0.45],
            [0.34, 0.33, 0.33],
        ]
    )


def _state(target: torch.Tensor, prediction: torch.Tensor) -> State:
    return State(
        {
            TARGET: {Port.T_GIVEN_X: CategoricalTreatment(target.log())},
            PREDICTION: {Port.T_GIVEN_X: CategoricalTreatment(prediction.log())},
        }
    )


def _context(gaussian: ConfidenceGaussian, step: int) -> TrainContext:
    return TrainContext(
        global_step=step,
        schema=make_schema(),
        stage="joint_fit",
        objective_states={SOFTMATCH_TERM: gaussian},
    )


def _rows() -> torch.Tensor:
    return torch.arange(BATCH_SIZE)


def _recipe_schema() -> Schema:
    return Schema(
        features=tuple(FeatureSpec(f"x{i}", "continuous") for i in range(6)),
        treatment_cardinality=2,
        outcome=OutcomeSpec(),
    )


def test_the_three_emas_match_equations_six_to_eight_over_three_steps() -> None:
    policy = _policy(decay=0.8)
    state = ConfidenceGaussian(NUM_TREATMENTS, policy)
    expected_mean = torch.tensor(1.0 / NUM_TREATMENTS, dtype=torch.float64)
    expected_variance = torch.tensor(1.0, dtype=torch.float64)
    expected_marginal = torch.full(
        (NUM_TREATMENTS,), 1.0 / NUM_TREATMENTS, dtype=torch.float64
    )

    for step in range(3):
        probs = torch.roll(_probabilities(), shifts=step, dims=0).double()
        confidence = probs.max(dim=-1).values
        expected_mean = 0.8 * expected_mean + 0.2 * confidence.mean()
        expected_variance = 0.8 * expected_variance + 0.2 * confidence.var(
            unbiased=True
        )
        expected_marginal = 0.8 * expected_marginal + 0.2 * probs.mean(dim=0)
        state.observe(step, probs)

        assert state.mean == pytest.approx(float(expected_mean))
        assert state.variance == pytest.approx(float(expected_variance))
        assert torch.allclose(state.marginal, expected_marginal)


def test_the_first_batch_is_folded_in_before_it_is_weighted() -> None:
    state = ConfidenceGaussian(NUM_TREATMENTS, _policy())
    before = (state.mean, state.variance, state.marginal)
    state.observe(0, _probabilities())
    assert state.last_observed_step == 0
    assert state.mean != before[0]
    assert state.variance != before[1]
    assert not torch.equal(state.marginal, before[2])


def test_uniform_alignment_is_identity_for_a_uniform_running_marginal() -> None:
    probs = _probabilities()
    state = ConfidenceGaussian(NUM_TREATMENTS, _policy())
    assert torch.allclose(state.aligned(probs), probs)
    assert torch.allclose(
        state.weights(probs), state.weights(probs, apply_alignment=False)
    )

    no_alignment = ConfidenceGaussian(NUM_TREATMENTS, _policy(alignment="none"))
    no_alignment.observe(0, probs)
    assert torch.equal(no_alignment.aligned(probs), probs)


def test_uniform_alignment_changes_only_weights_not_pseudo_labels() -> None:
    target = _probabilities()
    prediction = torch.flip(target, dims=(1,))
    gaussian = ConfidenceGaussian(NUM_TREATMENTS, _policy())
    term = _objective().compute(
        _state(target, prediction),
        make_batch(),
        _rows(),
        _context(gaussian, 0),
    )
    expected_labels = target.argmax(dim=-1)
    expected = -prediction.log().gather(1, expected_labels[:, None]).squeeze(1)
    expected *= gaussian.weights(target)
    assert float(term.value) == pytest.approx(float(expected.mean()))


def test_pre_ua_weight_profile_uses_the_same_gaussian_moments() -> None:
    probs = torch.tensor(
        [
            [0.80, 0.10, 0.10],
            [0.75, 0.15, 0.10],
            [0.70, 0.20, 0.10],
            [0.65, 0.25, 0.10],
        ]
    )
    state = ConfidenceGaussian(NUM_TREATMENTS, _policy(decay=0.5))
    state.observe(0, probs)
    raw = state.weights(probs, apply_alignment=False)
    confidence = probs.max(dim=-1).values
    delta = torch.clamp(confidence - state.mean, max=0.0)
    expected = torch.exp(-(delta.square() / (2.0 * state.variance / 2.0**2)))
    assert torch.allclose(raw, expected)
    assert not torch.allclose(state.weights(probs), raw)


def test_weights_are_positive_bounded_and_flat_above_the_updated_mean() -> None:
    probs = _probabilities()
    state = ConfidenceGaussian(NUM_TREATMENTS, _policy(alignment="none"))
    state.observe(0, probs)
    weights = state.weights(probs)
    confidence = probs.max(dim=-1).values
    assert bool((weights > 0.0).all())
    assert bool((weights <= 1.0).all())
    assert torch.equal(
        weights[confidence >= state.mean],
        torch.ones_like(weights[confidence >= state.mean]),
    )


def test_near_zero_effective_variance_is_the_constant_gate_limit() -> None:
    target = _probabilities()
    prediction = torch.flip(target, dims=(1,))
    policy = _policy(n_sigma=1e6, alignment="none")
    gaussian = ConfidenceGaussian(NUM_TREATMENTS, policy)
    gaussian.observe(0, target)
    ours = _objective(weighting=policy).compute(
        _state(target, prediction),
        make_batch(),
        _rows(),
        _context(gaussian, 0),
    )
    gate = PseudoLabelTreatmentNLL(
        port=Port.T_GIVEN_X,
        target=TARGET,
        prediction=PREDICTION,
        threshold=gaussian.mean,
        sharpening="hard",
        stop_grad="target",
        rows="all",
    ).compute(
        _state(target, prediction),
        make_batch(),
        _rows(),
        TrainContext(global_step=0, schema=make_schema()),
    )
    assert float(ours.value) == pytest.approx(float(gate.value), abs=1e-7)


def test_large_variance_is_the_ungated_pseudo_label_limit() -> None:
    target = _probabilities()
    prediction = torch.flip(target, dims=(1,))
    policy = _policy(alignment="none")
    gaussian = ConfidenceGaussian(NUM_TREATMENTS, policy)
    gaussian._variance = torch.tensor(1e12, dtype=torch.float64)
    gaussian._last_step = 0
    gaussian._last_rows = BATCH_SIZE
    ours = _objective(weighting=policy).compute(
        _state(target, prediction),
        make_batch(),
        _rows(),
        _context(gaussian, 0),
    )
    expected = -prediction.log().gather(1, target.argmax(dim=-1)[:, None]).mean()
    assert float(ours.value) == pytest.approx(float(expected), rel=1e-6)


def test_the_policy_and_objective_reject_unreviewed_shapes() -> None:
    with pytest.raises(LossError, match="alignment"):
        _policy(alignment="distribution")
    with pytest.raises(LossError, match="positive"):
        _policy(n_sigma=0)
    with pytest.raises(LossError, match="at least two"):
        ConfidenceGaussian(NUM_TREATMENTS, _policy()).observe(0, _probabilities()[:1])
    with pytest.raises(CardKeyError, match=r"losses\.confidence_threshold"):
        SoftWeightedTreatmentNLL(
            port=Port.T_GIVEN_X,
            target=TARGET,
            prediction=PREDICTION,
            num_treatments=NUM_TREATMENTS,
            sharpening="hard",
            stop_grad="target",
        )


def test_the_recipe_plan_is_the_reviewed_four_term_program() -> None:
    recipe = softmatch(_recipe_schema())
    stage = recipe.program[0]
    assert [weighted.objective.name for weighted in stage.objectives] == [
        "observed_outcome_nll",
        "observed_treatment_nll",
        SOFTMATCH_TERM,
        "missing_treatment_marginal_nll",
    ]
    objective = stage.objectives[2].objective
    assert isinstance(objective, SoftWeightedTreatmentNLL)
    assert objective.target == WEAK_X
    assert objective.prediction == STRONG_X
    assert stage.objectives[1].objective.realisation == WEAK_X_LABELLED  # type: ignore[attr-defined]
    assert objective.batch_coupled
    assert objective.detaches == frozenset({(Port.T_GIVEN_X, WEAK_X)})

    plan = compile(recipe).plan
    assert plan.hyperparameters["losses.confidence_threshold"] == (SOFTMATCH_WEIGHTING)
    assert plan.hyperparameters["losses.weights"][f"joint_fit.{SOFTMATCH_TERM}"] == 1.0
    assert [view.name for view in recipe.views] == ["weak_x", "strong_x"]
    assert [transform.p for transform in recipe.views[1].transforms] == [0.1, 0.2]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# The views (card §4's prose, which §4's YAML has no key for)
# ---------------------------------------------------------------------------


def test_the_two_declared_views_are_the_ones_the_compiled_plan_runs() -> None:
    """Card §4's view paragraph asks for exactly this comparison.

    "§4's YAML has no key for a view, so a Tier 0 test must compare this
    paragraph's two transforms against the compiled plan, as
    `tests/invariants/test_freematch.py` does." The weak view is `fixmatch`'s
    at two draws; the strong one is `flexmatch`'s 0.1-then-0.2, which is
    deviation 2 and the one thing §2's first limitation makes load-bearing
    here.
    """
    plan = compile(softmatch(_recipe_schema())).plan
    assert [view.name for view in plan.views] == ["weak_x", "strong_x"]
    weak, strong = plan.views
    assert weak.transforms == (
        f"FeatureMask(p={WEAK_MASK_RATE}, columns=all, value=0.0)",
    )
    assert weak.draws == 2
    assert strong.transforms == (
        f"FeatureMask(p={WEAK_MASK_RATE}, columns=all, value=0.0)",
        f"FeatureMask(p={STRONG_MASK_RATE}, columns=all, value=0.0)",
    )
    assert strong.draws == 1
    assert STRONG_MASK_RATE == 0.2


def test_the_cards_prose_about_the_views_matches_the_plan() -> None:
    """The same staleness guard `freematch`'s Tier 0 carries, for the same reason."""
    text = CARD.read_text(encoding="utf-8")
    mapping = text.split("### 3.2 Mapping to xty2", 1)[1].split("## 4.", 1)[0]
    checklist = text.split("## 4. Mechanics checklist", 1)[1].split("## 5.", 1)[0]
    for section, where in ((mapping, "§3.2"), (checklist, "§4")):
        assert f"FeatureMask(p={STRONG_MASK_RATE})" in section, (
            f"{where} does not name the strong view the plan runs "
            f"(p={STRONG_MASK_RATE})"
        )
        assert f"FeatureMask(p={WEAK_MASK_RATE})" in section
        for line in section.splitlines():
            if "FeatureMask(p=0.5)" in line:
                assert "fixmatch" in line or "deviation 2" in line, (
                    f"{where} names FeatureMask(p=0.5) without saying it is "
                    f"`fixmatch`'s: {line!r}"
                )


def test_the_term_declares_the_conventions_the_card_asks_it_to_publish() -> None:
    """Card §6.2's fifth Tier 0 assertion: the digest carries both conventions.

    `plan_details()` is what a reviewer diffs against §3.2 and §4, so the
    denominator convention and the alignment target have to be *in* it rather
    than merely true of the code.
    """
    stage = compile(softmatch(_recipe_schema())).stage("joint_fit")
    assert stage.objectives[2].plan_details == (
        "label = arg max of the unaligned target realisation",
        "weight = truncated Gaussian of aligned confidence (eq. 9)",
        "mu_hat and sigma_hat^2 use EMA decay 0.999 (eq. 7)",
        "sigma_hat^2 uses the unbiased B_U/(B_U-1) batch variance",
        "Gaussian denominator = 2 * sigma_hat^2 / 2^2",
        "weight confidence alignment = uniform",
        "uniform alignment target = u(K); pseudo-label remains unaligned",
        "all three EMAs fold in this batch before this batch is weighted",
        "denominator = every eligible row; weights multiply inside the mean",
    )


# ---------------------------------------------------------------------------
# Card §4 against the plan (`FIDELITY.md` §1.2, `CLAUDE.md` hard rule 4)
# ---------------------------------------------------------------------------


def _card_section_four() -> dict[str, str | dict[str, str]]:
    """Card §4 as data: `{canonical_key: value}` or `{key: {scope: value}}`."""
    text = CARD.read_text(encoding="utf-8")
    section = text.split("## 4. Mechanics checklist", 1)[1].split(
        "## 5. Deviations from the paper", 1
    )[0]
    match = re.search(r"```yaml\n(.*?)```", section, re.DOTALL)
    assert match is not None
    answered: dict[str, str | dict[str, str]] = {}
    current = ""
    key = ""
    for line in match.group(1).splitlines():
        statement = line.split("#", 1)[0].rstrip()
        if not statement:
            continue
        indent = len(statement) - len(statement.lstrip())
        name, _, value = statement.strip().partition(":")
        if indent == 0:
            current = name
        elif indent == 2:
            key = f"{current}.{name}"
            if value.strip() == "n/a":
                key = ""
                continue
            answered[key] = value.strip()
        elif indent == 4 and key:
            nested = answered.get(key)
            if not isinstance(nested, dict):
                nested = {}
                answered[key] = nested
            nested[name] = value.strip()
    return answered


def _rendered(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


def test_every_answered_card_key_reaches_the_plan() -> None:
    """`CLAUDE.md`: a non-`n/a` §4 key must reach `plan.hyperparameters`."""
    plan = compile(softmatch(_recipe_schema())).plan
    answered = set(_card_section_four())
    missing = sorted(answered - set(plan.hyperparameters))
    assert not missing, "card keys missing from plan: " + ", ".join(missing)
    assert "losses.confidence_threshold" in answered
    assert plan.hyperparameters["losses.confidence_threshold"] == SOFTMATCH_WEIGHTING


def test_the_card_and_the_plan_agree_on_every_value_section_four_states() -> None:
    """Key presence is not the cross-check; the values are."""
    hyperparameters = compile(softmatch(_recipe_schema())).plan.hyperparameters
    mismatched: list[str] = []
    symbolic = {"architecture.widths_depths": {"K": "2", "X_REPR": "200"}}
    checked = 0
    for key, stated in _card_section_four().items():
        planned = hyperparameters.get(key)
        if planned is None:
            mismatched.append(f"{key}: absent from the plan")
            continue
        if isinstance(stated, str):
            if not isinstance(planned, dict) and _rendered(planned) != stated:
                mismatched.append(f"{key}: card {stated!r} vs plan {planned!r}")
            checked += 1
            continue
        assert isinstance(planned, dict), f"{key} is scoped in the card only"
        for scope, value in stated.items():
            if scope not in planned:
                mismatched.append(f"{key}[{scope}]: absent from the plan")
                continue
            resolved = value
            for symbol, concrete in symbolic.get(key, {}).items():
                resolved = resolved.replace(symbol, concrete)
            if _rendered(planned[scope]) != resolved:
                mismatched.append(
                    f"{key}[{scope}]: card {resolved!r} vs plan {planned[scope]!r}"
                )
            checked += 1
    assert not mismatched, "card and plan disagree: " + "; ".join(mismatched)
    assert checked >= 55


def test_the_card_status_matches_the_tier2_evidence_and_names_the_recipe() -> None:
    """§6.3 carries a recorded ten-seed run, so the status is its outcome.

    `deviating` rather than `reproduced` because one of the eight declared
    targets — §6's quality guardrail on the weighted pseudo-label impurity —
    is missed, and `FIDELITY.md` §3 forbids retuning a tolerance after seeing
    a result. The written §5 explanation the status requires is asserted here
    too, because `assert_result_matches_card` only sees it on a nightly.
    """
    text = CARD.read_text(encoding="utf-8")
    assert "**Status:** `deviating`" in text
    assert "### Tier 2 outcome" in text
    assert "weighted_impurity_vs_1.25x_gate" in text
    assert "| [`softmatch.md`](recipes/softmatch.md) | `softmatch` |" in (
        CARD.parents[1] / "RECIPES.md"
    ).read_text(encoding="utf-8")
    assert repr(SOFTMATCH_WEIGHTING) in text
