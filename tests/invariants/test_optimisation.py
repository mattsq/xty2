"""Tier 0 — the optimisation declarations (`DESIGN.md` §9.1, `FIDELITY.md` §2).

Everything in `core/optimisation.py` exists so that a paper-governed number
cannot be inherited from a framework default. The tests are therefore mostly
rejections: a spec missing a field, a knob the chosen optimiser would ignore, a
threshold on a disabled clip. Each one is a way a recipe could read as though
it stated the paper's optimiser while running something else.

The two that are not rejections check the things a card cannot: that the weight
decay reaches the parameters the spec says it reaches, and that the learning
rate at a step is what the schedule says it is.
"""

import pytest
import torch
from xty2.core import (
    REQUIRED,
    CompileError,
    Constant,
    GradientClipping,
    Ramp,
    Step,
    WeightDecay,
    card_hyperparameters,
    gradient_norm,
)

from tests.invariants.conftest import optimiser


def _linear() -> torch.nn.Linear:
    return torch.nn.Linear(3, 2)


# ---------------------------------------------------------------------------
# No paper-governed field falls through to a default (DESIGN.md §9.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["name", "lr", "weight_decay", "lr_schedule", "clipping"]
)
def test_every_bound_field_must_be_set(field: str) -> None:
    with pytest.raises(CompileError, match="no usable default"):
        optimiser(**{field: REQUIRED})


def test_weight_decay_needs_parameter_reach() -> None:
    # `optimisation.weight_decay` is coefficient, component scope and bias/norm
    # reach (`FIDELITY.md` §2), so a spec that names only the number has not
    # answered the key.
    with pytest.raises(CompileError, match="no usable default"):
        WeightDecay(value=0.1)


def test_weight_decay_needs_component_scope() -> None:
    with pytest.raises(CompileError, match="no usable default"):
        WeightDecay(value=0.1, on_norm_and_bias=False)


def test_clipping_needs_its_mode() -> None:
    with pytest.raises(CompileError, match="no usable default"):
        GradientClipping()


def test_every_bound_field_reaches_the_card_keys() -> None:
    keys = card_hyperparameters(
        optimiser(
            name="adamw",
            lr=1e-3,
            weight_decay=WeightDecay(
                value=0.01, on_norm_and_bias=False, components=None
            ),
            lr_schedule=Ramp(0.0, 1.0, steps=100),
            clipping=GradientClipping.by_norm(1.0),
        )
    )
    assert keys == {
        "optimisation.optimiser": "adamw(betas=(0.9, 0.999), eps=1e-08)",
        "optimisation.lr": 1e-3,
        "optimisation.lr_schedule": "ramp 0.0 -> 1.0 over 100 steps",
        "optimisation.weight_decay": (
            "0.01 (all trainable components; norm and bias exempt)"
        ),
        "gradients.gradient_clipping": "norm at 1.0",
    }


# ---------------------------------------------------------------------------
# A knob the optimiser ignores is a paper detail that vanished
# ---------------------------------------------------------------------------


def test_momentum_on_adam_is_rejected() -> None:
    with pytest.raises(CompileError, match="takes no momentum"):
        optimiser(name="adam", momentum=0.9)


def test_betas_on_sgd_are_rejected() -> None:
    with pytest.raises(CompileError, match="takes no betas"):
        optimiser(name="sgd", betas=(0.5, 0.9))


def test_nesterov_without_momentum_is_rejected() -> None:
    with pytest.raises(CompileError, match="Nesterov"):
        optimiser(name="sgd", nesterov=True)


def test_an_unknown_optimiser_says_what_exists() -> None:
    with pytest.raises(CompileError, match="unknown optimiser"):
        optimiser(name="lamb")


def test_a_negative_learning_rate_is_rejected() -> None:
    with pytest.raises(CompileError, match="lr must be positive"):
        optimiser(lr=-1.0)


def test_the_description_names_the_whole_optimiser() -> None:
    # The card key binds this string, so momentum has to be in it: "SGD" and
    # "SGD with Nesterov momentum 0.9" are one line apart in a paper and a
    # world apart in a run.
    spec = optimiser(name="sgd", momentum=0.9, nesterov=True)
    assert spec.description == "sgd(momentum=0.9, nesterov=True)"


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------


def test_a_threshold_on_a_disabled_clip_is_rejected() -> None:
    with pytest.raises(CompileError, match="disabled clip"):
        GradientClipping(mode="none", threshold=1.0)


def test_a_clip_without_a_threshold_is_rejected() -> None:
    with pytest.raises(CompileError, match="needs a threshold"):
        GradientClipping(mode="norm")


def test_an_unknown_clip_mode_is_rejected() -> None:
    with pytest.raises(CompileError, match="mode must be one of"):
        GradientClipping(mode="percentile", threshold=1.0)  # type: ignore[arg-type]


def test_norm_clipping_bounds_the_norm_and_reports_the_original() -> None:
    layer = _linear()
    layer(torch.ones(4, 3)).sum().backward()
    before = gradient_norm(layer.parameters())
    reported = GradientClipping.by_norm(0.01).apply(list(layer.parameters()))
    assert reported == pytest.approx(before)
    assert gradient_norm(layer.parameters()) == pytest.approx(0.01, rel=1e-5)


def test_value_clipping_bounds_every_element() -> None:
    layer = _linear()
    (layer(torch.ones(4, 3)).sum() * 10).backward()
    GradientClipping.by_value(0.5).apply(list(layer.parameters()))
    assert all(
        bool((p.grad.abs() <= 0.5 + 1e-9).all())
        for p in layer.parameters()
        if p.grad is not None
    )


def test_no_clipping_leaves_the_gradient_alone() -> None:
    layer = _linear()
    layer(torch.ones(4, 3)).sum().backward()
    before = [p.grad.clone() for p in layer.parameters() if p.grad is not None]
    reported = GradientClipping.none().apply(list(layer.parameters()))
    after = [p.grad for p in layer.parameters() if p.grad is not None]
    assert reported == pytest.approx(gradient_norm(layer.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_the_norm_of_no_gradients_is_zero() -> None:
    assert gradient_norm(_linear().parameters()) == 0.0


# ---------------------------------------------------------------------------
# What the executor reads
# ---------------------------------------------------------------------------


def test_the_learning_rate_is_the_base_times_the_schedule() -> None:
    spec = optimiser(lr=0.2, lr_schedule=Step((0.0, 1.0), (3,)))
    assert [spec.lr_at(step) for step in range(5)] == [0.0, 0.0, 0.0, 0.2, 0.2]


def test_a_bare_number_is_coerced_to_a_constant_schedule() -> None:
    assert optimiser(lr_schedule=2.0).lr_at(99) == pytest.approx(0.1)


def test_weight_decay_splits_the_parameter_groups() -> None:
    layer = _linear()
    spec = optimiser(
        weight_decay=WeightDecay(value=0.03, on_norm_and_bias=False, components=None)
    )
    groups = spec.build(list(layer.named_parameters())).param_groups
    assert [group["weight_decay"] for group in groups] == [0.03, 0.0]
    assert [p.ndim for p in groups[0]["params"]] == [2]
    assert [p.ndim for p in groups[1]["params"]] == [1]


def test_weight_decay_on_everything_is_one_group() -> None:
    layer = _linear()
    spec = optimiser(
        weight_decay=WeightDecay(value=0.03, on_norm_and_bias=True, components=None)
    )
    groups = spec.build(list(layer.named_parameters())).param_groups
    assert [group["weight_decay"] for group in groups] == [0.03]
    assert len(groups[0]["params"]) == 2


def test_weight_decay_can_be_scoped_to_one_component() -> None:
    first = _linear()
    second = _linear()
    parameters = [
        (f"encoder.{name}", parameter) for name, parameter in first.named_parameters()
    ]
    parameters += [
        (f"tarnet_head.{name}", parameter)
        for name, parameter in second.named_parameters()
    ]
    spec = optimiser(
        weight_decay=WeightDecay(
            value=0.03,
            on_norm_and_bias=False,
            components=("tarnet_head",),
        )
    )
    groups = spec.build(parameters).param_groups
    decayed = {id(parameter) for parameter in groups[0]["params"]}
    assert decayed == {id(second.weight)}
    assert all(id(parameter) not in decayed for parameter in first.parameters())


def test_a_scoped_decay_must_match_a_parameter_component() -> None:
    spec = optimiser(
        weight_decay=WeightDecay(
            value=0.03,
            on_norm_and_bias=False,
            components=("tarnet_head",),
        )
    )
    with pytest.raises(CompileError, match="must reach the component"):
        spec.build(list(_linear().named_parameters()))


def test_no_decay_is_one_undecayed_group() -> None:
    layer = _linear()
    groups = optimiser().build(list(layer.named_parameters())).param_groups
    assert [group["weight_decay"] for group in groups] == [0.0]


def test_building_over_nothing_says_what_is_wrong() -> None:
    with pytest.raises(CompileError, match="no parameters"):
        optimiser().build([])


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
        ("sgd", torch.optim.SGD),
    ],
)
def test_each_name_builds_its_optimiser(name: str, expected: type) -> None:
    layer = _linear()
    built = optimiser(name=name).build(list(layer.named_parameters()))
    assert isinstance(built, expected)


def test_the_plan_lines_name_every_card_field() -> None:
    lines = optimiser(
        name="adam",
        lr=1e-3,
        weight_decay=WeightDecay(value=0.0, on_norm_and_bias=False, components=None),
        lr_schedule=Constant(1.0),
        clipping=GradientClipping.by_value(0.5),
    ).describe_lines()
    assert lines == (
        "optimiser     adam(betas=(0.9, 0.999), eps=1e-08)",
        "lr            0.001",
        "lr schedule   constant 1.0",
        "weight decay  none",
        "clipping      value at 0.5",
    )
