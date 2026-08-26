"""Tier 0 — weight schedules (`DESIGN.md` §6).

A schedule is the mechanism by which a paper's "λ = 0.5, ramped over the first
5,000 steps" becomes something a reviewer can read off the execution plan. Two
properties make that worth anything, and neither is obvious from the type:

* it is a **pure function of the step**, so a trace is reproducible and a
  gradient probe — which calls nothing extra — cannot be blamed for changing it;
* `describe()` is **stable**, so the plan diff is signal.

The rest is arithmetic, and it is here because a ramp that is off by one step at
its boundary is exactly the silent difference `FIDELITY.md` §2 lists under
`losses.schedules`.
"""

import math
from itertools import pairwise

import pytest
from xty2.core import (
    Constant,
    CosineDecay,
    ExponentialDecay,
    Ramp,
    Schedule,
    SigmoidRamp,
    Step,
    Xty2Error,
    as_schedule,
)

# ---------------------------------------------------------------------------
# Constant
# ---------------------------------------------------------------------------


def test_a_constant_is_the_same_at_every_step() -> None:
    schedule = Constant(0.25)
    assert [schedule(step) for step in (0, 1, 10_000)] == [0.25, 0.25, 0.25]
    assert schedule.nominal == 0.25
    assert schedule.describe() == "constant 0.25"


def test_a_non_finite_constant_is_rejected() -> None:
    # A NaN weight poisons the total and every gradient downstream of it, and
    # nothing further along has any way to attribute it.
    with pytest.raises(Xty2Error, match="must be finite"):
        Constant(float("nan"))


# ---------------------------------------------------------------------------
# Ramp
# ---------------------------------------------------------------------------


def test_a_ramp_starts_at_start_and_settles_at_end() -> None:
    schedule = Ramp(0.0, 0.5, steps=1_000)
    assert schedule(0) == 0.0
    assert schedule(500) == pytest.approx(0.25)
    assert schedule(1_000) == 0.5
    assert schedule(10_000) == 0.5


def test_a_ramp_reaches_its_end_exactly_at_its_length() -> None:
    # The off-by-one that matters: `steps` is the step at which the ramp is
    # finished, not the last step on which it is still climbing.
    schedule = Ramp(0.0, 1.0, steps=4)
    assert [schedule(step) for step in range(6)] == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]


def test_a_ramp_may_go_down() -> None:
    schedule = Ramp(1.0, 0.0, steps=2)
    assert [schedule(step) for step in range(4)] == [1.0, 0.5, 0.0, 0.0]
    assert schedule.nominal == 0.0


def test_a_ramp_reports_its_end_as_the_nominal_weight() -> None:
    # Papers state the weight and the ramp separately; the plan prints
    # `nominal` as the weight and `describe()` as the schedule, so a card's
    # `losses.weights` line can be checked against the paper's λ directly.
    assert Ramp(0.0, 0.5, steps=5_000).nominal == 0.5


def test_a_ramp_describes_itself_in_steps() -> None:
    assert Ramp(0.0, 0.5, steps=5_000).describe() == "ramp 0.0 -> 0.5 over 5000 steps"


def test_a_zero_length_ramp_is_rejected() -> None:
    with pytest.raises(Xty2Error, match="at least 1"):
        Ramp(0.0, 1.0, steps=0)


# ---------------------------------------------------------------------------
# Mean Teacher sigmoid ramp
# ---------------------------------------------------------------------------


def test_a_sigmoid_ramp_uses_the_pinned_mean_teacher_formula() -> None:
    schedule = SigmoidRamp(end=3.0, steps=40)
    assert schedule(0) == pytest.approx(3.0 * math.exp(-5.0))
    assert schedule(20) == pytest.approx(3.0 * math.exp(-1.25))
    assert schedule(40) == 3.0
    assert schedule(4_000) == 3.0


def test_a_sigmoid_ramp_reports_its_end_and_exact_formula() -> None:
    schedule = SigmoidRamp(end=3.0, steps=40)
    assert schedule.nominal == 3.0
    assert schedule.describe() == (
        "sigmoid ramp to 3.0 over 40 steps: 3.0 * exp(-5 * (1 - min(step/40, 1))^2)"
    )


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_a_sigmoid_ramp_needs_a_positive_integer_length(steps: object) -> None:
    with pytest.raises(Xty2Error, match="integer at least 1"):
        SigmoidRamp(end=1.0, steps=steps)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FixMatch cosine decay
# ---------------------------------------------------------------------------


def test_cosine_decay_is_the_fixmatch_rate_formula() -> None:
    # FixMatch §2.4: eta * cos(7 pi k / 16 K). Compared against that expression
    # written out, rather than against the implementation's own rearrangement.
    total = 3_000
    schedule = CosineDecay(steps=total, phase=7 / 16)
    for step in (0, 1, 750, 1_500, 2_999, 3_000):
        assert schedule(step) == pytest.approx(
            math.cos(7.0 * math.pi * step / (16.0 * total))
        )


def test_cosine_decay_holds_its_final_level_and_never_turns_back_up() -> None:
    schedule = CosineDecay(steps=100, phase=7 / 16)
    assert schedule(100) == pytest.approx(math.cos(7.0 * math.pi / 16.0))
    assert schedule(10_000) == schedule(100)
    values = [schedule(step) for step in range(101)]
    assert all(later <= earlier for earlier, later in pairwise(values))


def test_cosine_decay_reports_the_level_it_settles_at() -> None:
    schedule = CosineDecay(steps=100, phase=0.5, initial=0.2)
    assert schedule.nominal == pytest.approx(0.0, abs=1e-12)
    assert CosineDecay(steps=100, phase=7 / 16).nominal == pytest.approx(
        math.cos(7.0 * math.pi / 16.0)
    )


def test_cosine_decay_describes_its_formula_stably() -> None:
    assert CosineDecay(steps=3_000, phase=7 / 16).describe() == (
        "cosine 1.0 * cos(pi * 0.4375 * min(step/3000, 1))"
    )


@pytest.mark.parametrize("phase", [0.0, -0.1, 0.75, 1.0])
def test_a_phase_that_would_go_negative_is_rejected(phase: float) -> None:
    with pytest.raises(Xty2Error, match=r"phase must be in \(0, 0.5\]"):
        CosineDecay(steps=100, phase=phase)


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_cosine_decay_needs_a_positive_integer_length(steps: object) -> None:
    with pytest.raises(Xty2Error, match="integer at least 1"):
        CosineDecay(steps=steps, phase=7 / 16)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


def test_a_step_schedule_jumps_at_its_boundaries() -> None:
    schedule = Step((0.0, 0.5, 1.0), (2, 5))
    assert [schedule(step) for step in range(7)] == [0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0]
    assert schedule.nominal == 1.0
    assert schedule.describe() == "step 0.0 from 0, 0.5 from 2, 1.0 from 5"


def test_a_step_needs_one_more_weight_than_boundary() -> None:
    with pytest.raises(Xty2Error, match="exactly one more weight than boundary"):
        Step((0.0, 1.0), (2, 5))


def test_step_boundaries_must_increase() -> None:
    with pytest.raises(Xty2Error, match="strictly increasing"):
        Step((0.0, 0.5, 1.0), (5, 2))


def test_a_step_boundary_at_zero_is_rejected() -> None:
    # The weight before the first boundary is `weights[0]`, so a boundary at 0
    # would name a segment of zero length and make the first weight dead.
    with pytest.raises(Xty2Error, match="positive step numbers"):
        Step((0.0, 1.0), (0,))


# ---------------------------------------------------------------------------
# Staircase exponential decay
# ---------------------------------------------------------------------------


def test_exponential_decay_jumps_at_the_declared_step_interval() -> None:
    schedule = ExponentialDecay(gamma=0.97, every=100)
    assert schedule(0) == 1.0
    assert schedule(99) == 1.0
    assert schedule(100) == pytest.approx(0.97)
    assert schedule(299) == pytest.approx(0.97**2)
    assert schedule.describe() == "staircase 1.0 * 0.97^floor(step/100)"


@pytest.mark.parametrize("gamma", [0.0, 1.01])
def test_exponential_decay_rejects_a_non_decay_factor(gamma: float) -> None:
    with pytest.raises(Xty2Error, match="in \\(0, 1\\]"):
        ExponentialDecay(gamma=gamma, every=100)


def test_exponential_decay_needs_a_positive_interval() -> None:
    with pytest.raises(Xty2Error, match="at least 1"):
        ExponentialDecay(gamma=0.97, every=0)


# ---------------------------------------------------------------------------
# The contract every schedule holds to
# ---------------------------------------------------------------------------


def _schedules() -> list[Schedule]:
    return [
        Constant(1.0),
        Ramp(0.0, 0.5, steps=100),
        SigmoidRamp(end=2.0, steps=40),
        CosineDecay(steps=100, phase=7 / 16),
        Step((0.0, 1.0), (50,)),
        ExponentialDecay(gamma=0.97, every=100),
    ]


@pytest.mark.parametrize("schedule", _schedules(), ids=lambda s: type(s).__name__)
def test_a_schedule_is_a_pure_function_of_the_step(schedule: Schedule) -> None:
    # Called repeatedly, and out of order: a schedule with an internal cursor
    # would pass the first assertion and fail the second.
    forwards = [schedule(step) for step in range(120)]
    assert [schedule(step) for step in range(120)] == forwards
    assert [schedule(step) for step in reversed(range(120))] == list(reversed(forwards))


@pytest.mark.parametrize("schedule", _schedules(), ids=lambda s: type(s).__name__)
def test_a_negative_step_is_rejected(schedule: Schedule) -> None:
    with pytest.raises(Xty2Error, match="non-negative"):
        schedule(-1)


@pytest.mark.parametrize("schedule", _schedules(), ids=lambda s: type(s).__name__)
def test_str_is_describe(schedule: Schedule) -> None:
    assert str(schedule) == schedule.describe()


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


def test_a_number_is_a_constant_schedule() -> None:
    assert as_schedule(0.5) == Constant(0.5)
    assert as_schedule(Ramp(0.0, 1.0, steps=2)) == Ramp(0.0, 1.0, steps=2)


def test_a_missing_weight_is_not_coerced_to_anything() -> None:
    # A number is an unambiguous schedule; nothing else is. `Weighted` has no
    # default weight at all, so there is nothing here for `None` to mean.
    with pytest.raises(Xty2Error, match="a number or a Schedule"):
        as_schedule(None)  # type: ignore[arg-type]
