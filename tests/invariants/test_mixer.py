"""Tier 0 — the loss mixer and the §6.2 logging surface.

Three things are established here.

**Weighting and reduction are the mixer's, not the objective's** (`DESIGN.md`
§6, §6.1), so the raw per-objective value stays comparable across runs and
across modes.

**A term with no eligible rows stays out of the total and stays in the log**
(§1.3). Excluded, because a mean over an empty tensor is a `NaN` that poisons
the step; logged, because an objective that has quietly stopped seeing rows is
exactly what the trace exists to make visible.

**The gradient cosines mean what they say.** They are checked against
objectives constructed so that their gradients are *analytically* identical
(`+1`) or *analytically* opposed (`-1`), because "turn it on and look at a run"
never established anything about them. The probe is off by default and so off
in CI: it costs one extra backward pass per objective and proves nothing about
correctness.
"""

from dataclasses import dataclass, field

import pytest
import torch
from torch import Tensor
from xty2.core import (
    DEFAULT,
    CompiledObjective,
    CompiledRun,
    ComponentGraph,
    LossError,
    LossTerm,
    Port,
    Ramp,
    Realisation,
    RowIndex,
    Rows,
    Stage,
    State,
    Step,
    TrainContext,
    Weighted,
    XTYBatch,
    compile,
    reduce_rows,
)
from xty2.training import (
    GradientProbe,
    GradientReport,
    LossMixer,
    MixedLoss,
    gradient_report,
)

from tests.invariants.conftest import (
    BATCH_SIZE,
    HIDDEN,
    ToyEncoder,
    backward,
    make_batch,
    make_schema,
    objective,
    two_head_recipe,
    weighted,
)

# ---------------------------------------------------------------------------
# Objectives constructed to have a known value and a known gradient
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaledRepr:
    """`per_row = scale * sum(x_repr)`, so the gradient direction is `scale`.

    Two of these with the same sign have analytically identical gradient
    directions and must give a cosine of `+1`; two with opposite signs are
    analytically opposed and must give `-1`. Nothing about the scale's
    magnitude may move the cosine, which is what makes `+2.0` against `+1.0` a
    real test rather than a restatement.
    """

    name: str
    scale: float
    rows: Rows = "all"
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(Port.X_REPR, DEFAULT)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset()

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del batch, ctx
        representation = state[DEFAULT][Port.X_REPR]
        assert isinstance(representation, Tensor)
        return reduce_rows(
            self.scale * representation.sum(dim=1), rows, diagnostics=self.diagnostics
        )


@dataclass(frozen=True)
class Constantly:
    """A term whose value is a fixed number on every eligible row."""

    name: str
    per_row: float
    rows: Rows = "all"

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(Port.X_REPR, DEFAULT)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset()

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del state, ctx
        return reduce_rows(
            torch.full((batch.batch_size,), self.per_row, dtype=batch.x.dtype), rows
        )


@dataclass(frozen=True)
class Misreporting:
    """An objective that lies about how many rows it saw."""

    name: str = "misreporting"
    rows: Rows = "all"

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(Port.X_REPR, DEFAULT)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset()

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del state, ctx, rows
        return LossTerm(value=torch.zeros((), dtype=batch.x.dtype), n=999)


@dataclass(frozen=True)
class NotATerm:
    """An objective that returns something other than a `LossTerm`."""

    name: str = "not_a_term"
    rows: Rows = "all"

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(Port.X_REPR, DEFAULT)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset()

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> float:
        del state, batch, rows, ctx
        return 0.0


# ---------------------------------------------------------------------------
# A one-component recipe, so "shared parameters" is unambiguous
# ---------------------------------------------------------------------------


def _encoder_run(*terms: Weighted) -> tuple[CompiledRun, ToyEncoder]:
    encoder = ToyEncoder(width=HIDDEN)
    recipe = two_head_recipe(
        system=ComponentGraph([encoder]),
        program=(Stage(name="fit", objectives=terms, trainable=("encoder",)),),
    )
    return compile(recipe), encoder


def _mix(
    *terms: Weighted,
    probe: GradientProbe | None = None,
    step: int = 0,
    batch: XTYBatch | None = None,
) -> tuple[MixedLoss, ToyEncoder]:
    run, encoder = _encoder_run(*terms)
    data = batch if batch is not None else make_batch()
    mixer = LossMixer.for_stage(run.stage("fit"), probe=probe)
    ctx = TrainContext(global_step=step, schema=make_schema(), stage="fit")
    mixed = mixer.mix(
        run.state("fit", data), data, ctx, parameters=tuple(encoder.parameters())
    )
    return mixed, encoder


def _report(mixed: MixedLoss) -> GradientReport:
    """The probe's report, asserting it actually ran."""
    assert mixed.gradients is not None
    return mixed.gradients


# ---------------------------------------------------------------------------
# The per-step log (DESIGN.md §6.2)
# ---------------------------------------------------------------------------


def test_every_objective_logs_its_raw_value_weight_and_row_count() -> None:
    mixed, _ = _mix(
        weighted(Constantly("a", per_row=2.0), weight=0.5),
        weighted(Constantly("b", per_row=3.0), weight=2.0),
    )
    first = mixed.term("a")
    assert (first.value, first.weight, first.weighted, first.n) == (
        2.0,
        0.5,
        1.0,
        BATCH_SIZE,
    )
    second = mixed.term("b")
    assert (second.value, second.weight, second.weighted) == (3.0, 2.0, 6.0)
    assert float(mixed.total) == pytest.approx(7.0)


def test_the_logged_value_is_unweighted() -> None:
    # An objective that applied its own weight would make the logged raw value
    # incomparable across runs, which is the whole reason weighting is here.
    heavy, _ = _mix(weighted(Constantly("a", per_row=2.0), weight=10.0))
    light, _ = _mix(weighted(Constantly("a", per_row=2.0), weight=0.1))
    assert heavy.term("a").value == light.term("a").value == 2.0


def test_the_log_carries_the_eligible_rows_and_the_reduction() -> None:
    mixed, _ = _mix(
        weighted(Constantly("a", per_row=1.0, rows="t_observed"), reduction="sum")
    )
    assert mixed.term("a").rows == ("t_observed",)
    assert mixed.term("a").reduction == "sum"


def test_an_objectives_diagnostics_reach_the_log() -> None:
    mixed, _ = _mix(weighted(ScaledRepr("a", 1.0, diagnostics={"probe": 0.25})))
    assert mixed.term("a").diagnostics == {"probe": 0.25}


def test_coverage_is_logged_per_objective() -> None:
    mixed, _ = _mix(weighted(Constantly("a", per_row=1.0, rows="t_observed")))
    # The fixture batch observes the treatment on four of seven rows.
    assert mixed.term("a").n == 4
    assert mixed.term("a").coverage == pytest.approx(4 / BATCH_SIZE)


def test_the_step_renders_as_a_table() -> None:
    mixed, _ = _mix(weighted(Constantly("a", per_row=2.0), weight=0.5))
    rendered = str(mixed)
    assert "step 0 (stage fit, B = 7" in rendered
    assert "a " in rendered
    assert "weighted" in rendered


def test_asking_for_an_objective_that_was_not_mixed_says_what_was() -> None:
    mixed, _ = _mix(weighted(Constantly("a", per_row=1.0)))
    with pytest.raises(LossError, match=r"\['a'\]"):
        mixed.term("b")


# ---------------------------------------------------------------------------
# The zero-eligible-row rule (DESIGN.md §1.3)
# ---------------------------------------------------------------------------


def test_a_zero_row_term_never_reaches_the_total() -> None:
    all_observed = make_batch(t_observed=torch.ones(BATCH_SIZE, dtype=torch.bool))
    mixed, _ = _mix(
        weighted(Constantly("present", per_row=1.0)),
        weighted(Constantly("absent", per_row=5.0, rows="t_missing")),
        batch=all_observed,
    )
    assert mixed.term("absent").n == 0
    assert mixed.term("absent").weighted == 0.0
    assert float(mixed.total) == pytest.approx(1.0)


def test_a_zero_row_term_is_still_logged() -> None:
    # Excluded from the total, present in the trace: an objective that has
    # stopped seeing rows must be visible, not absent.
    all_observed = make_batch(t_observed=torch.ones(BATCH_SIZE, dtype=torch.bool))
    mixed, _ = _mix(
        weighted(Constantly("present", per_row=1.0)),
        weighted(Constantly("absent", per_row=5.0, rows="t_missing")),
        batch=all_observed,
    )
    assert [term.name for term in mixed.terms] == ["present", "absent"]
    assert [term.name for term in mixed.active] == ["present"]
    assert mixed.term("absent").coverage == 0.0


def test_no_nan_reaches_the_total_when_every_term_is_empty() -> None:
    all_observed = make_batch(t_observed=torch.ones(BATCH_SIZE, dtype=torch.bool))
    mixed, _ = _mix(
        weighted(Constantly("absent", per_row=5.0, rows="t_missing")),
        batch=all_observed,
    )
    assert float(mixed.total) == 0.0
    assert not torch.isnan(mixed.total)


# ---------------------------------------------------------------------------
# Reduction and schedules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reduction, expected",
    [("mean", 2.0), ("sum", 8.0), ("population", 8.0 / BATCH_SIZE)],
)
def test_the_reduction_mode_decides_the_contribution(
    reduction: str, expected: float
) -> None:
    mixed, _ = _mix(
        weighted(
            Constantly("a", per_row=2.0, rows="t_observed"),
            reduction=reduction,  # type: ignore[arg-type]
        )
    )
    assert float(mixed.total) == pytest.approx(expected)


def test_the_weight_is_read_at_the_current_step() -> None:
    ramp = Ramp(0.0, 1.0, steps=10)
    start, _ = _mix(weighted(Constantly("a", per_row=4.0), weight=ramp), step=0)
    middle, _ = _mix(weighted(Constantly("a", per_row=4.0), weight=ramp), step=5)
    end, _ = _mix(weighted(Constantly("a", per_row=4.0), weight=ramp), step=10)
    assert [float(m.total) for m in (start, middle, end)] == [
        pytest.approx(0.0),
        pytest.approx(2.0),
        pytest.approx(4.0),
    ]


def test_a_step_schedule_is_read_at_the_current_step() -> None:
    schedule = Step((0.0, 1.0), (3,))
    before, _ = _mix(weighted(Constantly("a", per_row=1.0), weight=schedule), step=2)
    after, _ = _mix(weighted(Constantly("a", per_row=1.0), weight=schedule), step=3)
    assert float(before.total) == 0.0
    assert float(after.total) == 1.0


# ---------------------------------------------------------------------------
# The objective contract, enforced at the mixer
# ---------------------------------------------------------------------------


def test_an_objective_that_miscounts_its_rows_is_rejected() -> None:
    # `n` drives the coverage log and both the `sum` and `population`
    # reductions, so a wrong one is a wrong loss, not a wrong label.
    with pytest.raises(LossError, match="reported n = 999"):
        _mix(weighted(Misreporting()))


def test_an_objective_that_returns_something_else_is_rejected() -> None:
    with pytest.raises(LossError, match="compute returns a LossTerm"):
        _mix(weighted(NotATerm()))  # type: ignore[arg-type]


def test_a_mixer_with_no_objectives_is_rejected() -> None:
    with pytest.raises(LossError, match="at least one objective"):
        LossMixer(())


def test_the_mixer_reports_the_names_it_mixes() -> None:
    run, _ = _encoder_run(
        weighted(Constantly("a", per_row=1.0)), weighted(Constantly("b", per_row=1.0))
    )
    assert LossMixer.for_stage(run.stage("fit")).names == ("a", "b")


def test_the_mixer_reads_the_rows_the_compiler_resolved() -> None:
    # It is built from compiled objectives precisely so that it cannot
    # re-derive the eligible set: the intersection of stage and objective
    # scope is the compiler's answer (DESIGN.md §7.0).
    run, _ = _encoder_run(weighted(Constantly("a", per_row=1.0, rows="t_observed")))
    compiled = run.stage("fit").objectives[0]
    assert isinstance(compiled, CompiledObjective)
    assert compiled.rows == ("t_observed",)


# ---------------------------------------------------------------------------
# The gradient probe (DESIGN.md §6.2)
# ---------------------------------------------------------------------------


def test_the_probe_is_off_by_default() -> None:
    # Off by default is what makes it off in CI. Turning it on is a per-run
    # choice, and P9 is the packet that records a trace from one.
    assert not GradientProbe().enabled
    assert not GradientProbe().should_run(0)
    run, _ = _encoder_run(weighted(Constantly("a", 1.0)))
    assert not LossMixer.for_stage(run.stage("fit")).probe.enabled


def test_no_gradients_are_measured_without_a_probe() -> None:
    mixed, _ = _mix(weighted(ScaledRepr("a", 1.0)), weighted(ScaledRepr("b", -1.0)))
    assert mixed.gradients is None


def test_the_periodic_probe_runs_on_its_cadence() -> None:
    probe = GradientProbe.periodic(200)
    assert probe.enabled
    assert probe.should_run(0)
    assert probe.should_run(400)
    assert not probe.should_run(199)


def test_a_non_positive_cadence_is_rejected() -> None:
    with pytest.raises(LossError, match="positive step count or None"):
        GradientProbe(every=0)


def test_analytically_identical_gradients_give_a_cosine_of_plus_one() -> None:
    mixed, _ = _mix(
        weighted(ScaledRepr("a", 1.0)),
        weighted(ScaledRepr("c", 2.0)),
        probe=GradientProbe(every=1),
    )
    report = _report(mixed)
    assert report.cosines[("a", "c")] == pytest.approx(1.0, abs=1e-5)


def test_analytically_opposed_gradients_give_a_cosine_of_minus_one() -> None:
    mixed, _ = _mix(
        weighted(ScaledRepr("a", 1.0)),
        weighted(ScaledRepr("b", -1.0)),
        probe=GradientProbe(every=1),
    )
    report = _report(mixed)
    assert report.cosines[("a", "b")] == pytest.approx(-1.0, abs=1e-5)


def test_the_cosine_is_scale_free_but_the_norm_is_not() -> None:
    # The weight is what tells you a term has become numerically irrelevant, so
    # the norm has to move with it; the direction of the conflict must not.
    data = make_batch()
    light, _ = _mix(
        weighted(ScaledRepr("a", 1.0)),
        weighted(ScaledRepr("b", -1.0), weight=1.0),
        probe=GradientProbe(every=1),
        batch=data,
    )
    heavy, _ = _mix(
        weighted(ScaledRepr("a", 1.0)),
        weighted(ScaledRepr("b", -1.0), weight=8.0),
        probe=GradientProbe(every=1),
        batch=data,
    )
    weak, strong = _report(light), _report(heavy)
    assert weak.cosines[("a", "b")] == pytest.approx(
        strong.cosines[("a", "b")], abs=1e-5
    )
    assert strong.norms["b"] == pytest.approx(8.0 * weak.norms["b"], rel=1e-5)


def test_a_term_scheduled_to_zero_reads_as_numerically_irrelevant() -> None:
    # The other half of §6.2's rationale: a term that has been ramped out of
    # the loss should show a zero gradient norm rather than disappear.
    mixed, _ = _mix(
        weighted(ScaledRepr("a", 1.0)),
        weighted(ScaledRepr("b", -1.0), weight=Ramp(0.0, 1.0, steps=100)),
        probe=GradientProbe(every=1),
        step=0,
    )
    report = _report(mixed)
    assert report.norms["b"] == 0.0
    assert report.cosines[("a", "b")] == 0.0


def test_the_report_names_how_many_parameters_are_shared() -> None:
    mixed, encoder = _mix(
        weighted(ScaledRepr("a", 1.0)),
        weighted(ScaledRepr("b", -1.0)),
        probe=GradientProbe(every=1),
    )
    report = _report(mixed)
    assert report.shared_parameters == len(list(encoder.parameters()))
    assert report.step == 0


def test_the_report_renders_stably() -> None:
    mixed, _ = _mix(
        weighted(ScaledRepr("a", 1.0)),
        weighted(ScaledRepr("b", -1.0)),
        probe=GradientProbe(every=1),
    )
    rendered = _report(mixed).render()
    assert "gradients @ step 0" in rendered
    assert "cos(a, b) = -1.0000" in rendered
    assert rendered in str(mixed)


def test_the_probe_leaves_the_total_differentiable() -> None:
    # It measures with `retain_graph`, so the executor can still descend the
    # total afterwards. A probe that consumed the graph would turn logging on
    # into a crash at step 200.
    mixed, encoder = _mix(
        weighted(ScaledRepr("a", 1.0)),
        weighted(ScaledRepr("b", -1.0)),
        probe=GradientProbe(every=1),
    )
    backward(mixed.total)
    assert all(p.grad is not None for p in encoder.parameters())


def test_the_probe_does_not_populate_grad_by_itself() -> None:
    _, encoder = _mix(
        weighted(ScaledRepr("a", 1.0)),
        weighted(ScaledRepr("b", -1.0)),
        probe=GradientProbe(every=1),
    )
    assert all(p.grad is None for p in encoder.parameters())


def test_the_probe_needs_parameters_to_measure_on() -> None:
    run, _ = _encoder_run(
        weighted(ScaledRepr("a", 1.0)), weighted(ScaledRepr("b", -1.0))
    )
    mixer = LossMixer.for_stage(run.stage("fit"), probe=GradientProbe(every=1))
    batch = make_batch()
    ctx = TrainContext(global_step=0, schema=make_schema(), stage="fit")
    with pytest.raises(LossError, match="no parameters that require a gradient"):
        mixer.mix(run.state("fit", batch), batch, ctx)


def test_gradient_report_is_deterministic() -> None:
    shared = torch.nn.Parameter(torch.randn(4))
    contributions = {
        "a": (3.0 * shared).sum(),
        "b": (-1.0 * shared).sum(),
    }
    first = gradient_report(contributions, [shared], step=7)
    second = gradient_report(contributions, [shared], step=7)
    assert first.norms == second.norms
    assert first.cosines == second.cosines
    assert first.cosines[("a", "b")] == pytest.approx(-1.0)


def test_a_parameter_only_one_objective_reaches_is_not_shared() -> None:
    # "These two terms are fighting" is only a statement about parameters both
    # of them reach; a private parameter cannot be in conflict with anything.
    shared = torch.nn.Parameter(torch.randn(4))
    private = torch.nn.Parameter(torch.randn(4))
    contributions = {
        "a": (shared + private).sum(),
        "b": (-shared).sum(),
    }
    report = gradient_report(contributions, [shared, private], step=0)
    assert report.shared_parameters == 1
    assert report.cosines[("a", "b")] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# The mixer over the real objectives, through a compiled recipe
# ---------------------------------------------------------------------------


def test_the_mixer_runs_a_compiled_two_head_stage() -> None:
    run = compile(two_head_recipe())
    batch = make_batch()
    mixer = LossMixer.for_stage(run.stage("fit"))
    ctx = TrainContext(global_step=0, schema=make_schema(), stage="fit")
    mixed = mixer.mix(run.state("fit", batch), batch, ctx)
    assert [term.name for term in mixed.terms] == ["outcome_nll", "treatment_nll"]
    assert float(mixed.total) == 0.0  # the toy objectives are identically zero


def test_two_objectives_may_not_share_a_name_in_one_stage() -> None:
    # Restated here because the log is keyed by name: two terms sharing one
    # would be indistinguishable in the trace that exists to attribute a bad
    # number. The compiler is where it is caught.
    with pytest.raises(Exception, match="more than one objective called"):
        _encoder_run(
            objective("same", Port.X_REPR),
            objective("same", Port.X_REPR),
        )
