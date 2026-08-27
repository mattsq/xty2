"""Data-loading declarations (`docs/proposals/loader.md`).

Two things were conflated under the word "loader", and they are known at
different times. The **policy** — how a population is split, standardised, made
missing and turned into batches — is part of the method: it is what
`FIDELITY.md` §2's `optimisation.batch_size`,
`optimisation.labelled_unlabelled_ratio` and `data.*` keys are about, and until
this module existed those five keys had nothing to bind to and three reviewed
cards carried `framework-limitation` rows saying so. The **data** is not part
of the method, and does not come near a `Recipe`.

So the declarations live here and are read by the compiler; the rows arrive at
run time as a `Dataset`; and `xty2.training.loading` is what applies one to the
other. The recipe never holds a tensor of training rows, which is the obstacle
`scarf.md` §5.1 named when it deferred this.

The second principle is `DESIGN.md` §7.1's, and it is why this is not a config
object: **the thing that does the work is the thing that reports it, and the
report is derived rather than declared.** A caller-supplied loader that
*reported* the batch size it used would be a provenance claim nothing can
falsify. So the executor builds the batches from the compiled policy, the plan
is the cause of the run rather than a description of it, and
`TrainingPopulation` carries the row ids its statistics were fitted on — the
same shape as `Checkpoint.trained_on_row_ids`, checked rather than trusted.

`ExternalBatches` is the one declaration that gives the rows back to the
caller. It is explicit, it prints in the plan, and it is rejected for any stage
holding a batch-coupled objective, which is where batch size is method-bearing.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Literal, TypeAlias, final, get_args

import torch
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.card_keys import REQUIRED, is_required
from xty2.core.errors import ArtifactError, CompileError, Xty2Error, require_str
from xty2.core.rows import Rows, row_mask, validate_population
from xty2.core.schema import Schema

Standardisation = Literal["none", "zscore"]
"""How a block of columns is centred and scaled. `zscore` is fitted on the
train assignment and applied everywhere; `none` passes values through."""

Mechanism = Literal["observed", "mcar"]
"""Where a row's treatment missingness comes from.

`observed` takes the dataset's own `t_observed` mask unchanged — the honest
declaration for data that arrives with its missingness already in it.
`mcar` induces missingness at a declared rate, which is what every Tier 1
fixture in this repository does by hand today.
"""


# ---------------------------------------------------------------------------
# The policy a recipe declares
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitSpec:
    """Which rows are the training population, and what protocol says so.

    The framework does not *derive* splits. Every card in this repository takes
    its split from the archive it benchmarks on or from the fixture that
    generated the data, so a random-fraction splitter here would be a mechanism
    no card asked for. What was missing is not the splitting; it is that
    nothing said which assignment is train, so nothing could check that the
    standardisation was fitted on it.

    Attributes:
        protocol: Prose naming the protocol, bound to `data.split_protocol`.
            The card governs it; this is where it stops being `n/a`.
        train: The `Dataset.assignments` entry holding the training rows.
    """

    protocol: str = REQUIRED
    train: str = "train"

    def __post_init__(self) -> None:
        if is_required(self.protocol):
            raise CompileError(
                "SplitSpec was constructed without `protocol`, the field bound "
                "to card key 'data.split_protocol'. The split a published "
                "number depends on is paper-governed, so it has no usable "
                "default (DESIGN.md §9.1)."
            )
        require_str("SplitSpec.protocol", self.protocol, error=CompileError)
        name = require_str("SplitSpec.train", self.train, error=CompileError)
        if not name.isidentifier():
            raise CompileError(
                f"SplitSpec.train is {name!r}; it names a Dataset assignment "
                "and must be a Python identifier"
            )

    def describe(self) -> str:
        return f"{self.protocol}; training rows are assignment {self.train!r}"


@dataclass(frozen=True)
class PreprocessSpec:
    """Standardisation, and — decisively — the rows it is fitted on.

    `FIDELITY.md` §2 annotates `data.standardisation` with "fit on which split?
    a classic leakage point", and `tarnet.md` §5.5 records the cost of having
    nowhere to say: *"a later runner could fit it on the wrong split and
    nothing in the plan would say so"*. There is no `fit_on` field here, and
    that is deliberate — the fit split is `SplitSpec.train`, so the two cannot
    disagree, and the leakage check has one thing to check against.
    """

    features: Standardisation = REQUIRED
    outcome: Standardisation = REQUIRED

    def __post_init__(self) -> None:
        for name in ("features", "outcome"):
            value = getattr(self, name)
            if is_required(value):
                raise CompileError(
                    f"PreprocessSpec was constructed without {name!r}. It binds "
                    "a `data.*` card key and is governed by the card, so it has "
                    "no usable default (DESIGN.md §9.1)."
                )
            if value not in get_args(Standardisation):
                raise CompileError(
                    f"PreprocessSpec.{name} is {value!r}; expected one of "
                    f"{list(get_args(Standardisation))!r}"
                )


@dataclass(frozen=True)
class MissingnessSpec:
    """How treatment missingness arises (`data.missingness_mechanism`).

    A rate belongs to `mcar` and to nothing else: an `observed` mechanism that
    also carried a rate would be a number the run ignores, printed in the plan
    as though it governed something.
    """

    mechanism: Mechanism = REQUIRED
    rate: float | None = None
    observed: int | None = None

    def __post_init__(self) -> None:
        if is_required(self.mechanism):
            raise CompileError(
                "MissingnessSpec was constructed without `mechanism`, the field "
                "bound to card key 'data.missingness_mechanism' (DESIGN.md §9.1)."
            )
        if self.mechanism not in get_args(Mechanism):
            raise CompileError(
                f"MissingnessSpec.mechanism is {self.mechanism!r}; expected one "
                f"of {list(get_args(Mechanism))!r}"
            )
        if self.mechanism == "observed":
            if self.rate is not None or self.observed is not None:
                raise CompileError(
                    "MissingnessSpec(mechanism='observed') takes the dataset's "
                    "own mask, so a rate or a label budget would govern nothing "
                    "while printing in the plan as though it did."
                )
            return
        if (self.rate is None) == (self.observed is None):
            raise CompileError(
                "MissingnessSpec(mechanism='mcar') states its budget exactly "
                "once: `rate` as a missing fraction, or `observed` as a count "
                "of labelled rows. Both would let the plan print one number "
                "while the run used the other."
            )
        if self.observed is not None:
            if type(self.observed) is not int or self.observed < 0:
                raise CompileError(
                    f"MissingnessSpec.observed must be a non-negative int, got "
                    f"{self.observed!r}"
                )
            return
        if not isinstance(self.rate, (int, float)) or isinstance(self.rate, bool):
            raise CompileError(
                f"MissingnessSpec(mechanism='mcar') needs a rate, got {self.rate!r}"
            )
        if not 0.0 <= float(self.rate) < 1.0:
            raise CompileError(
                f"MissingnessSpec rate must lie in [0, 1), got {self.rate!r}"
            )

    def describe(self) -> str:
        """The plan's line, and `data.missingness_mechanism`'s value.

        A **count** is a first-class budget rather than a rate rounded off,
        because that is how the semi-supervised literature states it: FixMatch
        reports CIFAR-10 at 40, 250 and 4,000 *labels*, and SCARF's
        label-scarce regime is stated the same way. A rate would make the
        declaration depend on how many rows the population happened to hold,
        which is exactly the kind of silent difference the card key exists to
        stop.
        """
        if self.mechanism == "observed":
            return "observed: the dataset's own t_observed mask, unchanged"
        if self.observed is not None:
            return (
                f"treatment MCAR to a budget of {self.observed} labelled rows, "
                "keyed by row_id"
            )
        return f"treatment MCAR at {float(self.rate or 0.0):.3f}, keyed by row_id"


@dataclass(frozen=True)
class DataSpec:
    """The recipe's data policy, and the owner of the four `data.*` card keys.

    The keys are bound here rather than on the three parts because a canonical
    key names one value and the split the standardisation was fitted on is part
    of what `data.standardisation` means. Composing the values from the parts
    is what keeps `SplitSpec.train` the single source of that fact.
    """

    split: SplitSpec = REQUIRED
    preprocess: PreprocessSpec = REQUIRED
    missingness: MissingnessSpec = REQUIRED

    CARD_KEYS: ClassVar[Mapping[str, str]] = {
        "split_protocol": "data.split_protocol",
        "standardisation": "data.standardisation",
        "outcome_scaling": "data.outcome_scaling",
        "missingness_mechanism": "data.missingness_mechanism",
    }

    def __post_init__(self) -> None:
        for name, expected in (
            ("split", SplitSpec),
            ("preprocess", PreprocessSpec),
            ("missingness", MissingnessSpec),
        ):
            value = getattr(self, name)
            if is_required(value):
                raise CompileError(
                    f"DataSpec was constructed without {name!r}. A recipe whose "
                    "stages sample declares a complete policy; a partial one "
                    "would leave a `data.*` card key with nothing to bind to, "
                    "which is what this capability exists to end."
                )
            if not isinstance(value, expected):
                raise CompileError(
                    f"DataSpec.{name} is {type(value)}, expected {expected.__name__}"
                )

    @property
    def split_protocol(self) -> str:
        """`data.split_protocol`."""
        return self.split.describe()

    @property
    def standardisation(self) -> str:
        """`data.standardisation`, carrying the split it is fitted on."""
        return f"x: {self.preprocess.features} fitted on {self.split.train!r}"

    @property
    def outcome_scaling(self) -> str:
        """`data.outcome_scaling`, carrying the split it is fitted on."""
        return f"y: {self.preprocess.outcome} fitted on {self.split.train!r}"

    @property
    def missingness_mechanism(self) -> str:
        """`data.missingness_mechanism`."""
        return self.missingness.describe()

    def describe_lines(self) -> tuple[str, ...]:
        """The plan's `data` block. Stable, so the digest covers the policy."""
        return (
            f"split            {self.split_protocol}",
            f"standardisation  {self.standardisation}",
            f"outcome scaling  {self.outcome_scaling}",
            f"missingness      {self.missingness_mechanism}",
        )

    @property
    def digest(self) -> str:
        """`sha256` of the rendered policy — what a population records it came from."""
        return hashlib.sha256("\n".join(self.describe_lines()).encode()).hexdigest()


# ---------------------------------------------------------------------------
# What a stage draws
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExternalBatches:
    """The caller supplies the batches; no policy is enforced for this stage.

    Explicit, never inherited: `Stage.sampler` is `REQUIRED`, so this is
    something a recipe *says*, and it prints in the plan where its absence
    would not. It is what four reviewed recipes declare, because
    `FIDELITY.md` §5.3 is explicit that repaying a debt is not licence to
    churn cards that are not paying it — and because "the caller supplies
    batches" is the true answer for a stage whose card marks
    `optimisation.batch_size` as `n/a`.

    The bar that stops it becoming an escape hatch is in `compile()`: a stage
    declaring this may not hold an objective whose value depends on the other
    rows of the batch.
    """

    def describe(self) -> str:
        return "external: the caller supplies batches; no quota is enforced"


@dataclass(frozen=True)
class Quota:
    """How many rows of one population enter each step's batch.

    `rows` is the `Rows` vocabulary of `DESIGN.md` §1.3 — the same words every
    objective and every stage scope already use, so "which rows" needs no new
    vocabulary and only "how many" does.

    Attributes:
        rows: The population drawn from.
        size: Rows per step, or rows *per level* when `stratify` is set.
        stratify: A categorical quantity whose levels are drawn evenly. `t` is
            the only one v1 can stratify on; it exists because PAWS draws a
            class-balanced support batch, which a two-field labelled/unlabelled
            sampler could not express.

    A quota larger than its population is an error, and cannot be otherwise:
    a repeating shuffled loader — which is what FixMatch's reference iterates
    the labelled set as, drawing `B = 64` from a 40-label split — would put the
    same row in one batch twice, and `XTYBatch.row_id` must be unique because
    artifacts and provenance are keyed by it (`DESIGN.md` §7.1). The ledger
    carries that as `batch-row-repetition`.
    """

    rows: Rows
    size: int
    stratify: Literal["t"] | None = None

    def __post_init__(self) -> None:
        validate_population(self.rows)
        if type(self.size) is not int or self.size < 1:
            raise CompileError(
                f"Quota over {self.rows!r} has size={self.size!r}; it must be a "
                "positive int"
            )
        if self.stratify not in (None, "t"):
            raise CompileError(
                f"Quota.stratify is {self.stratify!r}; v1 stratifies on 't' or "
                "on nothing"
            )
        if self.stratify == "t" and self.rows == "t_missing":
            raise CompileError(
                "a Quota cannot stratify on `t` over `t_missing` rows: the "
                "treatment is unobserved there, and reading it is a bug of the "
                "same kind as reading a sentinel (DESIGN.md §1.1)."
            )

    def describe(self) -> str:
        per = (
            f"{self.size} per {self.stratify}-level"
            if self.stratify
            else str(self.size)
        )
        return f"{self.rows:<12} {per}"


@dataclass(frozen=True)
class UniformSampler:
    """One fresh permutation of the population per step, first `batch_size` rows.

    Defined as exactly what every fixture in this repository already does by
    hand, so that adopting it is a change of *owner* rather than of arithmetic
    — a Tier 0 test asserts it against
    `xty2.evaluation.benchmarks.common.batch_indices` for a fixed seed.

    `labelled_unlabelled_ratio` is deliberately unbound: a card using this
    sampler marks the key `n/a`, and that `n/a` now says "this sampler enforces
    no quota", which is a fact about the method, rather than "xty2 has no
    loader", which was a fact about us.
    """

    batch_size: int = REQUIRED
    replacement: bool = False

    CARD_KEYS: ClassVar[Mapping[str, str]] = {"batch_size": "optimisation.batch_size"}

    def __post_init__(self) -> None:
        if is_required(self.batch_size):
            raise CompileError(
                "UniformSampler was constructed without `batch_size`, the field "
                "bound to card key 'optimisation.batch_size'. It has no usable "
                "default (DESIGN.md §9.1)."
            )
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise CompileError(
                f"UniformSampler batch_size must be a positive int, got "
                f"{self.batch_size!r}"
            )
        if type(self.replacement) is not bool:
            raise CompileError(
                f"UniformSampler replacement must be bool, got {self.replacement!r}"
            )

    @property
    def rows(self) -> tuple[Rows, ...]:
        """The populations this sampler draws from — every row."""
        return ("all",)

    def describe(self) -> str:
        how = "with" if self.replacement else "without"
        return f"uniform, batch_size={self.batch_size}, {how} replacement"


@dataclass(frozen=True)
class QuotaSampler:
    """A fixed number of rows from each declared population, every step.

    This is FixMatch's `mu`: eq. (5) mixes a labelled batch of `B` with an
    unlabelled batch of `mu B`, and until now nothing in a compiled recipe said
    so.

    **Both card keys are derived, not constructor arguments**, for the reason
    `DESIGN.md` §7.1 gives about `PseudoLabels.used_y`: a field a producer can
    set is a field a producer can set wrongly. `batch_size` is the sum of the
    quotas and `labelled_unlabelled_ratio` is their ratio, so the plan prints
    the ratio the sampler *runs* and `mu = 7` cannot be claimed by a recipe
    that draws 64 and 64.
    """

    quotas: tuple[Quota, ...] = REQUIRED

    CARD_KEYS: ClassVar[Mapping[str, str]] = {
        "batch_size": "optimisation.batch_size",
        "labelled_unlabelled_ratio": "optimisation.labelled_unlabelled_ratio",
    }

    def __post_init__(self) -> None:
        if is_required(self.quotas):
            raise CompileError(
                "QuotaSampler was constructed without `quotas`. The batch "
                "composition is the mechanic this sampler exists to pin down, "
                "so it has no default (DESIGN.md §9.1)."
            )
        # Through `object`, as `Weighted` checks its objective: the annotation
        # says tuple, the sentinel default says the annotation is a promise the
        # recipe has not kept yet, and only a runtime check can tell.
        candidate: object = self.quotas
        if isinstance(candidate, (str, bytes)) or not isinstance(candidate, Sequence):
            raise CompileError(
                f"QuotaSampler.quotas is {type(candidate)}; expected a sequence "
                "of Quota"
            )
        quotas = tuple(candidate)
        object.__setattr__(self, "quotas", quotas)
        if not quotas:
            raise CompileError("QuotaSampler needs at least one Quota")
        for quota in quotas:
            if not isinstance(quota, Quota):
                raise CompileError(
                    f"QuotaSampler.quotas holds {type(quota)}; expected Quota"
                )
        seen = [quota.rows for quota in quotas]
        duplicates = sorted({name for name in seen if seen.count(name) > 1})
        if duplicates:
            raise CompileError(
                f"QuotaSampler draws from {duplicates!r} more than once. Two "
                "quotas over one population are one quota with a larger size, "
                "and writing them apart would let the plan print a ratio the "
                "sampler does not run."
            )

    @property
    def rows(self) -> tuple[Rows, ...]:
        """The populations this sampler draws from, in declared order."""
        return tuple(quota.rows for quota in self.quotas)

    @property
    def batch_size(self) -> int:
        """`optimisation.batch_size` — derived, so it cannot be misdeclared."""
        return sum(quota.size for quota in self.quotas)

    @property
    def labelled_unlabelled_ratio(self) -> float | None:
        """`optimisation.labelled_unlabelled_ratio` — derived from the quotas.

        `None` where the recipe draws no labelled/unlabelled pair, which is a
        different statement from a ratio of one and prints as one.
        """
        sizes = {quota.rows: quota.size for quota in self.quotas}
        labelled = sizes.get("t_observed")
        unlabelled = sizes.get("t_missing")
        if labelled is None or unlabelled is None:
            return None
        return unlabelled / labelled

    def describe_lines(self) -> tuple[str, ...]:
        ratio = self.labelled_unlabelled_ratio
        head = f"quota, batch_size={self.batch_size}, without replacement"
        if ratio is not None:
            head += f", mu={ratio:g}"
        return (head, *(f"  {quota.describe()}" for quota in self.quotas))

    def describe(self) -> str:
        return " / ".join(self.describe_lines())


SamplerSpec: TypeAlias = "UniformSampler | QuotaSampler | ExternalBatches"
"""What a stage declares about the rows it steps on."""

SAMPLERS: tuple[type, ...] = (UniformSampler, QuotaSampler, ExternalBatches)


def sampler_lines(sampler: SamplerSpec) -> tuple[str, ...]:
    """The plan's `sampler` lines for any declared sampler."""
    if isinstance(sampler, QuotaSampler):
        return sampler.describe_lines()
    return (sampler.describe(),)


def draws_from_population(sampler: SamplerSpec) -> bool:
    """Does this sampler need a `Dataset` rather than a caller's iterable?"""
    return not isinstance(sampler, ExternalBatches)


# ---------------------------------------------------------------------------
# What the caller supplies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """A finite, immutable row table and its named partitions.

    A population is already an `XTYBatch` — `row_id` and `fold_id` are on it —
    so this wrapper earns its place through `assignments`, and that is
    deliberate. `assignments` is a **mapping of named subsets**, not a single
    `split` field, because the `repeated-cross-fitting` ledger row is blocked
    on exactly one thing: an `XTYBatch` carries a single `fold_id`, so a second
    partition has nowhere to live. Naming subsets rather than fixing one is
    what makes a second additive later. This does not discharge that row — the
    artifact contract and the fold-disjointness check are still written against
    one assignment — it stops being what stands in the way.

    Entries are subsets, not a partition: nothing here requires them to be
    disjoint or to cover every row, because `train` and a fold complement are
    both legitimate and overlap.

    Attributes:
        schema: The schema the rows are validated against.
        rows: The whole population, as one batch object.
        assignments: `{name: [n] long}` positions into `rows`.
    """

    schema: Schema
    rows: XTYBatch
    assignments: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.schema, Schema):
            raise Xty2Error(f"Dataset.schema is {type(self.schema)}, expected Schema")
        if not isinstance(self.rows, XTYBatch):
            raise Xty2Error(f"Dataset.rows is {type(self.rows)}, expected XTYBatch")
        self.schema.validate_batch(self.rows)
        resolved: dict[str, Tensor] = {}
        for name, index in dict(self.assignments).items():
            label = require_str("Dataset assignment name", name, error=Xty2Error)
            if not label.isidentifier():
                raise Xty2Error(
                    f"Dataset assignment {label!r} must be a Python identifier"
                )
            if not isinstance(index, Tensor):
                raise Xty2Error(
                    f"Dataset assignment {label!r} must be a [n] tensor, got "
                    f"{type(index)}"
                )
            if index.ndim != 1:
                raise Xty2Error(
                    f"Dataset assignment {label!r} must be a [n] tensor, got "
                    f"shape {tuple(index.shape)}"
                )
            if index.dtype != torch.long:
                raise Xty2Error(
                    f"Dataset assignment {label!r} must be long, got {index.dtype}"
                )
            if index.numel() and (
                int(index.min()) < 0 or int(index.max()) >= self.rows.batch_size
            ):
                raise Xty2Error(
                    f"Dataset assignment {label!r} indexes outside its "
                    f"{self.rows.batch_size}-row population"
                )
            if torch.unique(index).numel() != index.numel():
                raise Xty2Error(
                    f"Dataset assignment {label!r} names a row more than once"
                )
            resolved[label] = index.detach().clone()
        object.__setattr__(
            self, "assignments", MappingProxyType(dict(sorted(resolved.items())))
        )

    @property
    def batch_size(self) -> int:
        """How many rows the population holds."""
        return self.rows.batch_size

    def assignment(self, name: str) -> Tensor:
        """The positions of `name`, or a clear error naming what is there."""
        try:
            return self.assignments[name]
        except KeyError:
            raise Xty2Error(
                f"dataset has no assignment {name!r}; it carries "
                f"{sorted(self.assignments)!r}. The recipe's SplitSpec names "
                "the training assignment, and the standardisation is checked "
                "against it (docs/proposals/loader.md §5)."
            ) from None


# ---------------------------------------------------------------------------
# What the policy produces
# ---------------------------------------------------------------------------


@final
class _FactoryToken:
    """Guards `TrainingPopulation`, as `_FACTORY_TOKEN` guards a `Checkpoint`."""

    __slots__ = ()


_FACTORY_TOKEN = _FactoryToken()


class TrainingPopulation:
    """The split, standardised, missingness-applied rows a stage draws from.

    `fitted_on_row_ids` is the point of the object, and it is a straight lift
    of `Checkpoint.trained_on_row_ids` (`DESIGN.md` §7.1). Standardisation
    fitted on the test split is the same shape of error as a checkpoint
    predicting rows it trained on, and it gets the same treatment: the claim
    carries the rows that make it checkable, and the check is *run* rather than
    trusted.

    `statistics` is a plain mapping of tensors fitted once by the declared
    preprocessing, and nothing else. That boundary was drawn by a second
    consumer: ReMixMatch's distribution alignment maintains a running average
    of the *model's predictions* over the unlabelled stream, and had this been
    shaped as "the thing that answers statistical questions" it would have
    grown a model-dependent, time-varying interior. A statistic of the run
    belongs to whatever declares it, not here.
    """

    __slots__ = (
        "_assignment",
        "_fitted_on_row_ids",
        "_rows",
        "_spec_digest",
        "_statistics",
    )

    def __init__(
        self,
        *,
        rows: XTYBatch,
        assignment: str,
        statistics: Mapping[str, Tensor],
        fitted_on_row_ids: Tensor,
        spec_digest: str,
        issued_by: object = None,
    ) -> None:
        if issued_by is not _FACTORY_TOKEN:
            raise ArtifactError(
                "a TrainingPopulation is built by the loading factory — it is "
                "what `xty2.training.loading.build_population` returns — and "
                "not directly. `fitted_on_row_ids` is computed from the rows "
                "the statistics were actually fitted on, so a direct call "
                "would let a caller assert it instead (DESIGN.md §7.1)."
            )
        if fitted_on_row_ids.ndim != 1 or fitted_on_row_ids.dtype != torch.long:
            raise ArtifactError(
                "fitted_on_row_ids must be a [M] long tensor, got shape "
                f"{tuple(fitted_on_row_ids.shape)} of {fitted_on_row_ids.dtype}"
            )
        self._rows = rows
        self._assignment = assignment
        self._statistics: Mapping[str, Tensor] = MappingProxyType(
            {name: value.detach().clone() for name, value in sorted(statistics.items())}
        )
        self._fitted_on_row_ids = fitted_on_row_ids.detach().clone()
        self._spec_digest = spec_digest

    @classmethod
    def _issue(
        cls,
        *,
        rows: XTYBatch,
        assignment: str,
        statistics: Mapping[str, Tensor],
        fitted_on_row_ids: Tensor,
        spec_digest: str,
    ) -> TrainingPopulation:
        """Construct one. Package-private: the loading factory calls it."""
        return cls(
            rows=rows,
            assignment=assignment,
            statistics=statistics,
            fitted_on_row_ids=fitted_on_row_ids,
            spec_digest=spec_digest,
            issued_by=_FACTORY_TOKEN,
        )

    @property
    def rows(self) -> XTYBatch:
        """The training rows, after the declared policy."""
        return self._rows

    @property
    def assignment(self) -> str:
        """Which `Dataset.assignments` entry these rows are."""
        return self._assignment

    @property
    def statistics(self) -> Mapping[str, Tensor]:
        """What the declared preprocessing fitted. Read-only."""
        return self._statistics

    @property
    def fitted_on_row_ids(self) -> Tensor:
        """The row ids those statistics were fitted on. Checked, not trusted."""
        return self._fitted_on_row_ids

    @property
    def spec_digest(self) -> str:
        """The digest of the `DataSpec` that produced this population."""
        return self._spec_digest

    @property
    def batch_size(self) -> int:
        """How many training rows there are."""
        return self._rows.batch_size

    def eligible(self, population: Rows) -> Tensor:
        """`[n]` positions of the rows in `population`, for a sampler quota."""
        mask = row_mask(population, self._rows)
        return torch.nonzero(mask, as_tuple=False).flatten()

    def column(self, feature: str, schema: Schema) -> Tensor:
        """`[N]` values this feature takes across the training population.

        The one read a view transform needs: SCARF's corruption draws each
        replacement from "the uniform distribution over the values that feature
        takes on across the training dataset".
        """
        return self._rows.x[:, schema.index_of(feature)]

    def __repr__(self) -> str:
        return (
            f"TrainingPopulation(assignment={self._assignment!r}, "
            f"rows={self.batch_size}, statistics={sorted(self._statistics)!r})"
        )


def describe_statistics(statistics: Mapping[str, Any]) -> str:
    """A stable one-line summary of what preprocessing fitted."""
    return ", ".join(sorted(statistics)) or "nothing"
