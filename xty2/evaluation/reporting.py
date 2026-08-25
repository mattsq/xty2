"""Card parsing, replicate summaries and Tier 2 result-ledger writeback."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

Relation = Literal["<=", ">=", "between", "info"]
CardStatus = Literal[
    "draft",
    "reviewed",
    "implemented",
    "smoke-passing",
    "reproduced",
    "deviating",
]

_YAML_BLOCK = re.compile(r"```yaml\n(?P<body>.*?)\n```", re.DOTALL)
_STATUS = re.compile(
    r"(?m)^\*\*Status:\*\* `(?P<status>draft|reviewed|implemented|"
    r"smoke-passing|reproduced|deviating)`$"
)
_LEDGER = re.compile(
    r"(?P<head>### 6\.\d+ Result ledger\n\n"
    # The template writes the header as "Value ± stderr" and the cards
    # written before P12 write it as "Value +/- stderr". Both spellings name
    # the same reviewed column, so a card created from the template must not
    # be rejected as having no ledger.
    r"\| Date \| Commit \| Metric \| Value (?:\+/-|±) stderr \| "
    r"Within tolerance\? \|\n"
    r"\|---\|---\|---\|---\|---\|\n)"
    r"(?P<rows>(?:\|.*\|\n)+)"
)
_TIER2_EXPLANATION = re.compile(
    r"\n### Tier 2 outcome\n\n.*?(?=\n## 6\. Reproduction target)", re.DOTALL
)


@dataclass(frozen=True)
class ReproductionSpec:
    """The scalar controls parsed from one card's section 6 YAML block."""

    recipe: str
    card: Path
    values: Mapping[str, str]
    seed_count: int
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "card", Path(self.card))
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        if not self.recipe or not self.recipe.isidentifier():
            raise ValueError(f"invalid recipe name {self.recipe!r}")
        # Every Tier 2 row is reported as a sample mean plus a sample
        # standard error, and `mean_and_stderr` needs two replicates to have
        # one. Rejecting a one-seed card here names the card, rather than
        # failing deep inside aggregation after the whole benchmark has run.
        if self.seed_count < 2:
            raise ValueError(
                f"{self.recipe} reproduction seed count must be at least two "
                f"so that a sample stderr exists, got {self.seed_count}"
            )

    @property
    def primary_metric(self) -> str:
        """The card's ``metric`` or ``primary_metric`` scalar."""
        try:
            return self.values["metric"]
        except KeyError:
            try:
                return self.values["primary_metric"]
            except KeyError:
                raise ValueError(
                    f"{self.card} reproduction block names neither metric nor "
                    "primary_metric"
                ) from None

    def require(self, key: str, expected: str | None = None) -> str:
        """Read one required scalar and optionally assert its reviewed value."""
        try:
            value = self.values[key]
        except KeyError:
            raise ValueError(
                f"{self.card} reproduction block is missing required key {key!r}"
            ) from None
        if expected is not None and value != expected:
            raise ValueError(
                f"{self.card} has reproduction.{key}={value!r}; the benchmark "
                f"implements the reviewed value {expected!r}. Amend and review "
                "the card before changing the runner."
            )
        return value

    def bind(
        self,
        implemented: Mapping[str, str],
        *,
        documentation: Iterable[str] = (),
    ) -> None:
        """Account for *every* scalar the card's section-6 block declares.

        ``require`` binds one key; ``bind`` binds the block. Each declared
        scalar is either implemented at a named reviewed value or listed in
        ``documentation`` as prose no code reads. Binding only a subset is
        what makes Tier 2 provenance forgeable: the artifact carries
        ``spec.digest``, the digest of the whole block, so a card whose
        ``split`` or ``variant`` is amended without a matching benchmark
        change would still run the old protocol and stamp the new digest on
        the evidence. An unaccounted key stops the run instead.
        """
        expected = dict(implemented)
        inert = set(documentation)
        overlap = sorted(inert & set(expected))
        if overlap:
            raise ValueError(
                f"{self.card} benchmark binds {overlap!r} as both implemented "
                "and documentation; a key is one or the other"
            )
        declared = set(self.values)
        unbound = sorted(declared - set(expected) - inert)
        if unbound:
            raise ValueError(
                f"{self.card} reproduction block declares {unbound!r}, which "
                "this benchmark neither implements nor records as "
                "documentation. Implement the reviewed control or declare it "
                "inert before running Tier 2; otherwise the artifact would "
                "carry this block's digest for a protocol it did not execute."
            )
        absent = sorted((set(expected) | inert) - declared)
        if absent:
            raise ValueError(
                f"{self.card} reproduction block no longer declares {absent!r}, "
                "which this benchmark binds. Amend and review the card before "
                "changing the runner."
            )
        for key, value in expected.items():
            self.require(key, value)


@dataclass(frozen=True)
class MetricResult:
    """A replicate vector, its mean/stderr and an optional pass threshold."""

    name: str
    values: tuple[float, ...]
    relation: Relation = "info"
    target: float | tuple[float, float] | None = None
    unit: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))
        if not self.name:
            raise ValueError("a benchmark metric needs a non-empty name")
        if not self.values:
            raise ValueError(f"benchmark metric {self.name!r} has no replicates")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError(f"benchmark metric {self.name!r} must be finite")
        if self.relation not in ("<=", ">=", "between", "info"):
            raise ValueError(
                f"metric {self.name!r} has unknown relation {self.relation!r}"
            )
        if self.relation == "info" and self.target is not None:
            raise ValueError(
                f"informational metric {self.name!r} cannot carry a target"
            )
        if self.relation in ("<=", ">=") and (
            not isinstance(self.target, int | float)
            or isinstance(self.target, bool)
            or not math.isfinite(float(self.target))
        ):
            raise ValueError(f"required metric {self.name!r} needs a finite target")
        if self.relation == "between" and (
            not isinstance(self.target, tuple)
            or not all(math.isfinite(value) for value in self.target)
            or self.target[0] > self.target[1]
        ):
            raise ValueError(
                f"interval metric {self.name!r} needs finite (low, high), "
                f"got {self.target!r}"
            )

    @classmethod
    def upper_bound(
        cls,
        name: str,
        values: Iterable[float],
        target: float,
        *,
        unit: str = "",
    ) -> MetricResult:
        return cls(name, tuple(values), "<=", target, unit)

    @classmethod
    def lower_bound(
        cls,
        name: str,
        values: Iterable[float],
        target: float,
        *,
        unit: str = "",
    ) -> MetricResult:
        return cls(name, tuple(values), ">=", target, unit)

    @classmethod
    def information(
        cls, name: str, values: Iterable[float], *, unit: str = ""
    ) -> MetricResult:
        return cls(name, tuple(values), "info", None, unit)

    @classmethod
    def interval(
        cls,
        name: str,
        values: Iterable[float],
        low: float,
        high: float,
        *,
        unit: str = "",
    ) -> MetricResult:
        return cls(name, tuple(values), "between", (low, high), unit)

    @property
    def mean(self) -> float:
        return mean_and_stderr(self.values)[0]

    @property
    def stderr(self) -> float:
        return mean_and_stderr(self.values)[1]

    @property
    def passed(self) -> bool | None:
        if self.relation == "info":
            return None
        assert self.target is not None
        if self.relation == "between":
            assert isinstance(self.target, tuple)
            return self.target[0] <= self.mean <= self.target[1]
        assert isinstance(self.target, int | float)
        if self.relation == "<=":
            return self.mean <= self.target
        return self.mean >= self.target

    @property
    def criterion(self) -> str:
        if self.relation == "info":
            return "informational"
        assert self.target is not None
        suffix = f" {self.unit}" if self.unit else ""
        if self.relation == "between":
            assert isinstance(self.target, tuple)
            return f"{self.target[0]:g} <= mean <= {self.target[1]:g}{suffix}"
        assert isinstance(self.target, int | float)
        return f"mean {self.relation} {self.target:g}{suffix}"

    def summary(self) -> str:
        suffix = f" {self.unit}" if self.unit else ""
        return f"{self.mean:.6g} +/- {self.stderr:.3g}{suffix}"

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values": list(self.values),
            "mean": self.mean,
            "stderr": self.stderr,
            "unit": self.unit,
            "relation": self.relation,
            "target": (
                list(self.target) if isinstance(self.target, tuple) else self.target
            ),
            "criterion": self.criterion,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """One complete recipe benchmark and the evidence needed by its card."""

    recipe: str
    commit: str
    date: str
    spec_digest: str
    metrics: tuple[MetricResult, ...]
    interpretation: str
    protocol_deviation: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if not self.recipe or not self.recipe.isidentifier():
            raise ValueError(f"invalid benchmark recipe {self.recipe!r}")
        if not self.commit:
            raise ValueError("a benchmark result must name its source commit")
        try:
            Date.fromisoformat(self.date)
        except ValueError:
            raise ValueError(
                f"benchmark date must be ISO YYYY-MM-DD, got {self.date!r}"
            ) from None
        if not self.spec_digest:
            raise ValueError("a benchmark result must carry the card-spec digest")
        if not self.metrics:
            raise ValueError("a benchmark result needs at least one metric")
        names = [metric.name for metric in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate benchmark metric names {names!r}")
        counts = {len(metric.values) for metric in self.metrics}
        if len(counts) != 1:
            raise ValueError(
                f"all benchmark metrics must cover the same replicates, got {counts!r}"
            )
        if not any(metric.relation != "info" for metric in self.metrics):
            raise ValueError("a benchmark result needs at least one required metric")
        if not self.interpretation.strip():
            raise ValueError("a benchmark result needs an interpretation")

    @property
    def replicates(self) -> int:
        return len(self.metrics[0].values)

    @property
    def target_passed(self) -> bool:
        required = [
            metric.passed for metric in self.metrics if metric.relation != "info"
        ]
        return all(passed is True for passed in required)

    @property
    def status(self) -> Literal["reproduced", "deviating"]:
        if self.protocol_deviation is not None:
            return "deviating"
        return "reproduced" if self.target_passed else "deviating"

    @property
    def explanation(self) -> str:
        parts = [self.interpretation.strip()]
        if self.protocol_deviation is not None:
            parts.append(self.protocol_deviation.strip())
        failed = [
            f"{metric.name} was {metric.summary()} against {metric.criterion}"
            for metric in self.metrics
            if metric.passed is False
        ]
        if failed:
            parts.append("Failed target(s): " + "; ".join(failed) + ".")
        return " ".join(parts)

    def metric(self, name: str) -> MetricResult:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(
            f"benchmark {self.recipe!r} has no metric {name!r}; it has "
            f"{[metric.name for metric in self.metrics]!r}"
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe,
            "commit": self.commit,
            "date": self.date,
            "spec_digest": self.spec_digest,
            "replicates": self.replicates,
            "status": self.status,
            "target_passed": self.target_passed,
            "protocol_deviation": self.protocol_deviation,
            "interpretation": self.interpretation,
            "explanation": self.explanation,
            "metrics": [metric.as_json() for metric in self.metrics],
        }

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def mean_and_stderr(values: Iterable[float]) -> tuple[float, float]:
    """Sample mean and standard error, requiring at least two replicates."""
    resolved = tuple(float(value) for value in values)
    if len(resolved) < 2:
        raise ValueError(
            f"mean_and_stderr needs at least two replicates, got {len(resolved)}"
        )
    if not all(math.isfinite(value) for value in resolved):
        raise ValueError("replicate values must be finite")
    mean = math.fsum(resolved) / len(resolved)
    variance = math.fsum((value - mean) ** 2 for value in resolved) / (
        len(resolved) - 1
    )
    return mean, math.sqrt(variance / len(resolved))


def load_reproduction_spec(
    card: Path, *, recipe: str | None = None
) -> ReproductionSpec:
    """Parse the reviewed scalar controls from card section 6."""
    card = Path(card)
    text = card.read_text(encoding="utf-8")
    blocks = [
        match.group("body")
        for match in _YAML_BLOCK.finditer(text)
        if match.group("body").startswith("reproduction:\n")
    ]
    if len(blocks) != 1:
        raise ValueError(
            f"{card} must contain exactly one section-6 reproduction YAML "
            f"block, found {len(blocks)}"
        )
    block = blocks[0]
    values: dict[str, str] = {}
    for line in block.splitlines()[1:]:
        match = re.fullmatch(r"  ([a-z][a-z0-9_]*):(?:\s*(.*))?", line)
        if match is None:
            continue
        value = _strip_yaml_scalar(match.group(2) or "")
        if value:
            values[match.group(1)] = value
    for key in ("dataset", "tolerance", "seeds", "report"):
        if key not in values:
            raise ValueError(f"{card} reproduction block is missing {key!r}")
    seed_count = _seed_count(values["seeds"], card)
    resolved_recipe = recipe or card.stem
    spec = ReproductionSpec(
        recipe=resolved_recipe,
        card=card,
        values=values,
        seed_count=seed_count,
        digest=hashlib.sha256((block + "\n").encode()).hexdigest(),
    )
    _ = spec.primary_metric
    return spec


def card_status(text: str) -> CardStatus:
    """Read the one status declaration from a card."""
    matches = list(_STATUS.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            f"a card must contain exactly one status declaration, found {len(matches)}"
        )
    return matches[0].group("status")  # type: ignore[return-value]


def assert_result_matches_card(result: BenchmarkResult, card: Path) -> None:
    """Require the fresh Tier 2 outcome to agree with the recorded card state."""
    text = Path(card).read_text(encoding="utf-8")
    status = card_status(text)
    if status not in ("reproduced", "deviating"):
        raise AssertionError(
            f"{card} is {status!r}; a Tier 2 run must be recorded as "
            "'reproduced' or 'deviating' before the nightly suite can pass"
        )
    if status != result.status:
        raise AssertionError(
            f"{card} records status {status!r}, but the fresh benchmark is "
            f"{result.status!r}: {result.explanation}"
        )
    if status == "deviating" and "### Tier 2 outcome" not in text:
        raise AssertionError(
            f"{card} records 'deviating' without the written section-5 Tier 2 "
            "explanation required by FIDELITY.md §1.1"
        )


def update_card_text(text: str, result: BenchmarkResult) -> str:
    """Return card Markdown with status, ledger row and deviation explanation."""
    _ = card_status(text)
    text = _STATUS.sub(f"**Status:** `{result.status}`", text, count=1)
    ledger_match = _LEDGER.search(text)
    if ledger_match is None:
        raise ValueError("card has no canonical section-6 result ledger table")
    required = [metric for metric in result.metrics if metric.relation != "info"]
    metric_names = "<br>".join(metric.name for metric in required)
    values = "<br>".join(metric.summary() for metric in required)
    within = (
        "yes" if result.target_passed and result.protocol_deviation is None else "no"
    )
    row = (
        f"| {result.date} | `{result.commit}` | {metric_names} | {values} | "
        f"{within} |\n"
    )
    old_rows = ledger_match.group("rows")
    blank = "| | | | | |\n"
    new_rows = old_rows.replace(blank, row, 1) if blank in old_rows else old_rows + row
    text = (
        text[: ledger_match.start()]
        + ledger_match.group("head")
        + new_rows
        + text[ledger_match.end() :]
    )
    text = _TIER2_EXPLANATION.sub("", text)
    if result.status == "deviating":
        explanation = (
            "\n### Tier 2 outcome\n\n"
            f"On {result.date}, commit `{result.commit}` produced a `deviating` "
            f"result: {result.explanation}\n"
        )
        marker = "\n## 6. Reproduction target"
        if marker not in text:
            raise ValueError("card has no section 6 heading for Tier 2 explanation")
        text = text.replace(marker, explanation + marker, 1)
    return text


def update_card(card: Path, result: BenchmarkResult) -> Path:
    """Write ``update_card_text`` back to ``card``."""
    card = Path(card)
    card.write_text(
        update_card_text(card.read_text(encoding="utf-8"), result),
        encoding="utf-8",
    )
    return card


def _strip_yaml_scalar(value: str) -> str:
    value = value.split("  #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _seed_count(value: str, card: Path) -> int:
    if value.isdigit():
        count = int(value)
        if count > 0:
            return count
    match = re.search(r"(?:^|\b)r\s*=\s*(\d+)\.\.(\d+)(?:\b|$)", value)
    if match is not None:
        first, last = (int(item) for item in match.groups())
        if last >= first:
            return last - first + 1
    raise ValueError(
        f"{card} has unsupported reproduction.seeds={value!r}; use a positive "
        "integer or an inclusive r=FIRST..LAST range"
    )


__all__ = [
    "BenchmarkResult",
    "MetricResult",
    "ReproductionSpec",
    "assert_result_matches_card",
    "card_status",
    "load_reproduction_spec",
    "mean_and_stderr",
    "update_card",
    "update_card_text",
]
