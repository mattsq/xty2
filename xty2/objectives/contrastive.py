"""Instance discrimination between two realisations of one embedding (§4.2).

This is SCARF's `L_cont`: embed a row and a corrupted copy of it, score every
anchor against every contrast row by cosine similarity, and charge the softmax
cross-entropy that puts each row's own copy on the diagonal.

It is the first objective in the repository whose per-row value depends on the
*other* rows of the batch, and three consequences of that are decisions rather
than details.

* **The eligible rows are both the anchors and the candidates.** `DESIGN.md`
  §4 hands an objective a `RowIndex` and says the term is a mean over it. For a
  contrastive loss that set does double duty, and taking negatives from outside
  it would be reading rows the objective is not entitled to by another route.
  So the similarity matrix is built over the eligible rows alone; every other
  position of the returned `[B]` per-row tensor is a structural zero that
  `reduce_rows` then drops.
* **The batch size is a hyperparameter of this loss.** The number of negatives
  is `n - 1`, so a term declared once means something different under a
  different `BatchSource`. xty2 has no loader (`DESIGN.md` §11.4), so this is
  recorded in the consuming card's §6 rather than enforced here.
* **Nothing is detached.** SCARF descends both branches — it is not BYOL or
  SimSiam, and there is no target side to stop. `detaches` is empty and the
  dead-trainable rule therefore reaches everything upstream of both
  realisations, which is what a stage training the encoder through this term
  needs.
* **The cosine is taken here, not assumed of the port.** `s_ij` is defined as a
  normalised inner product, so both sides are normalised inside `compute`. A
  producer that already emits unit vectors — `ProjectionHead` with
  `normalisation="row_l2"`, which is what SCARF's `g` does — is therefore
  idempotent under this term rather than required by it, and the loss is the
  same number either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.errors import LossError, PortContractError, Xty2Error, require_str
from xty2.core.graph import Realisation, State
from xty2.core.loss import LossTerm, TrainContext, reduce_rows
from xty2.core.ports import Port, port_spec
from xty2.core.rows import RowIndex, Rows, validate_population


@dataclass(frozen=True)
class InfoNCEContrastive:
    """`-log( exp(s_ii/tau) / ((1/n) sum_k exp(s_ik/tau)) )`, cosine `s`.

    The expression is SCARF's, transcribed in `docs/recipes/scarf.md` §3.1,
    including two choices that distinguish it from SimCLR's NT-Xent and that a
    reimplementation silently gets wrong: the similarity matrix is *cross-view*
    and one-directional — `s_ij` pairs anchor `i` with contrast `j`, and there
    is no second term contrasting the other way — and the normaliser includes
    the positive pair `k = i`. The `1/n` inside it is an additive `log n` on the
    value that contributes no gradient; it is kept so that the number this
    objective logs is the number the paper's expression evaluates to.

    Attributes:
        port: The tensor port both sides read. SCARF reads the pre-train head's
            output, not the encoder's.
        anchor: The realisation supplying the rows of the similarity matrix —
            SCARF's uncorrupted `z`.
        contrast: The realisation supplying its columns, one positive and
            `n - 1` negatives — SCARF's corrupted `z~`.
        temperature: `tau`. Binds `losses.temperature`, so it has no default
            (`DESIGN.md` §9.1).
        rows: The population this term is entitled to, and therefore also the
            set its negatives come from.
        name: Keys the per-objective log (§6.2).
    """

    port: Port
    anchor: Realisation
    contrast: Realisation
    temperature: float = REQUIRED
    rows: Rows = "all"
    name: str = "info_nce_contrastive"

    CARD_KEYS: ClassVar[dict[str, str]] = {"temperature": "losses.temperature"}

    def __post_init__(self) -> None:
        card_hyperparameters(self)
        if not require_str("contrastive objective name", self.name, error=LossError):
            raise LossError("InfoNCEContrastive.name must be non-empty")
        if not isinstance(self.port, Port):
            raise LossError(
                f"InfoNCEContrastive.port must be a Port, got {type(self.port)}"
            )
        if port_spec(self.port).kind != "tensor":
            raise LossError(
                f"InfoNCEContrastive scores an embedding, but port {self.port!s} "
                f"carries {port_spec(self.port).kind}. A contrastive loss over a "
                "distribution port waits for the card that needs one "
                "(DESIGN.md §11)."
            )
        anchor: object = self.anchor
        contrast: object = self.contrast
        if not isinstance(anchor, Realisation) or not isinstance(contrast, Realisation):
            raise LossError(
                "InfoNCEContrastive.anchor and contrast must be Realisations"
            )
        if self.anchor == self.contrast:
            raise LossError(
                f"InfoNCEContrastive contrasts {self.anchor} with itself. Every "
                "diagonal similarity would be exactly 1 and the task would be "
                "to tell a row from its own identical copy; a positive pair is "
                "two realisations of one row (SCARF §3)."
            )
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, int | float
        ):
            raise LossError(
                f"InfoNCEContrastive.temperature must be a number, got "
                f"{type(self.temperature)}"
            )
        if not float(self.temperature) > 0.0:
            raise LossError(
                f"InfoNCEContrastive.temperature divides the similarities and "
                f"must be positive, got {self.temperature!r}"
            )
        object.__setattr__(self, "temperature", float(self.temperature))
        try:
            validate_population(self.rows)
        except Xty2Error as error:
            raise LossError(f"InfoNCEContrastive {self.name!r}: {error}") from error

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.port, self.anchor), (self.port, self.contrast)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        """Nothing: SCARF descends both branches (module note)."""
        return frozenset()

    def plan_details(self) -> tuple[str, ...]:
        """Arithmetic the ports, rows and card keys do not already say.

        The first two lines name the two realisations *by role*, and that is
        load-bearing rather than decorative. `requires` is a set, so the plan
        renders the pair in a canonical order and cannot show which side is
        `s_ij`'s row; the loss is not symmetrised, so swapping them computes a
        different number. Without these lines an anchored-on-the-corrupted-row
        recipe printed a byte-identical plan and hashed to the same digest,
        which is exactly the provenance collision `DESIGN.md` §4 requires an
        objective to emit through `plan_details`.
        """
        return (
            f"anchor rows (s_ij row index) = {self.anchor}",
            f"contrast columns (s_ij column index) = {self.contrast}",
            "similarity = cosine(anchor row, contrast row)",
            "positive = the same row under the contrast realisation",
            "negatives = the other eligible rows, and only those",
            "denominator = (1/n) * sum over every eligible contrast row, "
            "including the positive",
        )

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del ctx
        anchor = self._embedding(state, self.anchor, batch)
        contrast = self._embedding(state, self.contrast, batch)
        if rows.numel() == 0:
            return LossTerm.empty(like=anchor)

        left = torch.nn.functional.normalize(anchor.index_select(0, rows), dim=-1)
        right = torch.nn.functional.normalize(contrast.index_select(0, rows), dim=-1)
        similarity = left @ right.transpose(0, 1) / float(self.temperature)
        eligible = int(rows.numel())
        # `-s_ii + logsumexp_k s_ik - log n` — the paper's expression in log
        # space. The `- log n` is the `1/N` inside its denominator: constant in
        # the parameters, and dropped by no-one because the logged value should
        # be the one the paper writes.
        positive = similarity.diagonal()
        per_eligible = (
            similarity.logsumexp(dim=-1)
            - positive
            - torch.log(torch.tensor(float(eligible), dtype=similarity.dtype))
        )
        per_row = torch.zeros(
            batch.batch_size, dtype=per_eligible.dtype, device=per_eligible.device
        ).index_copy(0, rows, per_eligible)
        return reduce_rows(
            per_row, rows, diagnostics=_alignment(similarity, self.temperature)
        )

    def _embedding(
        self, state: State, realisation: Realisation, batch: XTYBatch
    ) -> Tensor:
        value = state[realisation][self.port]
        if not isinstance(value, Tensor):
            raise PortContractError(
                f"objective {self.name!r} read port {str(self.port)!r} under "
                f"{realisation} as an embedding tensor, but it carries "
                f"{type(value)}. Its PortSpec is the contract (DESIGN.md §2)."
            )
        if value.shape[0] != batch.batch_size:
            raise LossError(
                f"InfoNCEContrastive {self.name!r} got {value.shape[0]} rows "
                f"from {realisation} for a batch of {batch.batch_size}"
            )
        return value


def _alignment(similarity: Tensor, temperature: float) -> dict[str, float]:
    """Whether the mechanism is working, rather than whether the loss fell.

    `alignment` is the mean cosine similarity of a row to its own contrast
    realisation and `uniformity` the mean to every other eligible row, both
    with the temperature divided back out so they are cosines and comparable
    across recipes. A collapsed representation — every embedding the same
    direction — shows up here as the two converging on each other, which no
    single loss number distinguishes: the collapsed loss is exactly `0` (every
    `s_ij` equal cancels against the `1/n`), and `0` is neither a floor nor a
    failure value. A discriminating representation scores *below* it, and an
    untrained network usually does too.

    With fewer than two eligible rows there is no off-diagonal to average, so
    `uniformity` is **omitted** rather than filled in. Reporting the diagonal
    under that name would emit `alignment == uniformity`, which is the signature
    this pair uses for collapse, on a batch that has told us nothing.
    """
    eligible = similarity.shape[0]
    scaled = similarity.detach() * temperature
    positive = float(scaled.diagonal().mean())
    if eligible < 2:
        return {"alignment": positive}
    off_diagonal = (float(scaled.sum()) - float(scaled.diagonal().sum())) / (
        eligible * (eligible - 1)
    )
    return {"alignment": positive, "uniformity": off_diagonal}


__all__ = ["InfoNCEContrastive"]
