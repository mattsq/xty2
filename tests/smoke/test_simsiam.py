"""Card 6.3: all three declared Tier 1 seeds, no directional assertions."""

import json
import math

import pytest
from xty2.evaluation.simsiam_study import study


@pytest.mark.parametrize("base", [42, 142, 242])
def test_paired_simsiam_study(base: int) -> None:
    metrics = study(
        base,
        train_rows=512,
        test_rows=512,
        pretrain_steps=128,
        fit_steps=256,
        eval_batches=4,
    )
    assert all(math.isfinite(value) for value in metrics.values())
    print(json.dumps({"base": base, "metrics": metrics}, sort_keys=True))
