"""BYOL card §6.3: all four arms on every declared seed, no direction gate."""

import json
import math

import pytest
from xty2.evaluation.byol_study import ARMS, study


@pytest.mark.parametrize("base", [42, 43, 44])
def test_paired_byol_study(base: int) -> None:
    metrics = study(base)
    for arm in ARMS:
        assert math.isfinite(metrics[f"{arm}_outcome_nll"])
        assert math.isfinite(metrics[f"{arm}_encoder_effective_rank"])
    print(json.dumps({"base": base, "metrics": metrics}, sort_keys=True))
