"""BYOL card §6.3: all five arms on every declared seed, no direction gate."""

import json
import math

import pytest
from xty2.evaluation.byol_study import ARMS, study

PRETRAINED = tuple(arm for arm in ARMS if arm != "no_pretrain")


@pytest.mark.parametrize("base", [42, 43, 44])
def test_paired_byol_study(base: int) -> None:
    metrics = study(base)
    for arm in ARMS:
        assert math.isfinite(metrics[f"{arm}_outcome_nll"])
        assert math.isfinite(metrics[f"{arm}_encoder_effective_rank"])
    for arm in PRETRAINED:
        alignment = metrics[f"{arm}_view_alignment"]
        assert math.isfinite(alignment)
        assert -1.0 <= alignment <= 1.0
        # §6.4's `A` and §3.1's distance are two readings of the same two
        # normalised vectors, so `A = 1 - residual/4` holds exactly whenever
        # both clear the squared-norm floor — which the recorded BN
        # variance/epsilon diagnostics show they do by five orders of
        # magnitude. This is a wiring tie, not an independent oracle: it binds
        # `A` to the pair the distance reads, so an `A` taken against the
        # online projection instead of the teacher's breaks it. The oracle for
        # the statistic itself is Tier 0's `scalar_view_alignment`.
        assert alignment == pytest.approx(
            1.0 - metrics[f"{arm}_predictor_residual"] / 4.0, abs=1e-6
        )
    # Deviation 7's control differs from the full arm only in target decay, so
    # a run in which they are bit-identical is a binding failure, not a null.
    assert metrics["source_ema_target_online_lag"] != metrics["full_target_online_lag"]
    assert metrics["zero_decay_target_online_lag"] == 0.0
    print(json.dumps({"base": base, "metrics": metrics}, sort_keys=True))
