# Tier 2 — card-defined validation

Each test invokes the reusable `xty2.evaluation.runner` for one recipe. The
runner parses that card's §6 `reproduction` block, enforces its seed count and
reviewed thresholds, writes all replicate values to `runs/tier2/<recipe>.json`,
and checks the fresh outcome against the card's recorded `reproduced` or
`deviating` status.

The nightly workflow derives its matrix from `xty2.evaluation.benchmarks.RECIPES`
and runs one file per recipe as a separate job, so a failure names the recipe and
the expensive cases can use the hosted runner's CPU cores. A count here would
rot every time a card acquires a module, which is why there is no longer one.

To regenerate a ledger row locally after a reviewed protocol change:

```bash
uv run --no-sync python -m xty2.evaluation.runner \
  --recipe tarnet --write-ledger
```

`XTY2_WRITE_TIER2_LEDGERS=1` gives the pytest adapter the same writeback
behaviour. Normal nightly runs leave the checkout unchanged and upload `runs/`
as the evidence artifact.
