# xty2

Composable semi-supervised causal learning for tabular data.

xty2 separates representation, parameterisation, objectives, data views, and
training order. Named methods are declarative recipes assembled from those
parts, so a result can be traced to a specific mechanic rather than a monolithic
model class.

Each recipe also has a reviewed spec card. The card records the source method,
equations, executable hyperparameters, deviations, unknowns, and a predeclared
reproduction target. Fast invariants and synthetic smoke fits run on every PR;
published-number benchmarks run nightly.

## Quick start

The first few rows of `example.csv` might look like:

```csv
x1,x2,t,y,t_observed
0.2,1.1,0,1.5,1
-0.4,0.3,1,2.2,1
0.8,-0.2,0,0.7,0
```

The file needs at least 100 rows to fill the default recipe batch size. Then
create and train a missing-treatment TARNet recipe with:

```python
import numpy as np
import torch

from xty2.core import Dataset, FeatureSpec, Schema, XTYBatch, compile
from xty2.recipes import tarnet_extension
from xty2.training import run_program

table = np.loadtxt("example.csv", delimiter=",", skiprows=1)
schema = Schema(
    features=(
        FeatureSpec("x1", "continuous"),
        FeatureSpec("x2", "continuous"),
    ),
    treatment_cardinality=2,
)
rows = XTYBatch(
    x=torch.as_tensor(table[:, :2], dtype=torch.float32),
    t=torch.as_tensor(table[:, 2], dtype=torch.long),
    y=torch.as_tensor(table[:, 3], dtype=torch.float32),
    t_observed=torch.as_tensor(table[:, 4], dtype=torch.bool),
    y_observed=torch.ones(len(table), dtype=torch.bool),
    row_id=torch.arange(len(table)),
)
dataset = Dataset(
    schema=schema,
    rows=rows,
    assignments={"fit": torch.arange(len(table))},
)

torch.manual_seed(0)
recipe = tarnet_extension(schema)
run = compile(recipe)
result = run_program(run, {"joint_fit": dataset}, seed=0)
checkpoint = result.stage("joint_fit").checkpoint
```

On a row where `t_observed` is `0`, `t` must still contain a valid placeholder
treatment; the observation mask is the only missingness indicator.

### Build a recipe from scratch

Recipes can also be assembled directly from components, objectives, and stages.
After writing and reviewing `docs/recipes/tiny_tarnet.md`, this small recipe can
reuse the `schema` and `rows` created above:

```python
from itertools import repeat

from xty2.components import MLPEncoder, TARNetHead
from xty2.core import (
    ComponentGraph,
    Constant,
    ExternalBatches,
    GradientClipping,
    OptimiserSpec,
    Recipe,
    Stage,
    WeightDecay,
    Weighted,
)
from xty2.objectives import ObservedOutcomeMSE

initialisation = "torch Linear default Kaiming-uniform"
torch.manual_seed(0)
recipe = Recipe(
    name="tiny_tarnet",
    schema=schema,
    system=ComponentGraph(
        [
            MLPEncoder(
                input_dim=schema.num_features,
                widths=(32,),
                activation="relu",
                normalisation="none",
                dropout=0.0,
                initialisation=initialisation,
            ),
            TARNetHead(
                representation_dim=32,
                num_treatments=schema.treatment_cardinality,
                outcome=schema.outcome,
                widths=(16,),
                activation="relu",
                normalisation="none",
                dropout=0.0,
                initialisation=initialisation,
                output_parameterisation="K means; fixed Gaussian scale=1.0",
            ),
        ]
    ),
    program=(
        Stage(
            name="fit",
            objectives=(
                Weighted(ObservedOutcomeMSE(), weight=1.0, reduction="population"),
            ),
            trainable=("mlp_encoder", "tarnet_head"),
            optimiser=OptimiserSpec(
                name="adam",
                lr=1e-3,
                weight_decay=WeightDecay.none(),
                lr_schedule=Constant(1.0),
                clipping=GradientClipping.none(),
            ),
            steps=100,
            sampler=ExternalBatches(),
        ),
    ),
    card="docs/recipes/tiny_tarnet.md",
)

result = run_program(compile(recipe), {"fit": repeat(rows, 100)}, seed=0)
```

## Documentation

Start at [`docs/README.md`](docs/README.md). It explains which document is
authoritative and which one to read for a given task. Recipe cards are indexed
at [`docs/RECIPES.md`](docs/RECIPES.md).

## Development

```bash
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev]"

uv run pytest tests/invariants tests/smoke
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict
```

Tier 0 is `tests/invariants/`, Tier 1 is `tests/smoke/`, and Tier 2 is
`tests/benchmarks/`. Directory-based markers are applied automatically.
