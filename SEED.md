
Building on https://www.github.com/XTYLearner

Your registry was probably at the wrong level. cycle_dual, mean_teacher, cnflow, TARNet, and so on should remain available as named recipes, but each recipe should compile from smaller components, objectives and training stages.

For XTYLearner, I would separate five questions that monolithic model classes currently collapse:

1. What probabilistic quantities are represented?
2. How is each quantity parameterised?
3. Which losses train them?
4. Which data views and subsets does each loss use?
5. In what order are those losses and models trained?

1. Give every model a set of semantic outputs

The common currency should be named statistical quantities, rather than particular architectures:

class Port(str, Enum):
    X_REPR = "x_repr"
    XY_REPR = "xy_repr"
    Y_GIVEN_XT = "p(y|x,t)"
    T_GIVEN_X = "p(t|x)"
    T_GIVEN_XY = "q(t|x,y)"
    JOINT_ENERGY = "energy(x,t,y)"
    RECONSTRUCTION = "reconstruction"
    TEACHER_T_PROBS = "teacher_t_probs"

A component declares what it consumes and produces:

class Component(nn.Module):
    requires: set[Port]
    provides: set[Port]
    def forward(
        self,
        state: dict[Port, Any],
        batch: XTYBatch,
    ) -> dict[Port, Any]:
        ...

Examples:

* An MLP encoder consumes x and produces X_REPR.
* A TARNet head consumes X_REPR and produces Y_GIVEN_XT.
* A categorical propensity head produces T_GIVEN_X.
* A treatment-inference head consumes x,y and produces T_GIVEN_XY.
* A conditional flow consumes X_REPR and categorical t as context, then produces Y_GIVEN_XT.
* A joint EBM produces JOINT_ENERGY.

This lets a conditional flow outcome model use the same treatment prior, posterior head and semi-supervised losses as a Gaussian outcome model.

The outputs should be typed distribution-like objects:

class OutcomeDistribution(Protocol):
    def log_prob(self, y: Tensor) -> Tensor: ...
    def mean(self) -> Tensor: ...
    def sample(self, n: int) -> Tensor: ...
class TreatmentDistribution(Protocol):
    @property
    def probs(self) -> Tensor: ...
    def log_prob(self, t: Tensor) -> Tensor: ...

That fixes one of the problems you encountered in cnflow_model.py: t remains categorical context instead of being awkwardly included inside a continuous flow.

2. Make losses independent objects

A model should produce statistical quantities. It should not own the whole training objective.

class Objective(Protocol):
    requires: set[Port]
    def compute(
        self,
        state: dict[Port, Any],
        batch: XTYBatch,
        context: TrainContext,
    ) -> LossTerm:
        ...

Useful XTYLearner objectives would include:

ObservedOutcomeNLL
ObservedTreatmentNLL
MissingTreatmentMarginalNLL
PosteriorKLDivergence
MaskedFeatureReconstruction
VATConsistency
WeakStrongConsistency
EntropyMinimisation
PseudoLabelTreatmentLoss
CycleConsistency
RepresentationBalance
DragonNetTargetedLoss
OrdinalTreatmentLoss
TeacherStudentConsistency

Each objective also declares its row population:

ObservedTreatmentNLL(rows="t_observed")
MissingTreatmentMarginalNLL(rows="t_missing")
VATConsistency(rows="all")

This matters because your labeled and unlabeled observations are different views of the same X,T,Y problem, rather than entirely different datasets.

For four cruise settings, the missing-treatment likelihood can use exact marginalisation:

-\log \sum_{t=1}^{4}
p_\phi(y\mid x,t)\,p_\theta(t\mid x).

That objective only requires Y_GIVEN_XT and T_GIVEN_X. You could therefore use it with a TARNet outcome head, Gaussian density head, conditional flow or another compatible likelihood without rewriting the training loop.

If you want an amortised posterior q(t\mid x,y), that becomes another component and introduces objectives such as an ELBO or posterior-matching loss. It is no longer tangled into the outcome architecture.

3. Separate augmentation from the loss using named views

An augmentation should produce a view, with an explicit statement about what it preserves:

ViewSpec(
    name="strong_x",
    transforms=[
        FeatureMask(p=0.25),
        BoundedJitter(columns=["KTAS", "TRQ", "FF"]),
    ],
    preserves={"t", "y"},
)

A consistency loss then requests particular views:

ConsistencyLoss(
    prediction=Port.T_GIVEN_X,
    left_view="weak_x",
    right_view="strong_x",
)

For tabular and flight data, I would attach metadata to every feature:

FeatureSpec(
    name="MASS",
    kind="continuous",
    bounds=(...),
    perturbation_scale=...,
    mutable=True,
)

That prevents generic augmentation code from creating impossible combinations. You may also want conditional view generators, such as perturbing mass while recomputing derived quantities, rather than independently jittering every column.

4. Add a first-class loss mixer

Each objective should return its unweighted value and diagnostics. A separate mixer decides how objectives interact:

LossMixer([
    Weighted(ObservedOutcomeNLL(), weight=1.0),
    Weighted(ObservedTreatmentNLL(), weight=1.0),
    Weighted(
        MissingTreatmentMarginalNLL(),
        weight=Ramp(start=0.0, end=0.5, steps=5_000),
    ),
    Weighted(
        VATConsistency(),
        weight=Ramp(start=0.0, end=0.2, steps=10_000),
    ),
])

This gives you one place to support:

* Fixed weighting
* Warm-up and ramp schedules
* Alternating objectives
* Objective-specific update frequencies
* Loss normalisation
* Later, methods such as GradNorm or PCGrad

Log the following for every objective:

* Raw and weighted loss
* Number of eligible observations
* Gradient norm on shared parameters
* Pairwise gradient cosine similarity
* Pseudo-label coverage and calibration

Once you combine many objectives, total validation performance does not tell you whether one term has become numerically irrelevant or is fighting another.

5. Represent sequencing as a training program

Some procedures cannot be expressed cleanly as one weighted loss. Pseudo-labelling, distillation and targeted refitting are stage transitions.

program = Program([
    Stage(
        name="representation_pretraining",
        data="all",
        trainable=["x_encoder"],
        objectives=[
            MaskedFeatureReconstruction(),
            WeakStrongConsistency(),
        ],
    ),
    Stage(
        name="joint_xty",
        initialise_from="representation_pretraining",
        trainable=["x_encoder", "outcome", "propensity", "posterior"],
        objectives=[
            ObservedOutcomeNLL(),
            ObservedTreatmentNLL(),
            MissingTreatmentMarginalNLL(),
            VATConsistency(),
        ],
    ),
    Stage(
        name="build_teacher",
        initialise_from="joint_xty",
        action=FitEMATeacher(),
    ),
    Stage(
        name="pseudo_label",
        action=CreateSoftTreatmentLabels(
            teacher="build_teacher",
            threshold=0.90,
        ),
    ),
    Stage(
        name="refit",
        initialise_from="joint_xty",
        inputs=["pseudo_label"],
        objectives=[
            ObservedOutcomeNLL(),
            SoftTreatmentNLL(),
            TeacherStudentConsistency(),
        ],
    ),
    Stage(
        name="targeted_causal_fit",
        initialise_from="refit",
        executor="cross_fit",
        objectives=[
            DragonNetTargetedLoss(),
        ],
    ),
])

Stages should produce immutable artifacts such as checkpoints, teacher models and pseudo-label tables. Do not mutate the original dataset in place.

This structure handles Beyer-style recipes properly. Some objectives run jointly, others are introduced late, and pseudo-label generation sits between training phases.

6. Keep “models” as recipes

The user-facing registry can remain:

learner = xtylearner.create("cycle_vat", ...)

But cycle_vat now expands into:

Recipe(
    system=ComponentGraph([...]),
    program=Program([...]),
    evaluator=EvaluationSuite([...]),
)

You would have three registry levels:

Registry	Examples
Components	mlp_encoder, tarnet_head, conditional_flow, categorical_posterior
Objectives and transforms	vat, marginal_t_nll, masked_features, cycle_loss
Recipes	tarnet, cnflow, cycle_dual, mean_teacher, s4l_xty

Recipes provide convenient defaults and reproducibility. Components provide experimentation.

I would resolve the registry once when compiling the configuration. The rest of the code should work with ordinary Python objects rather than repeatedly looking things up by string.

Suggested package structure

xtylearner/
    core/
        batch.py
        schema.py
        ports.py
        distributions.py
        graph.py
    components/
        encoders/
        outcome/
        treatment/
        posterior/
        density/
        energy/
    views/
        masking.py
        tabular.py
        perturbations.py
    objectives/
        supervised.py
        marginal.py
        consistency.py
        generative.py
        causal.py
        balancing.py
    training/
        stage.py
        program.py
        loss_mixer.py
        schedules.py
        executors.py
        artifacts.py
    recipes/
        tarnet.py
        cnflow.py
        cycle_dual.py
        mean_teacher.py
        s4l_xty.py
    evaluation/
        predictive.py
        causal.py
        calibration.py
        policy.py
    estimators/
        cate.py
        dml.py
        policy.py

Your array-based models should remain supported as explicit stage executors:

executor="gradient"
executor="array_fit"
executor="cross_fit"

That is cleaner than the current inference that a model with fit() but no loss() must use ArrayTrainer.

One causal-specific guardrail

Keep these quantities distinct:

* p(t\mid x): the treatment-assignment or propensity model
* q(t\mid x,y): a posterior used to infer missing treatments
* p(y\mid x,t): the outcome model

Using q(t\mid x,y) to create treatment pseudo-labels and then fitting p(y\mid x,t) on the same observations can create a circular fit. It is coherent inside a joint likelihood or ELBO. As a staged pseudo-labelling procedure, it generally needs out-of-fold predictions or another leakage control.

I would therefore make provenance part of every generated artifact:

PseudoLabels(
    source_stage="joint_xty",
    used_y=True,
    prediction_mode="out_of_fold",
    fold_ids=...,
)

The program compiler can reject unsafe combinations for causal estimation while permitting them for purely predictive experiments.

How I would migrate XTYLearner

I would avoid rewriting all 15-plus model families.

1. Introduce XTYBatch, semantic ports and distribution protocols.
2. Refactor one simple discriminative model, probably TARNet or MultiTask.
3. Refactor cnflow_model to test categorical context, likelihoods and exact missing-t marginalisation.
4. Refactor Mean Teacher or CycleVAT to test views, teachers and multi-loss training.
5. Adapt SSDML to test array and cross-fitting stages.
6. Express the old registry entries as recipes around those implementations.
7. Migrate the remaining models only when you next use them.

Those four examples exercise most of the required abstraction boundaries. If they compose cleanly, you have enough framework. I would resist building a completely general neural-programming DAG until a real fifth model requires it.