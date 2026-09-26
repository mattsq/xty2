"""A short execution check for VIME's pretraining and frozen transfer."""

from dataclasses import replace

import torch
from xty2.core import Program, compile
from xty2.evaluation.benchmarks.common import (
    continuous_schema,
    training_dataset,
    two_cluster_population,
)
from xty2.recipes import vime
from xty2.training import run_program


def test_pretext_heads_train_and_the_encoder_is_frozen_in_joint_fit() -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        schema = continuous_schema(6)
        train = two_cluster_population(256, seed=42, row_offset=0)
        dataset = training_dataset(schema, train.batch)
        torch.manual_seed(42)
        recipe = vime(schema)
        pretrain, fit = recipe.program
        run = compile(
            replace(recipe, program=Program((pretrain, replace(fit, steps=4))))
        )
        initial = {
            name.removeprefix("_components."): value.detach().clone()
            for name, value in run.graph.named_parameters()
        }
        result = run_program(run, {"pretrain": dataset, "joint_fit": dataset}, seed=123)
        pretrained = result.stage("pretrain")
        downstream = result.stage("joint_fit")
        assert pretrained.steps == 1_280
        assert downstream.steps == 4
        assert {term.name for term in pretrained.records[0].terms} == {
            "mask_estimation_bce",
            "feature_reconstruction",
        }
        assert all(
            torch.isfinite(torch.tensor(record.total)) for record in pretrained.records
        )
        for component in ("mlp_encoder", "mask_estimator", "feature_estimator"):
            keys = [
                name
                for name in pretrained.checkpoint.parameters
                if name.startswith(component + ".")
            ]
            assert keys
            assert any(
                not torch.equal(initial[name], pretrained.checkpoint.parameters[name])
                for name in keys
            )
        assert all(
            torch.equal(
                value,
                pretrained.checkpoint.parameters[name.removeprefix("_components.")],
            )
            for name, value in run.graph.named_parameters()
            if name.startswith("_components.mlp_encoder.")
        )
        assert set(downstream.checkpoint.components) == {
            "tarnet_head",
            "categorical_propensity",
        }
    finally:
        torch.set_num_threads(previous_threads)
