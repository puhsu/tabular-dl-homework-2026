"""Train one embedding type, following upstream examples/tabpack.py."""

import math
import random
import time
from functools import partial

import tabpackage
import tabpackage.ensemble
import tabpackage.metrics
import tabpackage.nn
import tabpackage.optim
import torch
from torch.nn import functional as F

from lib.data import build_data, load_info
from lib.metrics import compute_metrics
from tabpack import PiecewiseLinearEmbeddingsPack


def compute_bin_edges(train_x, n_bins):
    """Return a list of distinct training quantiles, one tensor per feature."""
    if not 1 < n_bins < len(train_x):
        raise ValueError("Need 1 < n_bins < number of training rows")
    quantile_levels = torch.linspace(
        0, 1, n_bins + 1, device=train_x.device, dtype=train_x.dtype
    )
    bin_edges = []
    for edges in torch.quantile(train_x, quantile_levels, dim=0).T:
        edges = edges.unique_consecutive()
        if len(edges) < 2 or not edges.isfinite().all():
            raise ValueError(
                "Each feature needs at least two distinct finite bin edges"
            )
        bin_edges.append(edges)
    return bin_edges


def make_pack_configs(train_x, embedding_type, n_models, seed):
    embedding_classes = {
        "cosine": tabpackage.nn.CosineEmbeddingsPack,
        "ple": PiecewiseLinearEmbeddingsPack,
    }
    shared_embedding_config: dict[str, type | int] = {
        "type": embedding_classes[embedding_type]
    }
    rng = random.Random(seed)

    def loguniform(low, high):
        return math.exp(rng.uniform(math.log(low), math.log(high)))

    if embedding_type == "ple":
        # A separate stream keeps backbone/optimizer draws paired with the cosine run.
        bin_rng = random.Random(seed + 1)
        counts = bin_rng.choices([8, 16, 32, 64, 128], k=n_models)
        shared_embedding_config["max_bins"] = max(counts)

    per_model_hyperparameters = []
    for i in range(n_models):
        embedding = {"d_embedding": rng.randint(8, 32)}
        init_scale = loguniform(0.01, 10.0)
        if embedding_type == "cosine":
            embedding["init_scale"] = init_scale
        else:
            embedding["bin_edges"] = [
                e.cpu() for e in compute_bin_edges(train_x, counts[i])
            ]
        per_model_hyperparameters.append(
            {
                "model": {
                    "numerical_embeddings": embedding,
                    "n_blocks": rng.randint(1, 3),
                    "dropout": rng.uniform(0.0, 0.5) if rng.randint(0, 1) else 0.0,
                },
                "optimizer": {
                    "adamw_lr": loguniform(0.0001, 0.005),
                    "weight_decay": loguniform(0.001, 1.0),
                    "muon_lr": loguniform(0.001, 0.1),
                },
            }
        )
    return shared_embedding_config, per_model_hyperparameters


def cross_entropy_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Average rows, not models: the trainer needs one loss per model.
    return F.cross_entropy(input.transpose(1, 2), target, reduction="none").mean(1)


def mse_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (input.squeeze(-1) - target).square().mean(1)


def regression_values(raw):
    return raw.squeeze(-1).float()


def cross_entropy_metric(*, y_true, y_pred):
    return tabpackage.metrics.multiclass_cross_entropy_pack(
        y_true=y_true, y_proba=y_pred
    )


def train(config):
    start = time.perf_counter()
    seed, device = config["seed"], config["device"]
    torch.set_num_threads(1)
    data = build_data(**config["data"], device=device, seed=seed)
    task = load_info(config["data"]["dataset_dir"])["task"]
    x = data["x_num"]
    y = {split: split_y.flatten() for split, split_y in data["y"].items()}
    train_y = y["train"]
    datasets = {split: {"x_num": x[split], "y": y[split]} for split in ("train", "val")}

    if task["type"] == "regression":
        y_mean, y_std = train_y.mean(), train_y.std().clamp_min(1e-8)
        for dataset in datasets.values():
            dataset["y"] = (dataset["y"] - y_mean) / y_std
        d_out = 1
        loss_fn = mse_loss
        prediction_transform = regression_values
        pack_metric = tabpackage.metrics.mse_pack
        output_type = "values"
    else:
        for dataset in datasets.values():
            dataset["y"] = dataset["y"].long()
        d_out = int(train_y.max()) + 1
        loss_fn = cross_entropy_loss
        prediction_transform = partial(torch.softmax, dim=-1, dtype=torch.float32)
        pack_metric = cross_entropy_metric
        output_type = "probabilities"

    embedding_type = config["embedding"]
    embedding, per_model_hyperparameters = make_pack_configs(
        x["train"], embedding_type, config["n_models"], seed
    )
    minimize = tabpackage.MetricDirection.MINIMIZE
    eval_criterion = tabpackage.EvalCriterionConfig(
        dataset="val", metric="loss", metric_direction=minimize
    )

    def ensemble_metric(*, y_true, y_pred):
        return pack_metric(y_true=y_true, y_pred=y_pred[None])[0]

    tabpack = tabpackage.train(
        datasets=datasets,
        model={
            "categorical_cardinalities": [],
            "numerical_embeddings": embedding,
            "d_block": 384,
            "d_out": d_out,
        },
        per_model_hyperparameters=per_model_hyperparameters,
        optimizer={"type": tabpackage.optim.MuonAdamWPack},
        loss_fn=loss_fn,
        prediction_transform=prediction_transform,
        batch_size=config["batch_size"],
        eval_batch_size=config["batch_size"],
        max_epochs=config["epochs"],
        early_stopping_patience=config["patience"],
        eval_metrics={"val": {"loss": pack_metric}},
        eval_criterion=eval_criterion,
        ensembles={
            "greedy": tabpackage.EnsembleConfig(
                algorithm={
                    "type": tabpackage.ensemble.greedy_ensemble,
                    "max_ensemble_size": config["ensemble_limit"],
                },
                online=True,
                prediction_transform=None,
                early_stopping_patience=config["patience"],
                algorithm_metric=pack_metric,
                algorithm_metric_direction=minimize,
                algorithm_dataset="val",
                algorithm_device=torch.device(device),
            )
        },
        ensemble_eval_metrics={"val": {"loss": ensemble_metric}},
        ensemble_eval_criterion=eval_criterion,
        stop_when_all_ensembles_are_finalized=True,
        seed=seed,
        device=device,
        aux_device="cpu",
        autocast_dtype=None,
    )

    @torch.inference_mode()
    def score(predictor, split):
        predictor.to(device).eval()
        batches = []
        for batch in x[split].split(config["batch_size"]):
            _, ensembles = predictor(batch, None)
            batches.append(ensembles["greedy"])
        prediction = torch.cat(batches)
        if output_type == "values":
            prediction = prediction * y_std + y_mean
        return compute_metrics(
            y[split],
            prediction,
            task["type"],
            output_type=output_type,
            metric=task["score"],
        )

    val_metrics = score(tabpack.make_predictor(ensemble_names=["greedy"]), "val")
    # Keep the feature transform, PLE edges and target scale in the same coordinates.
    # Only the selected models are fitted again, from scratch, using both splits.
    refit_data = {
        key: torch.cat([datasets["train"][key], datasets["val"][key]])
        for key in ("x_num", "y")
    }
    print(f"Refit on train + val: {len(refit_data['y'])} rows", flush=True)
    refitted = tabpackage.retrain(
        tabpack, ensemble_name="greedy", dataset=refit_data, seed=seed
    )
    test_metrics = score(refitted.make_predictor(), "test")
    member_ids = tabpack.load_ensemble_member_ids("greedy").tolist()
    report = {
        "function": "train_tabpack.train",
        "embedding": embedding_type,
        "metrics": {"val": val_metrics, "test": test_metrics},
        "refit_rows": len(refit_data["y"]),
        "members": member_ids,
        "member_steps": tabpack.load_ensemble_member_steps("greedy").tolist(),
        "time": time.perf_counter() - start,
    }
    return report
