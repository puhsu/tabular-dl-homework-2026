"""Evaluate frozen Kumo on validation (train context) and test (train + validation context)."""

import time

import torch

from kumo import KumoMedium
from kumo_cache import build_context_cache, predict_batch
from lib.data import build_data, load_info
from lib.metrics import compute_metrics


@torch.inference_mode()
def predict_queries(model, context, labels, query, batch_size):
    """Encode the labeled context once, then predict unlabeled rows in batches."""
    categorical = torch.zeros(context.shape[1], dtype=torch.bool, device=context.device)
    cache = build_context_cache(model, context, labels, categorical, batch_size)
    return torch.cat(
        [
            predict_batch(model, batch, categorical, cache, batch_size)
            for batch in query.split(batch_size)
        ]
    )


@torch.inference_mode()
def evaluate(config):
    start = time.perf_counter()
    torch.set_num_threads(1)
    task = load_info(config["data"]["dataset_dir"])["task"]
    data = build_data(**config["data"], device=config["device"], seed=config["seed"])
    x, y = data["x_num"], data["y"]
    regression = task["type"] == "regression"
    model = KumoMedium.load(
        config["checkpoint"], classification=not regression, device=config["device"]
    )

    metrics, context_sizes = {}, {}
    context, context_y = x["train"], y["train"].flatten()
    for split in ("val", "test"):
        if split == "test":
            context = torch.cat([context, x["val"]])
            context_y = torch.cat([context_y, y["val"].flatten()])
        context_sizes[split] = len(context)
        if regression:
            mean, scale = context_y.mean(), context_y.std().clamp_min(1e-8)
            labels = (context_y - mean) / scale
        else:
            labels = context_y.long()
        print(
            f"{split}: {len(labels)} context rows, {len(x[split])} queries", flush=True
        )
        raw = predict_queries(model, context, labels, x[split], config["batch_size"])
        prediction = (
            raw.mean(1) * scale + mean
            if regression
            else raw[:, : int(labels.max()) + 1]
        )
        metrics[split] = compute_metrics(
            y[split].flatten(),
            prediction,
            task["type"],
            output_type="values" if regression else "logits",
            metric=task["score"],
        )
    report = dict(
        function="train_kumo.evaluate",
        metrics=metrics,
        context=context_sizes,
        checkpoint=config["checkpoint"],
        time=time.perf_counter() - start,
    )
    return report
