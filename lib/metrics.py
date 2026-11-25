"""Shared prediction semantics and higher-is-better selection scores."""

import math

import torch
from torch.nn import functional as F


def compute_metrics(y, prediction, task_type, *, output_type, metric):
    """Score original-unit regression predictions, classification logits or probabilities."""
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Non-finite predictions")
    expected = ("values",) if task_type == "regression" else ("logits", "probabilities")
    if output_type not in expected:
        raise ValueError(f"{task_type} expects output_type in {expected}")
    metrics = _compute_metrics(y, prediction, task_type, output_type)
    metrics["score"] = metrics[metric] * (
        -1 if metric in ("rmse", "mae", "cross-entropy") else 1
    )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Non-finite metrics")
    return metrics


def _compute_metrics(y, prediction, task_type, output_type):
    y, prediction = y.cpu(), prediction.cpu()
    if task_type == "regression":
        error = prediction - y
        return {
            "rmse": error.square().mean().sqrt().item(),
            "mae": error.abs().mean().item(),
            "r2": (1 - error.square().sum() / (y - y.mean()).square().sum()).item(),
        }
    logits = output_type == "logits"

    if task_type == "multiclass":
        return {
            "accuracy": (prediction.argmax(1) == y).float().mean().item(),
            "cross-entropy": (
                F.cross_entropy(prediction, y.long())
                if logits
                else F.nll_loss(prediction.clamp_min(1e-7).log(), y.long())
            ).item(),
        }
    loss = (
        F.binary_cross_entropy_with_logits(prediction, y.float())
        if logits
        else F.binary_cross_entropy(prediction.clamp(1e-7, 1 - 1e-7), y.float())
    )
    threshold = 0.0 if logits else 0.5
    metrics = {
        "accuracy": ((prediction >= threshold) == y).float().mean().item(),
        "cross-entropy": loss.item(),
    }
    # Rank-based AUC with average ranks for tied predictions.
    order = prediction.argsort()
    _, counts = torch.unique_consecutive(prediction[order], return_counts=True)
    ends = counts.cumsum(0)
    ranks = torch.repeat_interleave((ends - counts + 1 + ends).float() / 2, counts)
    positive = y[order] == 1
    n_positive = positive.sum().item()
    n_negative = len(y) - n_positive
    if n_positive and n_negative:
        metrics["roc-auc"] = (
            ranks[positive].sum().item() - n_positive * (n_positive + 1) / 2
        ) / (n_positive * n_negative)
    return metrics
