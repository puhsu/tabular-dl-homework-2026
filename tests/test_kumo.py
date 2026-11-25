"""Golden probes need no download. Set KUMO_REG / KUMO_CLS to test actual checkpoints."""

import os
from pathlib import Path

import pytest
import torch

from kumo import KumoMedium, groups
from train_kumo import predict_queries


@pytest.fixture(scope="module")
def fixture():
    return torch.load(Path(__file__).with_name("medium_probes.pt"), weights_only=True)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(97)
    return KumoMedium().eval()


def test_group_wraparound():
    assert groups(4).tolist() == [[0, 1, 3], [1, 2, 0], [2, 3, 1], [3, 0, 2]]


@pytest.mark.parametrize("stage", ["cells", "encoder", "icl"])
@torch.inference_mode()
def test_stages(model, fixture, stage):
    x, y, cat = (fixture[k] for k in ("x", "y", "cat"))
    if stage == "cells":
        actual = model.row_embedding.cell_embedding(x, len(y), cat)
    elif stage == "encoder":
        actual = model.encoder(x, y, cat)
    else:
        actual = model.icl(fixture["encoder"], y)  # independent of your encoder
    torch.testing.assert_close(actual, fixture[stage], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("name", ["reg", "cls"])
@torch.inference_mode()
def test_released_checkpoint(fixture, name):
    path = os.environ.get("KUMO_" + name.upper())
    if not path:
        pytest.skip("Set KUMO_" + name.upper() + " to the released Medium checkpoint")
    x, y, cat = (fixture[k] for k in ("x", "y", "cat"))
    binary = name == "cls"
    y = (y > 0).long() if binary else y
    model = KumoMedium.load(path, classification=binary)
    actual = model(x, y, cat)
    torch.testing.assert_close(actual, fixture[name], atol=1e-4, rtol=1e-4)

    # The runner caches all labeled rows and reuses them across query batches.
    numerical_only = model(x, y, torch.zeros_like(cat))
    cached = predict_queries(model, x[: len(y)], y, x[len(y) :], batch_size=2)
    torch.testing.assert_close(cached, numerical_only, atol=1e-4, rtol=1e-4)
