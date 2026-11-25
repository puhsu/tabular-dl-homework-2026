import pytest
import torch
from tabpackage.nn import NumericalFeatureEmbeddingsPack, TabPack

from ple_reference import reference_piecewise_linear_encoding
from tabpack import (
    PiecewiseLinearEmbeddingsPack,
    encoding_coefficients,
    piecewise_linear_encoding,
)
from train_tabpack import compute_bin_edges


def check_gradients(actual, expected, tensors):
    torch.testing.assert_close(actual, expected)
    derivative = torch.randn_like(expected)
    for a, b in zip(
        torch.autograd.grad(actual, tensors, derivative),
        torch.autograd.grad(expected, tensors, derivative),
        strict=True,
    ):
        torch.testing.assert_close(a, b)
        assert torch.isfinite(a).all()


@pytest.mark.parametrize("packed", [False, True])
def test_piecewise_linear_encoding(packed):
    edges = [torch.tensor([0.0, 8.0, 16.0]), torch.tensor([1.0, 3.0])]
    coefficients = encoding_coefficients(edges, 4)
    rows = torch.tensor([[-2.0, 0.0], [0.0, 1.0], [5.0, 2.0], [8.0, 3.0], [21.0, 4.0]])
    if packed:
        rows = torch.stack([rows, rows.flip(0)])
        edges = [edges, [e + 1 for e in edges]]
        coefficients = tuple(
            torch.stack(pair)
            for pair in zip(
                coefficients, encoding_coefficients(edges[1], 4), strict=True
            )
        )
    rows.requires_grad_()
    check_gradients(
        piecewise_linear_encoding(rows, *coefficients),
        reference_piecewise_linear_encoding(rows, edges, max_bins=4),
        (rows,),
    )


SMALL_EDGES = [torch.tensor([0.0, 3.0]), torch.tensor([1.0, 3.0, 7.0])]
LARGE_EDGES = [torch.tensor([0.0, 1.0, 3.0, 5.0, 7.0]), torch.tensor([1.0, 3.0])]
MODEL_EDGES = (SMALL_EDGES, SMALL_EDGES, LARGE_EDGES, LARGE_EDGES)


def test_reference_width():
    rows = torch.tensor([[1.5, 2.0]])
    single = reference_piecewise_linear_encoding(rows, SMALL_EDGES)
    torch.testing.assert_close(single, torch.tensor([[[0.0, 0.5], [0.5, 0.0]]]))
    packed = reference_piecewise_linear_encoding(
        rows.expand(2, -1, -1), [SMALL_EDGES, LARGE_EDGES]
    )
    assert packed.shape == (2, 1, 2, 4)
    torch.testing.assert_close(
        packed[0], reference_piecewise_linear_encoding(rows, SMALL_EDGES, max_bins=4)
    )


def make_pack(bin_edges=MODEL_EDGES, dims=(2, 4, 3, 3)):
    return PiecewiseLinearEmbeddingsPack(
        2,
        list(dims),
        pack_size=len(bin_edges),
        max_d_embedding=4,
        bin_edges=bin_edges,
        max_bins=4,
    )


@pytest.mark.parametrize("widths", [3, [2, 3]])
def test_tabpack_integration(widths):
    model = TabPack(
        n_numerical_features=2,
        categorical_cardinalities=[],
        pack_size=2,
        numerical_embeddings={
            "type": PiecewiseLinearEmbeddingsPack,
            "d_embedding": widths,
            "bin_edges": [SMALL_EDGES, LARGE_EDGES],
            "max_bins": 4,
        },
        d_out=1,
        n_blocks=1,
        d_block=8,
        dropout=0.0,
    )
    assert isinstance(model.num_module, NumericalFeatureEmbeddingsPack)
    assert model.num_module.get_output_shape() == torch.Size([2, 3])
    assert model(torch.randn(5, 2), None).shape == (2, 5, 1)


def test_coefficients():
    w, b, single = encoding_coefficients(SMALL_EDGES, 4)
    torch.testing.assert_close(
        w, torch.tensor([[0.0, 0.0, 0.0, 1 / 3], [0.5, 0.0, 0.0, 0.25]])
    )
    torch.testing.assert_close(
        b, torch.tensor([[0.0, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, -0.75]])
    )
    assert single.tolist() == [True, False]


def test_embedding_init():
    layer = make_pack()
    assert layer.encoding_weight.shape == (4, 2, 4, 4)
    assert layer.linear_weight.shape == layer.linear_bias.shape == (4, 2, 4)
    assert set(dict(layer.named_parameters())) == {
        "linear_weight",
        "linear_bias",
        "encoding_weight",
    }
    assert {"w", "b", "single_bin_mask", "output_mask"} <= set(
        dict(layer.named_buffers())
    )
    torch.testing.assert_close(
        layer.output_mask, torch.arange(4)[None] < torch.tensor([2, 4, 3, 3])[:, None]
    )
    assert torch.count_nonzero(layer.encoding_weight) == 0
    bound = torch.tensor([2.0, 4.0, 3.0, 3.0]).rsqrt()[:, None, None]
    for parameter in (layer.linear_weight, layer.linear_bias):
        assert (parameter.abs() <= bound).all()
        assert parameter.count_nonzero() > 0


def test_initial_embedding_is_linear():
    layer = make_pack()
    rows = torch.randn(4, 5, 2)
    expected = (
        rows[..., None] * layer.linear_weight[:, None] + layer.linear_bias[:, None]
    )
    torch.testing.assert_close(layer(rows), expected * layer.output_mask[:, None, None])


def test_ple_module():
    torch.manual_seed(4)
    layer = make_pack()
    ids = torch.tensor([3, 0, 3])
    rows = torch.randn(3, 8, 2, requires_grad=True)
    with torch.no_grad():
        layer.encoding_weight.normal_()  # Exercise the residual after its zero initialization.
        layer.linear_bias.normal_()
    outputs = []
    for batch, i in zip(rows, ids, strict=True):
        encoding = reference_piecewise_linear_encoding(
            batch, MODEL_EDGES[i], max_bins=layer.encoding_weight.shape[-2]
        )
        cols = []
        for f in range(batch.shape[1]):
            cols.append(
                batch[:, f, None] * layer.linear_weight[i, f]
                + layer.linear_bias[i, f]
                + encoding[:, f] @ layer.encoding_weight[i, f]
            )
        outputs.append(torch.stack(cols, 1) * layer.output_mask[i])
    check_gradients(
        layer(rows, ids),
        torch.stack(outputs),
        (rows, layer.encoding_weight, layer.linear_weight, layer.linear_bias),
    )


def test_ple_checkpoint():
    torch.manual_seed(5)
    model = make_pack()
    ids = torch.tensor([3, 0, 3])
    rows = torch.randn(3, 8, 2, requires_grad=True)
    actual = model(rows, ids)
    # Reconstruct just the low-bin member: its allocated width must not shrink.
    clone = make_pack([SMALL_EDGES], [2])
    clone.load_state_dict_at(
        model.state_dict(), pack_idx=torch.tensor([0]), state_dict_idx=ids[1:2]
    )
    torch.testing.assert_close(clone(rows[1:2]), actual[1:2])


def test_train_quantiles():
    x = torch.tensor([[0.0], [0.0], [0.0], [2.0], [4.0]])
    torch.testing.assert_close(compute_bin_edges(x, 4), [torch.tensor([0.0, 2.0, 4.0])])
    with pytest.raises(ValueError, match="bin edges"):
        compute_bin_edges(torch.ones_like(x), 4)
