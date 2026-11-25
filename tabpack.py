"""PLE Embeddings and their TabPack integration."""

import torch
from tabpackage.nn import BufferPack, NumericalFeatureEmbeddingsPack, ParameterPack


def piecewise_linear_encoding(x, w, b, single_bin_mask):
    """[..., B, F] -> [..., B, F, T]. Keep extrapolating tails (both for a single bin)."""
    raise NotImplementedError("TODO: piecewise_linear_encoding")


def encoding_coefficients(bin_edges, max_bins):
    """Prepare fixed interval coefficients from one 1-D edge tensor per feature.

    Return w and b of shape (n_features, max_bins) and a boolean
    single-bin mask of shape (n_features,). Store the last interval at -1.
    """

    raise NotImplementedError("TODO: encoding_coefficients")


class PiecewiseLinearEmbeddingsPack(NumericalFeatureEmbeddingsPack):
    def __init__(
        self,
        n_features,
        d_embedding,
        *,
        max_d_embedding=None,
        pack_size,
        bin_edges,
        max_bins,
    ):
        super().__init__(pack_size=pack_size)
        if isinstance(d_embedding, int):
            if max_d_embedding is not None and max_d_embedding != d_embedding:
                raise ValueError("max_d_embedding must equal a shared embedding width")
            d_embedding = [d_embedding] * pack_size
        if max_d_embedding is None:
            max_d_embedding = max(d_embedding)
        if len(d_embedding) != pack_size or not all(
            0 < d <= max_d_embedding for d in d_embedding
        ):
            raise ValueError(
                "Expected one positive embedding width per member within max_d_embedding"
            )

        # For each model, compute coefficients from its list bin_edges
        # Stack w, b and single_bin_mask in model order and register as BufferPack.
        # Initialize linear_weight, linear_bias, encoding_weight (ParameterPack)
        # and output_mask (BufferPack). Shapes and initialization: README, part 2.

        raise NotImplementedError("TODO: initialize PiecewiseLinearEmbeddingsPack")

    @property
    def pack_size(self):
        return self.linear_bias.shape[0]

    def get_output_shape(self) -> torch.Size:
        """Shape of one object: (n_features, max_d_embedding)."""
        return torch.Size(self.linear_bias.shape[1:])

    def forward(self, x, pack_idx=None):
        # Add the raw-value embedding and the encoding correction, then mask output channels.
        raise NotImplementedError("TODO: forward")
