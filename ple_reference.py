"""Reference piecewise-linear encoding from the original (unpadded) bin edges."""

import torch


def reference_piecewise_linear_encoding(x_num, edges, *, max_bins=None):
    """Encode each numerical feature independently using its fitted bin edges.

    Args:
        x_num (Tensor): Batch of numerical inputs, optionally one batch per model.
        edges (sequence): For one model, a list of ``F`` strictly increasing 1-D
            tensors. If feature ``f`` has ``n`` bins, ``edges[f]`` has shape
            ``(n + 1,)``. For a pack, ``edges[k][f]`` contains the edges for
            feature ``f`` of model ``k``. Do not pad the edges.
        max_bins (int, optional): Defaults to the largest interval count in ``edges``.
            Set a larger width to match the storage of an enclosing pack.

    Shape:
        - Input: ``(B, F)`` for one model, or ``(K, B, F)`` for a pack. (should work with both)
        - Output: ``(B, F, T)`` or ``(K, B, F, T)``, respectively, where ``B``
          is the batch size, ``F`` the number of features, ``K`` the number of
          models, and ``T`` is the output width. The last axis holds interval fractions,
          not learned embedding coordinates.

    """

    # This one is the reference implementation, not efficient, but
    # shows the computation that should be vectorized in your solution

    if max_bins is None:
        models = [edges] if x_num.ndim == 2 else edges
        max_bins = max(
            len(feature_edges) - 1 for model in models for feature_edges in model
        )

    if x_num.ndim > 2:
        return torch.stack(
            [
                reference_piecewise_linear_encoding(
                    model_x, model_edges, max_bins=max_bins
                )
                for model_x, model_edges in zip(x_num, edges, strict=True)
            ]
        )

    feature_encodings = []
    for feature_index, bin_edges in enumerate(edges):
        values = x_num[:, feature_index]
        n_bins = len(bin_edges) - 1
        if max_bins < n_bins:
            raise ValueError("max_bins is smaller than the number of intervals")
        fractions = []
        for bin_index in range(n_bins):
            left, right = bin_edges[bin_index : bin_index + 2]
            fraction = (values - left) / (right - left)
            if bin_index > 0:
                fraction = fraction.clamp_min(0)
            if bin_index < n_bins - 1:
                fraction = fraction.clamp_max(1)
            fractions.append(fraction)

        encoding = torch.stack(fractions, dim=-1)
        padding = x_num.new_zeros(len(values), max_bins - n_bins)
        feature_encodings.append(
            torch.cat([encoding[:, :-1], padding, encoding[:, -1:]], dim=-1)
        )
    return torch.stack(feature_encodings, dim=1)
