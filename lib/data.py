"""Load the tabular-common format and prepare features. Never transform targets."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


def load_info(dataset_dir):
    return json.loads((Path(dataset_dir).expanduser() / "info.json").read_text())


def load_data(dataset_dir, split_id=("default",), *, device):
    directory = Path(dataset_dir).expanduser()
    # Nested splits may share a test set in a parent directory (yr-template format).
    indices = {}
    split_dir = directory / "splits"
    for name in split_id:
        split_dir /= name
        if not split_dir.is_dir():
            raise FileNotFoundError(split_dir)
        for path in sorted(split_dir.glob("*.npy")):
            if path.stem in indices:
                raise ValueError(f"Duplicate split part: {path.stem}")
            indices[path.stem] = torch.from_numpy(np.load(path).astype(np.int64))
    if not {"train", "val", "test"} <= indices.keys():
        raise ValueError("Expected train, val, and test indices")

    data = {}
    for key in ("x_num", "x_bin", "x_cat", "y"):
        path = directory / f"{key}.npy"
        if not path.exists():
            continue
        array = np.load(path, allow_pickle=False)
        if array.dtype.kind in "US":
            # Lossless string -> integer representation, NOT a fitted vocabulary.
            # Training-only vocabulary fitting happens in encode_categories below.
            columns = []
            for column in array.T:
                codes = {
                    value: i for i, value in enumerate(sorted(set(column.tolist())))
                }
                columns.append(
                    torch.tensor([codes[value] for value in column.tolist()])
                )
            tensor = torch.stack(columns, dim=1)
        else:
            tensor = torch.from_numpy(array)
        if key in ("x_num", "x_bin"):
            tensor = tensor.float()
        data[key] = {part: tensor[idx].to(device) for part, idx in indices.items()}
    return data


class NumericalTransform(nn.Module):
    """Mean imputation followed by standardization or fit-time-noisy quantiles.

    Adapted conceptually from NVIDIA structured-data-models: fitted Torch buffers,
    searchsorted interpolation, symmetric interpolation at ties, finite normal tails.
    Noise (std=1e-5) is used during fitting only, as in yr-template.
    """

    def __init__(self, kind):
        super().__init__()
        if kind not in (None, "standard", "noisy-quantile"):
            raise ValueError(f"Unknown numerical transform: {kind}")
        self.kind = kind
        for name in ("mean", "scale", "quantiles", "references"):
            self.register_buffer(name, torch.empty(0))

    @torch.no_grad()
    def fit(self, x, *, seed=0):
        x = x.float()
        finite = torch.isfinite(x)
        count = finite.sum(0).clamp_min(1)
        self.mean = torch.where(finite, x, 0).sum(0) / count
        x = torch.where(finite, x, self.mean)
        self.scale = x.std(0, correction=0)
        self.scale = torch.where(self.scale > 0, self.scale, 1)
        if self.kind == "noisy-quantile":
            # CPU RNG gives the same noise on CPU/CUDA/MPS, independently of training.
            generator = torch.Generator().manual_seed(seed)
            noise = torch.randn(x.shape, generator=generator).to(x.device) * 1e-5
            n_quantiles = min(len(x), max(min(len(x) // 30, 1000), 10))
            self.references = torch.linspace(0, 1, n_quantiles, device=x.device)
            self.quantiles = torch.quantile(
                x + noise, self.references, dim=0
            ).T.contiguous()
        return self

    @staticmethod
    def _interp(x, grid, values):
        if grid.shape[1] == 1:
            return values[0].expand_as(x)
        idx = torch.searchsorted(grid, x.contiguous(), right=True)
        idx = idx.clamp(1, grid.shape[1] - 1)
        lo, hi = grid.gather(1, idx - 1), grid.gather(1, idx)
        width = hi - lo
        weight = (x - lo) / torch.where(width != 0, width, 1)
        out = values[idx - 1] + weight * (values[idx] - values[idx - 1])
        out = torch.where(x <= grid[:, :1], values[0], out)
        return torch.where(x >= grid[:, -1:], values[-1], out)

    def forward(self, x):
        x = torch.where(torch.isfinite(x), x, self.mean)
        if self.kind == "standard":
            return (x - self.mean) / self.scale
        if self.kind == "noisy-quantile":
            x = x.T
            forward = self._interp(x, self.quantiles, self.references)
            backward = self._interp(
                -x, -self.quantiles.flip(1), -self.references.flip(0)
            )
            probability = ((forward - backward) * 0.5).clamp(1e-7, 1 - 1e-7)
            return (math.sqrt(2) * torch.erfinv(2 * probability - 1)).T.contiguous()
        return x


def encode_categories(parts):
    """Known training categories map to 0..C-1; unseen categories map to C."""
    output = {part: torch.empty_like(x, dtype=torch.long) for part, x in parts.items()}
    for column in range(parts["train"].shape[1]):
        vocabulary = torch.unique(parts["train"][:, column], sorted=True)
        for part, x in parts.items():
            values = x[:, column].contiguous()
            idx = torch.searchsorted(vocabulary, values)
            known = (idx < len(vocabulary)) & (
                vocabulary[idx.clamp_max(len(vocabulary) - 1)] == values
            )
            output[part][:, column] = torch.where(known, idx, len(vocabulary))
    return output


def build_data(
    dataset_dir,
    split_id=("default",),
    *,
    device,
    num_transform="noisy-quantile",
    seed=0,
):
    data = load_data(dataset_dir, split_id, device=device)
    if "x_num" in data:
        transform = NumericalTransform(num_transform).fit(
            data["x_num"]["train"], seed=seed
        )
        data["x_num"] = {part: transform(x) for part, x in data["x_num"].items()}
    if "x_bin" in data:
        # Binary features remain numerical; do not discover feature types on test data.
        binary = data.pop("x_bin")
        for part, x in binary.items():
            binary[part] = torch.where(torch.isfinite(x), x, 2.0)
        if "x_num" in data:
            data["x_num"] = {
                part: torch.cat([x, binary[part]], 1)
                for part, x in data["x_num"].items()
            }
        else:
            data["x_num"] = binary
    if "x_cat" in data:
        # Missing categories have their own sentinel, distinct from unseen values.
        if data["x_cat"]["train"].is_floating_point():
            data["x_cat"] = {
                part: torch.where(torch.isnan(x), -torch.inf, x)
                for part, x in data["x_cat"].items()
            }
        data["x_cat"] = encode_categories(data["x_cat"])
    return data
