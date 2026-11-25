import torch
from torch.utils.benchmark import Timer

from ple_reference import reference_piecewise_linear_encoding
from tabpack import encoding_coefficients, piecewise_linear_encoding

SHAPE = (8, 128, 16, 16)  # models, rows, features, intervals
MIN_SPEEDUP = 10.0  # relative to the loop reference on the same CPU


def median_ms(fn, args):
    timer = Timer("fn(*args)", globals={"fn": fn, "args": args}, num_threads=1)
    return 1000 * timer.blocked_autorange(min_run_time=0.5).median


@torch.inference_mode()
def test_ple_speed():
    p, b, f, t = SHAPE
    rng = torch.Generator().manual_seed(0)
    rows = torch.rand(p, b, f, device="cpu", generator=rng)
    edges = [
        row.unbind()
        for row in ((torch.arange(t + 1) + torch.rand(p, f, t + 1, generator=rng) * 0.5) / t)
    ]
    coefficients = [encoding_coefficients(member_edges, t) for member_edges in edges]
    fast_args = (
        rows,
        *(torch.stack(values) for values in zip(*coefficients, strict=True)),
    )
    slow_args = (rows, edges)
    torch.testing.assert_close(
        piecewise_linear_encoding(*fast_args),
        reference_piecewise_linear_encoding(*slow_args),
        atol=1e-4,
        rtol=1e-4,
    )
    slow = median_ms(reference_piecewise_linear_encoding, slow_args)
    fast = median_ms(piecewise_linear_encoding, fast_args)
    print(
        f"CPU (1 thread): reference {slow:.3f} ms, submitted {fast:.3f} ms "
        f"({slow / fast:.1f}×); minimum {MIN_SPEEDUP:g}×",
        flush=True,
    )
    assert slow / fast >= MIN_SPEEDUP, (
        f"Expected at least {MIN_SPEEDUP:g}× speedup, got {slow / fast:.1f}× "
        f"(reference {slow:.3f} ms, submitted {fast:.3f} ms)"
    )
