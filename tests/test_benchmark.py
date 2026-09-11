"""Checks for benchmark timing boundaries and equivalent reference workloads."""

import numpy as np
import pytest

from tests import benchmark
from tests.conftest import requires_scipy, requires_torch, scipy_cubic


def test_timer_meets_budget_and_rotates_backends(monkeypatch):
    clock = 0.0
    last_backend = None
    boundaries = []
    calls = {"fast": 0, "slow": 0}

    def counter():
        boundaries.append(last_backend)
        return clock

    def work(name, duration):
        def run():
            nonlocal clock, last_backend
            clock += duration
            last_backend = name
            calls[name] += 1
            return name

        return run

    monkeypatch.setattr(benchmark.time, "perf_counter", counter)
    results = benchmark.time_callables(
        {"fast": work("fast", 0.00001), "slow": work("slow", 0.001)}, target_ms=12
    )
    # Each clock pair brackets a warmup call or a measured block.
    block_order = boundaries[1::2][2 * benchmark.WARMUP_RUNS :]
    assert block_order[:6] == ["fast", "slow", "slow", "fast", "fast", "slow"]
    for name, result in results.items():
        assert result.elapsed_ms >= 12
        assert result.blocks >= benchmark.MIN_BLOCKS
        assert calls[name] == result.runs + benchmark.WARMUP_RUNS + 1
        assert result.output == name
        assert result.iqr_ms == pytest.approx(0, abs=1e-10)
    assert results["fast"].runs > 250
    assert results["fast"].median_ms == pytest.approx(0.01)
    assert results["slow"].median_ms == pytest.approx(1)


@pytest.mark.parametrize("target_ms", [0, -1, float("nan"), float("inf")])
def test_timer_rejects_invalid_budget(target_ms):
    with pytest.raises(ValueError, match="finite and positive"):
        benchmark.time_callables({"unused": lambda: None}, target_ms)


@requires_torch
@pytest.mark.parametrize("shape", [(9, 8, 7), (2, 9, 8, 7), (2, 2, 9, 8, 7)])
@pytest.mark.parametrize(
    "mode,dtype,align_corners",
    [
        ("nearest", np.float32, False),
        ("nearest", np.uint8, False),
        ("nearest", np.int16, False),
        ("linear", np.float32, False),
        ("linear", np.float32, True),
        ("area", np.float32, False),
    ],
)
def test_prepared_resample_matches_public_reference(shape, mode, dtype, align_corners):
    data = benchmark.generate_resample_input(shape, dtype, 42)
    case = benchmark.ResampleCase("test", shape, (5, 6, 4), mode, dtype, align_corners)
    output = benchmark.prepare_torch_resample(data, case)()
    expected = benchmark.TorchReference.resample(data, case.output_size, mode, align_corners)
    assert tuple(output.shape[-3:]) == case.output_size
    assert output.ndim == 5
    actual = output.numpy().reshape(expected.shape).astype(expected.dtype)
    np.testing.assert_array_equal(actual, expected)
    if dtype == np.int16:
        assert output.dtype == benchmark.torch.float32
        assert expected.dtype == np.int16


@requires_torch
@pytest.mark.parametrize("case", benchmark.build_grid_sample_cases())
def test_prepared_grid_matches_public_reference(case):
    # Keep the padding/mode coverage of the real suite with small test volumes.
    from dataclasses import replace

    case = replace(case, input_shape=(1, 2, 9, 8, 7), grid_shape=(1, 5, 6, 4, 3))
    data, grid = benchmark.generate_grid_case_input(case)
    output = benchmark.prepare_torch_grid_sample(data, grid, case)().numpy()
    expected = benchmark.TorchReference.grid_sample(
        data, grid, mode=case.mode, padding_mode=case.padding_mode
    )
    np.testing.assert_array_equal(output, expected)


@requires_scipy
@pytest.mark.parametrize("align_corners", [False, True])
def test_scipy_direct_float32_preserves_cubic_values(align_corners):
    data = benchmark.generate_resample_input((9, 8, 7), np.float32, 42)
    output = benchmark.scipy_cubic_reference(data, (5, 11, 4), align_corners)
    expected = scipy_cubic(data, (5, 11, 4), align_corners=align_corners)
    assert output.dtype == np.float32
    np.testing.assert_array_equal(output, expected)
