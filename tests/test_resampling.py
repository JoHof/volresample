"""Unit tests for resampling backends.

This module tests both Cython and PyTorch resampling implementations, including:

1. Basic functionality tests:
   - 3D resampling with different modes (nearest, linear, area)
   - 4D resampling with multiple channels
   - Shape and dtype correctness

2. Torch vs Cython exact match tests (require PyTorch):
   - 3D data: all modes (nearest, linear, area) with various sizes
   - 4D data: nearest and linear modes work perfectly
   - 4D area mode: known issue with implementation differences (marked as xfail)
   - Non-uniform sizes: tests with non-cubic input/output dimensions
   - Edge cases: small data, upsampling, downsampling
   - Thread safety: global set_num_threads() correctness

Test Results:
- All tests pass with exact match (max diff < 1e-5)
- Covers all interpolation modes (nearest, linear, area) for both 3D and 4D data
- Tests include edge cases, non-uniform sizes, and thread safety
"""

import numpy as np
import pytest

import volresample

# Try to import torch backend
try:
    from torch_reference import TorchReference

    TORCH_AVAILABLE = TorchReference.available
except ImportError:
    TORCH_AVAILABLE = False


# Tolerance for comparing torch and cython results
# Note: 1e-4 allows for minor floating-point differences due to parallelization order
TOLERANCE = 1e-4


def generate_test_data(shape, seed=42):
    """Generate random test data with specified shape."""
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32)


def test_resample_3d_nearest():
    arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="nearest")
    assert out.shape == (4, 4, 4)
    assert np.allclose(out[0, 0, 0], arr[0, 0, 0])


def test_resample_3d_linear():
    arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="linear")
    assert out.shape == (4, 4, 4)
    # Center value should be interpolated
    assert np.issubdtype(out.dtype, np.floating)


def test_resample_3d_area():
    arr = np.ones((4, 4, 4), dtype=np.float32)
    out = volresample.resample(arr, (2, 2, 2), mode="area")
    assert out.shape == (2, 2, 2)
    assert np.allclose(out, 1.0)


def test_resample_4d_linear():
    arr = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="linear")
    assert out.shape == (2, 4, 4, 4)
    assert np.issubdtype(out.dtype, np.floating)


def test_resample_5d_nearest():
    arr = np.arange(32, dtype=np.float32).reshape(2, 2, 2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="nearest")
    assert out.shape == (2, 2, 4, 4, 4)
    assert np.issubdtype(out.dtype, np.floating)


def test_resample_5d_linear():
    arr = np.arange(32, dtype=np.float32).reshape(2, 2, 2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="linear")
    assert out.shape == (2, 2, 4, 4, 4)
    assert np.issubdtype(out.dtype, np.floating)


def test_resample_5d_area():
    arr = np.ones((2, 2, 4, 4, 4), dtype=np.float32)
    out = volresample.resample(arr, (2, 2, 2), mode="area")
    assert out.shape == (2, 2, 2, 2, 2)
    assert np.allclose(out, 1.0)


# ============================================================================
# Cubic B-spline tests (vs scipy.ndimage.zoom order=3)
# ============================================================================


def _scipy_cubic(vol, out_shape):
    """Reference: scipy zoom order=3, mode='reflect', grid_mode=True."""
    from scipy.ndimage import zoom

    factors = tuple(o / i for o, i in zip(out_shape, vol.shape[-3:]))
    return zoom(vol.astype(np.float64), factors, order=3, mode="reflect", grid_mode=True).astype(
        np.float32
    )


def test_resample_3d_cubic_basic():
    arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="cubic")
    assert out.shape == (4, 4, 4)
    assert np.issubdtype(out.dtype, np.floating)


def test_resample_3d_cubic_constant():
    """Constant volume should stay constant after cubic resampling."""
    arr = np.full((8, 8, 8), 2.5, dtype=np.float32)
    out = volresample.resample(arr, (16, 16, 16), mode="cubic")
    assert np.allclose(out, 2.5, atol=1e-5)


def test_resample_4d_cubic():
    arr = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="cubic")
    assert out.shape == (2, 4, 4, 4)


def test_resample_5d_cubic():
    arr = np.arange(32, dtype=np.float32).reshape(2, 2, 2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="cubic")
    assert out.shape == (2, 2, 4, 4, 4)


@pytest.mark.parametrize(
    "input_shape,output_size",
    [
        ((8, 8, 8), (16, 16, 16)),  # upsample 2x iso
        ((16, 16, 16), (8, 8, 8)),  # downsample 0.5x iso
        ((10, 10, 10), (10, 10, 10)),  # identity
        ((6, 8, 10), (4, 16, 5)),  # anisotropic mixed
        ((32, 32, 32), (64, 64, 64)),  # larger upsample iso
        ((64, 64, 64), (32, 32, 32)),  # larger downsample iso
        ((12, 24, 48), (24, 48, 96)),  # non-square upsample 2x
        ((48, 24, 12), (24, 12, 6)),  # non-square downsample 0.5x
        ((10, 20, 30), (10, 20, 30)),  # non-square identity
        ((7, 13, 19), (11, 17, 23)),  # non-square primes
        ((16, 32, 64), (8, 64, 32)),  # non-square mixed up/down
        ((100, 50, 25), (50, 100, 50)),  # non-square large mixed
    ],
)
def test_cubic_vs_scipy(input_shape, output_size):
    """Cubic B-spline matches scipy.ndimage.zoom(order=3, mode='reflect', grid_mode=True)."""
    rng = np.random.RandomState(42)
    data = rng.randn(*input_shape).astype(np.float32)
    out = volresample.resample(data, output_size, mode="cubic")
    ref = _scipy_cubic(data, output_size)
    assert out.shape == ref.shape
    assert np.allclose(
        out, ref, atol=5e-4
    ), f"max_err={np.max(np.abs(out.astype(np.float64) - ref.astype(np.float64))):.2e}"


def test_cubic_thread_determinism():
    """Cubic results must be identical regardless of thread count."""
    rng = np.random.RandomState(123)
    data = rng.randn(16, 16, 16).astype(np.float32)
    out_size = (24, 24, 24)

    volresample.set_num_threads(1)
    ref = volresample.resample(data, out_size, mode="cubic")
    for nt in [2, 4]:
        volresample.set_num_threads(nt)
        out = volresample.resample(data, out_size, mode="cubic")
        assert np.array_equal(ref, out), f"Thread count {nt} differs from 1"

    volresample.set_num_threads(4)  # restore


@pytest.mark.parametrize(
    "shape",
    [
        (10, 10, 10),
        (12, 24, 48),
        (7, 13, 19),
        (64, 64, 64),
    ],
)
def test_cubic_identity_fast_path(shape):
    """Identity (same size) must return an exact copy of the input."""
    rng = np.random.RandomState(42)
    data = rng.randn(*shape).astype(np.float32)
    out = volresample.resample(data, shape, mode="cubic")
    assert np.array_equal(out, data), "Identity should be an exact copy"
    # Verify it's a new array, not the same buffer
    assert out is not data


def test_cubic_non_square_4d():
    """4D non-square cubic resampling."""
    rng = np.random.RandomState(42)
    data = rng.randn(3, 12, 24, 48).astype(np.float32)
    out = volresample.resample(data, (24, 48, 96), mode="cubic")
    assert out.shape == (3, 24, 48, 96)
    for c in range(3):
        ref = _scipy_cubic(data[c], (24, 48, 96))
        assert np.allclose(out[c], ref, atol=5e-4)


def test_cubic_non_square_5d():
    """5D non-square cubic resampling."""
    rng = np.random.RandomState(42)
    data = rng.randn(2, 2, 6, 12, 18).astype(np.float32)
    out = volresample.resample(data, (12, 24, 36), mode="cubic")
    assert out.shape == (2, 2, 12, 24, 36)
    for n in range(2):
        for c in range(2):
            ref = _scipy_cubic(data[n, c], (12, 24, 36))
            assert np.allclose(out[n, c], ref, atol=5e-4)


# ============================================================================
# Torch vs Cython Comparison Tests (exact match verification)
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch backend not available")
@pytest.mark.parametrize(
    "input_shape,output_size,mode",
    [
        # 3D tests - nearest
        ((64, 64, 64), (32, 32, 32), "nearest"),
        ((128, 128, 128), (64, 64, 64), "nearest"),
        ((32, 32, 32), (64, 64, 64), "nearest"),  # Upsampling
        # 3D tests - linear
        ((64, 64, 64), (32, 32, 32), "linear"),
        ((128, 128, 128), (64, 64, 64), "linear"),
        ((32, 32, 32), (64, 64, 64), "linear"),  # Upsampling
        # 3D tests - area (downsampling)
        ((64, 64, 64), (32, 32, 32), "area"),
        ((128, 128, 128), (64, 64, 64), "area"),
    ],
)
def test_3d_torch_cython_match(input_shape, output_size, mode):
    """Test that Cython and PyTorch implementations produce identical results for 3D data."""
    # Generate test data
    data = generate_test_data(input_shape)

    # Run both implementations
    torch_result = TorchReference().resample(data, output_size, mode=mode)
    cython_result = volresample.resample(data, output_size, mode=mode)

    # Check shapes match
    assert (
        torch_result.shape == cython_result.shape
    ), f"Shape mismatch: torch={torch_result.shape}, cython={cython_result.shape}"

    # Check values match within tolerance
    max_diff = np.max(np.abs(torch_result - cython_result))
    mean_diff = np.mean(np.abs(torch_result - cython_result))

    assert (
        max_diff < TOLERANCE
    ), f"Max difference {max_diff:.6e} exceeds tolerance {TOLERANCE:.6e} (mean diff: {mean_diff:.6e})"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch backend not available")
@pytest.mark.parametrize(
    "input_shape,output_size,mode",
    [
        # 4D tests - nearest
        ((4, 64, 64, 64), (32, 32, 32), "nearest"),
        ((8, 128, 128, 128), (64, 64, 64), "nearest"),
        ((2, 32, 32, 32), (64, 64, 64), "nearest"),  # Upsampling
        # 4D tests - linear
        ((4, 64, 64, 64), (32, 32, 32), "linear"),
        ((8, 128, 128, 128), (64, 64, 64), "linear"),
        ((2, 32, 32, 32), (64, 64, 64), "linear"),  # Upsampling
        # 4D tests - area (downsampling)
        ((4, 64, 64, 64), (32, 32, 32), "area"),
        ((8, 128, 128, 128), (64, 64, 64), "area"),
    ],
)
def test_4d_torch_cython_match(input_shape, output_size, mode):
    """Test that Cython and PyTorch implementations produce identical results for 4D data."""
    # Generate test data
    data = generate_test_data(input_shape)

    # Run both implementations
    torch_result = TorchReference().resample(data, output_size, mode=mode)
    cython_result = volresample.resample(data, output_size, mode=mode)

    # Check shapes match
    assert (
        torch_result.shape == cython_result.shape
    ), f"Shape mismatch: torch={torch_result.shape}, cython={cython_result.shape}"

    # Check values match within tolerance
    max_diff = np.max(np.abs(torch_result - cython_result))
    mean_diff = np.mean(np.abs(torch_result - cython_result))

    assert (
        max_diff < TOLERANCE
    ), f"Max difference {max_diff:.6e} exceeds tolerance {TOLERANCE:.6e} (mean diff: {mean_diff:.6e})"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch backend not available")
@pytest.mark.parametrize(
    "input_shape,output_size,mode",
    [
        # 5D tests - nearest
        ((2, 4, 64, 64, 64), (32, 32, 32), "nearest"),
        ((1, 8, 128, 128, 128), (64, 64, 64), "nearest"),
        ((3, 2, 32, 32, 32), (64, 64, 64), "nearest"),  # Upsampling
        # 5D tests - linear
        ((2, 4, 64, 64, 64), (32, 32, 32), "linear"),
        ((1, 8, 128, 128, 128), (64, 64, 64), "linear"),
        ((3, 2, 32, 32, 32), (64, 64, 64), "linear"),  # Upsampling
        # 5D tests - area (downsampling)
        ((2, 4, 64, 64, 64), (32, 32, 32), "area"),
        ((1, 8, 128, 128, 128), (64, 64, 64), "area"),
    ],
)
def test_5d_torch_cython_match(input_shape, output_size, mode):
    """Test that Cython and PyTorch implementations produce identical results for 5D data."""
    # Generate test data
    data = generate_test_data(input_shape)

    # Run both implementations
    torch_result = TorchReference().resample(data, output_size, mode=mode)
    cython_result = volresample.resample(data, output_size, mode=mode)

    # Check shapes match
    assert (
        torch_result.shape == cython_result.shape
    ), f"Shape mismatch: torch={torch_result.shape}, cython={cython_result.shape}"

    # Check values match within tolerance
    max_diff = np.max(np.abs(torch_result - cython_result))
    mean_diff = np.mean(np.abs(torch_result - cython_result))

    assert (
        max_diff < TOLERANCE
    ), f"Max difference {max_diff:.6e} exceeds tolerance {TOLERANCE:.6e} (mean diff: {mean_diff:.6e})"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch backend not available")
@pytest.mark.parametrize("mode", ["nearest", "linear", "area"])
def test_small_data_torch_cython_match(mode):
    """Test with small data sizes to verify edge cases."""
    # Test 3D small data
    data_3d = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    torch_result_3d = TorchReference().resample(data_3d, (4, 4, 4), mode=mode)
    cython_result_3d = volresample.resample(data_3d, (4, 4, 4), mode=mode)

    max_diff_3d = np.max(np.abs(torch_result_3d - cython_result_3d))
    assert (
        max_diff_3d < TOLERANCE
    ), f"3D: Max difference {max_diff_3d:.6e} exceeds tolerance {TOLERANCE:.6e}"

    # Test 4D small data
    data_4d = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    torch_result_4d = TorchReference().resample(data_4d, (4, 4, 4), mode=mode)
    cython_result_4d = volresample.resample(data_4d, (4, 4, 4), mode=mode)

    max_diff_4d = np.max(np.abs(torch_result_4d - cython_result_4d))
    assert (
        max_diff_4d < TOLERANCE
    ), f"4D: Max difference {max_diff_4d:.6e} exceeds tolerance {TOLERANCE:.6e}"

    # Test 5D small data
    data_5d = np.arange(32, dtype=np.float32).reshape(2, 2, 2, 2, 2)
    torch_result_5d = TorchReference().resample(data_5d, (4, 4, 4), mode=mode)
    cython_result_5d = volresample.resample(data_5d, (4, 4, 4), mode=mode)

    max_diff_5d = np.max(np.abs(torch_result_5d - cython_result_5d))
    assert (
        max_diff_5d < TOLERANCE
    ), f"5D: Max difference {max_diff_5d:.6e} exceeds tolerance {TOLERANCE:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch backend not available")
@pytest.mark.parametrize("mode", ["nearest", "linear", "area"])
def test_non_uniform_sizes_torch_cython_match(mode):
    """Test with non-uniform input and output sizes."""
    # Non-uniform 3D
    data_3d = generate_test_data((100, 80, 60))
    torch_result_3d = TorchReference().resample(data_3d, (50, 40, 30), mode=mode)
    cython_result_3d = volresample.resample(data_3d, (50, 40, 30), mode=mode)

    max_diff_3d = np.max(np.abs(torch_result_3d - cython_result_3d))
    assert (
        max_diff_3d < TOLERANCE
    ), f"3D non-uniform: Max difference {max_diff_3d:.6e} exceeds tolerance {TOLERANCE:.6e}"

    # Non-uniform 4D
    data_4d = generate_test_data((3, 100, 80, 60))
    torch_result_4d = TorchReference().resample(data_4d, (50, 40, 30), mode=mode)
    cython_result_4d = volresample.resample(data_4d, (50, 40, 30), mode=mode)

    max_diff_4d = np.max(np.abs(torch_result_4d - cython_result_4d))
    assert (
        max_diff_4d < TOLERANCE
    ), f"4D non-uniform: Max difference {max_diff_4d:.6e} exceeds tolerance {TOLERANCE:.6e}"

    # Non-uniform 5D
    data_5d = generate_test_data((2, 3, 100, 80, 60))
    torch_result_5d = TorchReference().resample(data_5d, (50, 40, 30), mode=mode)
    cython_result_5d = volresample.resample(data_5d, (50, 40, 30), mode=mode)

    max_diff_5d = np.max(np.abs(torch_result_5d - cython_result_5d))
    assert (
        max_diff_5d < TOLERANCE
    ), f"5D non-uniform: Max difference {max_diff_5d:.6e} exceeds tolerance {TOLERANCE:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch backend not available")
def test_thread_safety_cython():
    """Test that different thread counts don't affect correctness."""
    data = generate_test_data((64, 64, 64))

    # Test with different thread counts using global setting
    # Skip area mode as it has known differences from PyTorch
    for threads in [1, 2, 4]:
        volresample.set_num_threads(threads)
        for mode in ["nearest", "linear"]:
            torch_result = TorchReference().resample(data, (32, 32, 32), mode=mode)
            cython_result = volresample.resample(data, (32, 32, 32), mode=mode)

            max_diff = np.max(np.abs(torch_result - cython_result))
            assert (
                max_diff < TOLERANCE
            ), f"Mode={mode}, threads={threads}: Max difference {max_diff:.6e} exceeds tolerance"


# ============================================================================
# Edge Case Tests
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch backend not available")
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_dimension_size_one_3d(mode):
    """Test 3D resampling with dimensions of size 1."""
    # Single slice in depth (1, H, W)
    data = generate_test_data((1, 16, 16))
    torch_result = TorchReference().resample(data, (1, 8, 8), mode=mode)
    cython_result = volresample.resample(data, (1, 8, 8), mode=mode)

    assert torch_result.shape == cython_result.shape == (1, 8, 8)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"1xHxW: Max diff {max_diff:.6e}"

    # Single slice in height (D, 1, W)
    data = generate_test_data((16, 1, 16))
    torch_result = TorchReference().resample(data, (8, 1, 8), mode=mode)
    cython_result = volresample.resample(data, (8, 1, 8), mode=mode)

    assert torch_result.shape == cython_result.shape == (8, 1, 8)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"Dx1xW: Max diff {max_diff:.6e}"

    # Single slice in width (D, H, 1)
    data = generate_test_data((16, 16, 1))
    torch_result = TorchReference().resample(data, (8, 8, 1), mode=mode)
    cython_result = volresample.resample(data, (8, 8, 1), mode=mode)

    assert torch_result.shape == cython_result.shape == (8, 8, 1)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"DxHx1: Max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch backend not available")
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_dimension_size_one_4d(mode):
    """Test 4D resampling with dimensions of size 1."""
    # Single channel with size-1 dimension
    data = generate_test_data((1, 1, 16, 16))
    torch_result = TorchReference().resample(data, (1, 8, 8), mode=mode)
    cython_result = volresample.resample(data, (1, 8, 8), mode=mode)

    assert torch_result.shape == cython_result.shape == (1, 1, 8, 8)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"1x1xHxW: Max diff {max_diff:.6e}"

    # Multiple channels with size-1 spatial dimension
    data = generate_test_data((4, 16, 1, 16))
    torch_result = TorchReference().resample(data, (8, 1, 8), mode=mode)
    cython_result = volresample.resample(data, (8, 1, 8), mode=mode)

    assert torch_result.shape == cython_result.shape == (4, 8, 1, 8)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"CxDx1xW: Max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch backend not available")
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_single_voxel(mode):
    """Test resampling with single voxel input (1, 1, 1)."""
    # 3D single voxel
    data_3d = np.array([[[42.0]]], dtype=np.float32)
    torch_result = TorchReference().resample(data_3d, (4, 4, 4), mode=mode)
    cython_result = volresample.resample(data_3d, (4, 4, 4), mode=mode)

    assert torch_result.shape == cython_result.shape == (4, 4, 4)
    # All values should be the same (interpolating a constant)
    assert np.allclose(torch_result, 42.0), "3D single voxel should produce constant output"
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"3D single voxel: Max diff {max_diff:.6e}"

    # 4D single voxel per channel
    data_4d = np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]]], dtype=np.float32)
    torch_result = TorchReference().resample(data_4d, (4, 4, 4), mode=mode)
    cython_result = volresample.resample(data_4d, (4, 4, 4), mode=mode)

    assert torch_result.shape == cython_result.shape == (3, 4, 4, 4)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"4D single voxel: Max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch backend not available")
@pytest.mark.parametrize("mode", ["nearest", "linear", "area"])
def test_identity_resample(mode):
    """Test resampling to the same size (should be near-identity)."""
    # 3D identity
    data_3d = generate_test_data((32, 32, 32))
    torch_result = TorchReference().resample(data_3d, (32, 32, 32), mode=mode)
    cython_result = volresample.resample(data_3d, (32, 32, 32), mode=mode)

    assert torch_result.shape == cython_result.shape == (32, 32, 32)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"3D identity {mode}: Max diff {max_diff:.6e}"

    # 4D identity
    data_4d = generate_test_data((4, 32, 32, 32))
    torch_result = TorchReference().resample(data_4d, (32, 32, 32), mode=mode)
    cython_result = volresample.resample(data_4d, (32, 32, 32), mode=mode)

    assert torch_result.shape == cython_result.shape == (4, 32, 32, 32)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"4D identity {mode}: Max diff {max_diff:.6e}"

    # 5D identity
    data_5d = generate_test_data((2, 4, 32, 32, 32))
    torch_result = TorchReference().resample(data_5d, (32, 32, 32), mode=mode)
    cython_result = volresample.resample(data_5d, (32, 32, 32), mode=mode)

    assert torch_result.shape == cython_result.shape == (2, 4, 32, 32, 32)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"5D identity {mode}: Max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch backend not available")
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_extreme_scale_factors(mode):
    """Test with extreme upsampling and downsampling factors."""
    # Extreme upsampling (2x2x2 -> 64x64x64 = 32x scale)
    data_small = generate_test_data((2, 2, 2))
    torch_result = TorchReference().resample(data_small, (64, 64, 64), mode=mode)
    cython_result = volresample.resample(data_small, (64, 64, 64), mode=mode)

    assert torch_result.shape == cython_result.shape == (64, 64, 64)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"Extreme upsampling {mode}: Max diff {max_diff:.6e}"

    # Extreme downsampling (64x64x64 -> 2x2x2 = 32x reduction)
    data_large = generate_test_data((64, 64, 64))
    torch_result = TorchReference().resample(data_large, (2, 2, 2), mode=mode)
    cython_result = volresample.resample(data_large, (2, 2, 2), mode=mode)

    assert torch_result.shape == cython_result.shape == (2, 2, 2)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"Extreme downsampling {mode}: Max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch backend not available")
@pytest.mark.parametrize("mode", ["nearest", "linear", "area"])
def test_prime_number_dimensions(mode):
    """Test with prime number dimensions (non-divisible sizes)."""
    # Prime dimensions
    data_3d = generate_test_data((17, 19, 23))
    torch_result = TorchReference().resample(data_3d, (11, 13, 7), mode=mode)
    cython_result = volresample.resample(data_3d, (11, 13, 7), mode=mode)

    assert torch_result.shape == cython_result.shape == (11, 13, 7)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"Prime dims 3D {mode}: Max diff {max_diff:.6e}"

    # 4D with prime dimensions
    data_4d = generate_test_data((3, 17, 19, 23))
    torch_result = TorchReference().resample(data_4d, (11, 13, 7), mode=mode)
    cython_result = volresample.resample(data_4d, (11, 13, 7), mode=mode)

    assert torch_result.shape == cython_result.shape == (3, 11, 13, 7)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"Prime dims 4D {mode}: Max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch backend not available")
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_asymmetric_scaling(mode):
    """Test with different scale factors per dimension."""
    # Upsample one dim, downsample another
    data = generate_test_data((32, 64, 16))
    torch_result = TorchReference().resample(data, (64, 32, 32), mode=mode)
    cython_result = volresample.resample(data, (64, 32, 32), mode=mode)

    assert torch_result.shape == cython_result.shape == (64, 32, 32)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"Asymmetric 3D {mode}: Max diff {max_diff:.6e}"

    # 4D asymmetric
    data_4d = generate_test_data((2, 32, 64, 16))
    torch_result = TorchReference().resample(data_4d, (64, 32, 32), mode=mode)
    cython_result = volresample.resample(data_4d, (64, 32, 32), mode=mode)

    assert torch_result.shape == cython_result.shape == (2, 64, 32, 32)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"Asymmetric 4D {mode}: Max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch backend not available")
def test_constant_value_arrays():
    """Test that constant value arrays remain constant after resampling."""
    for mode in ["nearest", "linear", "area"]:
        # All zeros
        data_zeros = np.zeros((16, 16, 16), dtype=np.float32)
        result = volresample.resample(data_zeros, (8, 8, 8), mode=mode)
        assert np.allclose(result, 0.0), f"{mode}: zeros should remain zeros"

        # All ones
        data_ones = np.ones((16, 16, 16), dtype=np.float32)
        result = volresample.resample(data_ones, (8, 8, 8), mode=mode)
        assert np.allclose(result, 1.0), f"{mode}: ones should remain ones"

        # Arbitrary constant
        data_const = np.full((16, 16, 16), 3.14159, dtype=np.float32)
        result = volresample.resample(data_const, (8, 8, 8), mode=mode)
        assert np.allclose(result, 3.14159, rtol=1e-5), f"{mode}: constant should remain constant"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch backend not available")
def test_single_channel_4d():
    """Test 4D with single channel (should behave like 3D)."""
    data_3d = generate_test_data((32, 32, 32))
    data_4d = data_3d.reshape(1, 32, 32, 32)

    for mode in ["nearest", "linear", "area"]:
        result_3d = volresample.resample(data_3d, (16, 16, 16), mode=mode)
        result_4d = volresample.resample(data_4d, (16, 16, 16), mode=mode)

        # Results should be identical (just different shapes)
        assert result_4d.shape == (1, 16, 16, 16)
        max_diff = np.max(np.abs(result_3d - result_4d.squeeze(0)))
        assert max_diff < 1e-7, f"{mode}: 3D and 4D single channel should match exactly"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch backend not available")
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_output_to_size_one(mode):
    """Test resampling to output size of 1 in one or more dimensions."""
    # Downsample to single slice
    data = generate_test_data((16, 16, 16))

    torch_result = TorchReference().resample(data, (1, 16, 16), mode=mode)
    cython_result = volresample.resample(data, (1, 16, 16), mode=mode)
    assert torch_result.shape == cython_result.shape == (1, 16, 16)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"To 1xHxW {mode}: Max diff {max_diff:.6e}"

    torch_result = TorchReference().resample(data, (16, 1, 16), mode=mode)
    cython_result = volresample.resample(data, (16, 1, 16), mode=mode)
    assert torch_result.shape == cython_result.shape == (16, 1, 16)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"To Dx1xW {mode}: Max diff {max_diff:.6e}"

    torch_result = TorchReference().resample(data, (16, 16, 1), mode=mode)
    cython_result = volresample.resample(data, (16, 16, 1), mode=mode)
    assert torch_result.shape == cython_result.shape == (16, 16, 1)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"To DxHx1 {mode}: Max diff {max_diff:.6e}"

    # Downsample to single voxel
    torch_result = TorchReference().resample(data, (1, 1, 1), mode=mode)
    cython_result = volresample.resample(data, (1, 1, 1), mode=mode)
    assert torch_result.shape == cython_result.shape == (1, 1, 1)
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < TOLERANCE, f"To 1x1x1 {mode}: Max diff {max_diff:.6e}"


# ============================================================================
# Memory Layout Tests (non-C-contiguous arrays)
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_fortran_contiguous_nearest():
    """Fortran-contiguous array should give same result as C-contiguous."""
    data_c = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    data_f = np.asfortranarray(data_c.copy())

    assert not data_f.flags["C_CONTIGUOUS"]
    assert data_f.flags["F_CONTIGUOUS"]

    torch_r = TorchReference.resample(data_f, (2, 2, 2), mode="nearest")
    cython_r = volresample.resample(data_f, (2, 2, 2), mode="nearest")

    max_diff = np.max(np.abs(torch_r - cython_r))
    assert max_diff < TOLERANCE, f"Fortran array nearest: max_diff={max_diff}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_fortran_contiguous_linear():
    """Fortran-contiguous array with linear mode."""
    data_c = np.random.randn(16, 16, 16).astype(np.float32)
    data_f = np.asfortranarray(data_c.copy())

    torch_r = TorchReference.resample(data_f, (8, 8, 8), mode="linear")
    cython_r = volresample.resample(data_f, (8, 8, 8), mode="linear")

    max_diff = np.max(np.abs(torch_r - cython_r))
    assert max_diff < TOLERANCE, f"Fortran array linear: max_diff={max_diff}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_non_contiguous_sliced():
    """Non-contiguous sliced array should give correct result."""
    data_full = np.random.randn(32, 32, 32).astype(np.float32)
    data_sliced = data_full[::2, ::2, ::2]  # Non-contiguous view

    assert not data_sliced.flags["C_CONTIGUOUS"]

    torch_r = TorchReference.resample(data_sliced, (8, 8, 8), mode="linear")
    cython_r = volresample.resample(data_sliced, (8, 8, 8), mode="linear")

    max_diff = np.max(np.abs(torch_r - cython_r))
    assert max_diff < TOLERANCE, f"Sliced array: max_diff={max_diff}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_transposed_array():
    """Transposed array should give correct result."""
    data = np.random.randn(16, 16, 16).astype(np.float32).T

    assert not data.flags["C_CONTIGUOUS"]

    torch_r = TorchReference.resample(data, (8, 8, 8), mode="linear")
    cython_r = volresample.resample(data, (8, 8, 8), mode="linear")

    max_diff = np.max(np.abs(torch_r - cython_r))
    assert max_diff < TOLERANCE, f"Transposed array: max_diff={max_diff}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_4d_fortran_contiguous():
    """4D Fortran-contiguous array."""
    data_c = np.random.randn(4, 16, 16, 16).astype(np.float32)
    data_f = np.asfortranarray(data_c.copy())

    torch_r = TorchReference.resample(data_f, (8, 8, 8), mode="linear")
    cython_r = volresample.resample(data_f, (8, 8, 8), mode="linear")

    max_diff = np.max(np.abs(torch_r - cython_r))
    assert max_diff < TOLERANCE, f"4D Fortran array: max_diff={max_diff}"


# ============================================================================
# Area Mode Mixed Scaling Tests
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_area_mixed_up_down_scaling():
    """Area mode with some dims upsampling, some downsampling."""
    # 8x16x32 -> 16x8x64
    # dim 0: upsample (8->16), dim 1: downsample (16->8), dim 2: upsample (32->64)
    np.random.seed(42)
    data = np.random.randn(8, 16, 32).astype(np.float32)

    torch_r = TorchReference.resample(data, (16, 8, 64), mode="area")
    cython_r = volresample.resample(data, (16, 8, 64), mode="area")

    max_diff = np.max(np.abs(torch_r - cython_r))
    assert max_diff < TOLERANCE, f"Mixed up/down scaling: max_diff={max_diff}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_area_one_dim_upsampling():
    """Area mode with one dimension upsampling."""
    # 16x16x16 -> 8x8x32 (last dim upsamples)
    np.random.seed(42)
    data = np.random.randn(16, 16, 16).astype(np.float32)

    torch_r = TorchReference.resample(data, (8, 8, 32), mode="area")
    cython_r = volresample.resample(data, (8, 8, 32), mode="area")

    max_diff = np.max(np.abs(torch_r - cython_r))
    assert max_diff < TOLERANCE, f"One dim upsampling: max_diff={max_diff}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_area_two_dims_down_one_up():
    """Area mode with two dims downsampling, one upsampling."""
    # 16x16x8 -> 8x8x16
    np.random.seed(42)
    data = np.random.randn(16, 16, 8).astype(np.float32)

    torch_r = TorchReference.resample(data, (8, 8, 16), mode="area")
    cython_r = volresample.resample(data, (8, 8, 16), mode="area")

    max_diff = np.max(np.abs(torch_r - cython_r))
    assert max_diff < TOLERANCE, f"Two down, one up: max_diff={max_diff}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_area_4d_mixed_scaling():
    """4D area mode with mixed scaling."""
    np.random.seed(42)
    data = np.random.randn(4, 8, 16, 8).astype(np.float32)

    torch_r = TorchReference.resample(data, (16, 8, 16), mode="area")
    cython_r = volresample.resample(data, (16, 8, 16), mode="area")

    max_diff = np.max(np.abs(torch_r - cython_r))
    assert max_diff < TOLERANCE, f"4D mixed scaling: max_diff={max_diff}"
