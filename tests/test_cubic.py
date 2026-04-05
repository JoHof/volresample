"""Tests for cubic B-spline resampling.

Covers basic smoke tests, scipy.ndimage.zoom(order=3) comparison,
align_corners, identity fast-path, thread determinism, and edge cases
(singleton / short axes).
"""

import numpy as np
import pytest
from conftest import ATOL_CUBIC, SCIPY_AVAILABLE, requires_scipy, scipy_cubic

import volresample

# ============================================================================
# Basic smoke tests
# ============================================================================


def test_resample_3d_cubic_basic():
    arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="cubic")
    assert out.shape == (4, 4, 4)
    assert np.issubdtype(out.dtype, np.floating)


def test_resample_3d_cubic_constant():
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


# ============================================================================
# Cubic vs scipy (align_corners=False, default)
# ============================================================================


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available")
@pytest.mark.parametrize(
    "input_shape,output_size",
    [
        ((8, 8, 8), (16, 16, 16)),
        ((16, 16, 16), (8, 8, 8)),
        ((10, 10, 10), (10, 10, 10)),
        ((6, 8, 10), (4, 16, 5)),
        ((32, 32, 32), (64, 64, 64)),
        ((64, 64, 64), (32, 32, 32)),
        ((12, 24, 48), (24, 48, 96)),
        ((48, 24, 12), (24, 12, 6)),
        ((10, 20, 30), (10, 20, 30)),
        ((7, 13, 19), (11, 17, 23)),
        ((16, 32, 64), (8, 64, 32)),
        ((100, 50, 25), (50, 100, 50)),
    ],
)
def test_cubic_vs_scipy(input_shape, output_size):
    rng = np.random.RandomState(42)
    data = rng.randn(*input_shape).astype(np.float32)
    out = volresample.resample(data, output_size, mode="cubic")
    ref = scipy_cubic(data, output_size)
    assert out.shape == ref.shape
    assert np.allclose(
        out, ref, atol=ATOL_CUBIC
    ), f"max_err={np.max(np.abs(out.astype(np.float64) - ref.astype(np.float64))):.2e}"


# ============================================================================
# Identity fast path
# ============================================================================


@pytest.mark.parametrize("shape", [(10, 10, 10), (12, 24, 48), (7, 13, 19), (64, 64, 64)])
def test_cubic_identity_fast_path(shape):
    rng = np.random.RandomState(42)
    data = rng.randn(*shape).astype(np.float32)
    out = volresample.resample(data, shape, mode="cubic")
    assert np.array_equal(out, data)
    assert out is not data


# ============================================================================
# Thread determinism
# ============================================================================


def test_cubic_thread_determinism():
    rng = np.random.RandomState(123)
    data = rng.randn(16, 16, 16).astype(np.float32)

    volresample.set_num_threads(1)
    ref = volresample.resample(data, (24, 24, 24), mode="cubic")
    for nt in [2, 4]:
        volresample.set_num_threads(nt)
        out = volresample.resample(data, (24, 24, 24), mode="cubic")
        assert np.array_equal(ref, out), f"Thread count {nt} differs from 1"
    volresample.set_num_threads(4)


# ============================================================================
# Non-square 4D / 5D
# ============================================================================


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available")
def test_cubic_non_square_4d():
    rng = np.random.RandomState(42)
    data = rng.randn(3, 12, 24, 48).astype(np.float32)
    out = volresample.resample(data, (24, 48, 96), mode="cubic")
    assert out.shape == (3, 24, 48, 96)
    for c in range(3):
        ref = scipy_cubic(data[c], (24, 48, 96))
        assert np.allclose(out[c], ref, atol=ATOL_CUBIC)


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available")
def test_cubic_non_square_5d():
    rng = np.random.RandomState(42)
    data = rng.randn(2, 2, 6, 12, 18).astype(np.float32)
    out = volresample.resample(data, (12, 24, 36), mode="cubic")
    assert out.shape == (2, 2, 12, 24, 36)
    for n in range(2):
        for c in range(2):
            ref = scipy_cubic(data[n, c], (12, 24, 36))
            assert np.allclose(out[n, c], ref, atol=ATOL_CUBIC)


# ============================================================================
# align_corners=True
# ============================================================================


def test_cubic_align_corners_false_unchanged():
    rng = np.random.RandomState(42)
    data = rng.randn(16, 16, 16).astype(np.float32)
    out_default = volresample.resample(data, (8, 8, 8), mode="cubic")
    out_explicit = volresample.resample(data, (8, 8, 8), mode="cubic", align_corners=False)
    assert np.array_equal(out_default, out_explicit)


def test_cubic_align_corners_basic():
    arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="cubic", align_corners=True)
    assert out.shape == (4, 4, 4)
    assert np.issubdtype(out.dtype, np.floating)


def test_cubic_align_corners_constant():
    arr = np.full((8, 8, 8), 2.5, dtype=np.float32)
    out = volresample.resample(arr, (16, 16, 16), mode="cubic", align_corners=True)
    assert np.allclose(out, 2.5, atol=1e-5)


def test_cubic_align_corners_4d():
    arr = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="cubic", align_corners=True)
    assert out.shape == (2, 4, 4, 4)


def test_cubic_align_corners_5d():
    arr = np.arange(32, dtype=np.float32).reshape(2, 2, 2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="cubic", align_corners=True)
    assert out.shape == (2, 2, 4, 4, 4)


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available")
@pytest.mark.parametrize(
    "input_shape,output_size",
    [
        ((8, 8, 8), (16, 16, 16)),
        ((16, 16, 16), (8, 8, 8)),
        ((10, 10, 10), (10, 10, 10)),
        ((6, 8, 10), (4, 16, 5)),
        ((32, 32, 32), (64, 64, 64)),
        ((64, 64, 64), (32, 32, 32)),
        ((12, 24, 48), (24, 48, 96)),
        ((7, 13, 19), (11, 17, 23)),
        ((16, 32, 64), (8, 64, 32)),
    ],
)
def test_cubic_align_corners_vs_scipy(input_shape, output_size):
    rng = np.random.RandomState(42)
    data = rng.randn(*input_shape).astype(np.float32)
    out = volresample.resample(data, output_size, mode="cubic", align_corners=True)
    ref = scipy_cubic(data, output_size, align_corners=True)
    assert out.shape == ref.shape
    assert np.allclose(
        out, ref, atol=ATOL_CUBIC
    ), f"max_err={np.max(np.abs(out.astype(np.float64) - ref.astype(np.float64))):.2e}"


def test_cubic_align_corners_identity():
    rng = np.random.RandomState(42)
    data = rng.randn(10, 10, 10).astype(np.float32)
    out = volresample.resample(data, (10, 10, 10), mode="cubic", align_corners=True)
    assert np.array_equal(out, data)


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available")
def test_cubic_align_corners_4d_vs_scipy():
    rng = np.random.RandomState(42)
    data = rng.randn(3, 12, 24, 48).astype(np.float32)
    out = volresample.resample(data, (24, 48, 96), mode="cubic", align_corners=True)
    assert out.shape == (3, 24, 48, 96)
    for c in range(3):
        ref = scipy_cubic(data[c], (24, 48, 96), align_corners=True)
        assert np.allclose(out[c], ref, atol=ATOL_CUBIC)


def test_cubic_align_corners_thread_determinism():
    rng = np.random.RandomState(123)
    data = rng.randn(16, 16, 16).astype(np.float32)

    volresample.set_num_threads(1)
    ref = volresample.resample(data, (24, 24, 24), mode="cubic", align_corners=True)
    for nt in [2, 4]:
        volresample.set_num_threads(nt)
        out = volresample.resample(data, (24, 24, 24), mode="cubic", align_corners=True)
        assert np.array_equal(ref, out), f"Thread count {nt} differs from 1"
    volresample.set_num_threads(4)


# ============================================================================
# Cubic singleton / short axes (from issue audit)
# ============================================================================


@requires_scipy
@pytest.mark.parametrize(
    "shape,out_shape",
    [
        ((1, 8, 8), (4, 16, 16)),
        ((8, 1, 8), (16, 4, 16)),
        ((1, 1, 8), (4, 4, 16)),
    ],
)
@pytest.mark.parametrize("align_corners", [False, True])
def test_cubic_singleton_axes_match_scipy(shape, out_shape, align_corners):
    data = np.full(shape, 2.5, dtype=np.float32)
    ref = scipy_cubic(data, out_shape, align_corners=align_corners)
    out = volresample.resample(data, out_shape, mode="cubic", align_corners=align_corners)
    assert np.allclose(out, ref, atol=1e-5)


@requires_scipy
@pytest.mark.parametrize("align_corners", [False, True])
def test_cubic_short_axes_match_scipy(align_corners):
    rng = np.random.default_rng(3)
    data = rng.normal(size=(2, 2, 2)).astype(np.float32)
    ref = scipy_cubic(data, (5, 5, 5), align_corners=align_corners)
    out = volresample.resample(data, (5, 5, 5), mode="cubic", align_corners=align_corners)
    assert np.allclose(out, ref, atol=ATOL_CUBIC)
