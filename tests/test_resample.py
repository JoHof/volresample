"""Tests for resample() — nearest, linear, and area modes.

Covers basic smoke tests, PyTorch comparison, align_corners (linear),
edge cases, memory layout, and area mixed-scaling.
"""

import numpy as np
import pytest
from conftest import ATOL, TORCH_AVAILABLE, make_data

import volresample

if TORCH_AVAILABLE:
    from torch_reference import TorchReference


# ============================================================================
# Basic smoke tests (no external reference needed)
# ============================================================================


def test_resample_3d_nearest():
    arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="nearest")
    assert out.shape == (4, 4, 4)
    assert np.allclose(out[0, 0, 0], arr[0, 0, 0])


def test_resample_3d_linear():
    arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="linear")
    assert out.shape == (4, 4, 4)
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


def test_resample_5d_linear():
    arr = np.arange(32, dtype=np.float32).reshape(2, 2, 2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="linear")
    assert out.shape == (2, 2, 4, 4, 4)


def test_resample_5d_area():
    arr = np.ones((2, 2, 4, 4, 4), dtype=np.float32)
    out = volresample.resample(arr, (2, 2, 2), mode="area")
    assert out.shape == (2, 2, 2, 2, 2)
    assert np.allclose(out, 1.0)


# ============================================================================
# Linear align_corners (no torch)
# ============================================================================


def test_linear_align_corners_basic():
    arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="linear", align_corners=True)
    assert out.shape == (4, 4, 4)
    assert np.isclose(out[0, 0, 0], arr[0, 0, 0], atol=1e-6)
    assert np.isclose(out[-1, -1, -1], arr[-1, -1, -1], atol=1e-6)


def test_linear_align_corners_constant():
    arr = np.full((8, 8, 8), 3.5, dtype=np.float32)
    out = volresample.resample(arr, (16, 16, 16), mode="linear", align_corners=True)
    assert np.allclose(out, 3.5, atol=1e-5)


def test_linear_align_corners_4d():
    arr = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="linear", align_corners=True)
    assert out.shape == (2, 4, 4, 4)


def test_linear_align_corners_5d():
    arr = np.arange(32, dtype=np.float32).reshape(2, 2, 2, 2, 2)
    out = volresample.resample(arr, (4, 4, 4), mode="linear", align_corners=True)
    assert out.shape == (2, 2, 4, 4, 4)


def test_linear_align_corners_false_unchanged():
    rng = np.random.RandomState(42)
    data = rng.randn(16, 16, 16).astype(np.float32)
    out_default = volresample.resample(data, (8, 8, 8), mode="linear")
    out_explicit = volresample.resample(data, (8, 8, 8), mode="linear", align_corners=False)
    assert np.array_equal(out_default, out_explicit)


# ============================================================================
# Torch vs Cython — nearest / linear / area (parametrised)
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize(
    "input_shape,output_size,mode",
    [
        ((64, 64, 64), (32, 32, 32), "nearest"),
        ((128, 128, 128), (64, 64, 64), "nearest"),
        ((32, 32, 32), (64, 64, 64), "nearest"),
        ((64, 64, 64), (32, 32, 32), "linear"),
        ((128, 128, 128), (64, 64, 64), "linear"),
        ((32, 32, 32), (64, 64, 64), "linear"),
        ((64, 64, 64), (32, 32, 32), "area"),
        ((128, 128, 128), (64, 64, 64), "area"),
    ],
)
def test_3d_torch_cython_match(input_shape, output_size, mode):
    data = make_data(input_shape)
    torch_result = TorchReference.resample(data, output_size, mode=mode)
    cython_result = volresample.resample(data, output_size, mode=mode)
    assert torch_result.shape == cython_result.shape
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < ATOL, f"max diff {max_diff:.2e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize(
    "input_shape,output_size,mode",
    [
        ((4, 64, 64, 64), (32, 32, 32), "nearest"),
        ((8, 128, 128, 128), (64, 64, 64), "nearest"),
        ((2, 32, 32, 32), (64, 64, 64), "nearest"),
        ((4, 64, 64, 64), (32, 32, 32), "linear"),
        ((8, 128, 128, 128), (64, 64, 64), "linear"),
        ((2, 32, 32, 32), (64, 64, 64), "linear"),
        ((4, 64, 64, 64), (32, 32, 32), "area"),
        ((8, 128, 128, 128), (64, 64, 64), "area"),
    ],
)
def test_4d_torch_cython_match(input_shape, output_size, mode):
    data = make_data(input_shape)
    torch_result = TorchReference.resample(data, output_size, mode=mode)
    cython_result = volresample.resample(data, output_size, mode=mode)
    assert torch_result.shape == cython_result.shape
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < ATOL, f"max diff {max_diff:.2e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize(
    "input_shape,output_size,mode",
    [
        ((2, 4, 64, 64, 64), (32, 32, 32), "nearest"),
        ((1, 8, 128, 128, 128), (64, 64, 64), "nearest"),
        ((3, 2, 32, 32, 32), (64, 64, 64), "nearest"),
        ((2, 4, 64, 64, 64), (32, 32, 32), "linear"),
        ((1, 8, 128, 128, 128), (64, 64, 64), "linear"),
        ((3, 2, 32, 32, 32), (64, 64, 64), "linear"),
        ((2, 4, 64, 64, 64), (32, 32, 32), "area"),
        ((1, 8, 128, 128, 128), (64, 64, 64), "area"),
    ],
)
def test_5d_torch_cython_match(input_shape, output_size, mode):
    data = make_data(input_shape)
    torch_result = TorchReference.resample(data, output_size, mode=mode)
    cython_result = volresample.resample(data, output_size, mode=mode)
    assert torch_result.shape == cython_result.shape
    max_diff = np.max(np.abs(torch_result - cython_result))
    assert max_diff < ATOL, f"max diff {max_diff:.2e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["nearest", "linear", "area"])
def test_small_data_torch_cython_match(mode):
    for shape in [(2, 2, 2), (2, 2, 2, 2), (2, 2, 2, 2, 2)]:
        data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
        ref = TorchReference.resample(data, (4, 4, 4), mode=mode)
        out = volresample.resample(data, (4, 4, 4), mode=mode)
        max_diff = np.max(np.abs(ref - out))
        assert max_diff < ATOL, f"{len(shape)}D {mode}: max diff {max_diff:.2e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["nearest", "linear", "area"])
def test_non_uniform_sizes_torch_cython_match(mode):
    for shape in [(100, 80, 60), (3, 100, 80, 60), (2, 3, 100, 80, 60)]:
        data = make_data(shape)
        ref = TorchReference.resample(data, (50, 40, 30), mode=mode)
        out = volresample.resample(data, (50, 40, 30), mode=mode)
        max_diff = np.max(np.abs(ref - out))
        assert max_diff < ATOL, f"{len(shape)}D {mode}: max diff {max_diff:.2e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_thread_safety_cython():
    data = make_data((64, 64, 64))
    for threads in [1, 2, 4]:
        volresample.set_num_threads(threads)
        for mode in ["nearest", "linear"]:
            ref = TorchReference.resample(data, (32, 32, 32), mode=mode)
            out = volresample.resample(data, (32, 32, 32), mode=mode)
            max_diff = np.max(np.abs(ref - out))
            assert max_diff < ATOL, f"{mode} threads={threads}: {max_diff:.2e}"
    volresample.set_num_threads(4)


# ============================================================================
# Area upsampling vs torch (from issue audit)
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_area_3d_upsampling_matches_torch():
    data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    ref = TorchReference.resample(data, (3, 3, 3), mode="area")
    out = volresample.resample(data, (3, 3, 3), mode="area")
    assert np.allclose(out, ref, atol=ATOL)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_area_4d_upsampling_matches_torch():
    data = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    ref = TorchReference.resample(data, (3, 3, 3), mode="area")
    out = volresample.resample(data, (3, 3, 3), mode="area")
    assert np.allclose(out, ref, atol=ATOL)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_area_5d_upsampling_matches_torch():
    data = np.arange(32, dtype=np.float32).reshape(2, 2, 2, 2, 2)
    ref = TorchReference.resample(data, (3, 3, 3), mode="area")
    out = volresample.resample(data, (3, 3, 3), mode="area")
    assert np.allclose(out, ref, atol=ATOL)


# ============================================================================
# Linear singleton spatial dims vs torch (from issue audit)
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize(
    "spatial,out_shape",
    [
        ((1, 4, 2), (2, 3, 4)),
        ((4, 1, 3), (3, 2, 2)),
    ],
)
def test_linear_singleton_spatial_dims_match_torch(spatial, out_shape):
    rng = np.random.default_rng(10)
    data = rng.normal(size=spatial).astype(np.float32)
    ref = TorchReference.resample(data, out_shape, mode="linear")
    out = volresample.resample(data, out_shape, mode="linear")
    assert np.allclose(out, ref, atol=ATOL)


# ============================================================================
# Torch vs Cython — linear align_corners
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize(
    "input_shape,output_size",
    [
        ((64, 64, 64), (32, 32, 32)),
        ((32, 32, 32), (64, 64, 64)),
        ((128, 128, 128), (64, 64, 64)),
        ((17, 19, 23), (11, 13, 7)),
        ((32, 64, 16), (64, 32, 32)),
    ],
)
def test_3d_linear_align_corners_torch_match(input_shape, output_size):
    data = make_data(input_shape)
    ref = TorchReference.resample(data, output_size, mode="linear", align_corners=True)
    out = volresample.resample(data, output_size, mode="linear", align_corners=True)
    assert ref.shape == out.shape
    max_diff = np.max(np.abs(ref - out))
    assert max_diff < ATOL, f"max diff {max_diff:.2e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize(
    "input_shape,output_size",
    [
        ((4, 64, 64, 64), (32, 32, 32)),
        ((2, 32, 32, 32), (64, 64, 64)),
        ((8, 128, 128, 128), (64, 64, 64)),
    ],
)
def test_4d_linear_align_corners_torch_match(input_shape, output_size):
    data = make_data(input_shape)
    ref = TorchReference.resample(data, output_size, mode="linear", align_corners=True)
    out = volresample.resample(data, output_size, mode="linear", align_corners=True)
    assert ref.shape == out.shape
    max_diff = np.max(np.abs(ref - out))
    assert max_diff < ATOL, f"max diff {max_diff:.2e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize(
    "input_shape,output_size",
    [
        ((2, 4, 64, 64, 64), (32, 32, 32)),
        ((3, 2, 32, 32, 32), (64, 64, 64)),
    ],
)
def test_5d_linear_align_corners_torch_match(input_shape, output_size):
    data = make_data(input_shape)
    ref = TorchReference.resample(data, output_size, mode="linear", align_corners=True)
    out = volresample.resample(data, output_size, mode="linear", align_corners=True)
    assert ref.shape == out.shape
    max_diff = np.max(np.abs(ref - out))
    assert max_diff < ATOL, f"max diff {max_diff:.2e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_linear_align_corners_corner_preservation():
    data = make_data((8, 8, 8))
    out = volresample.resample(data, (16, 16, 16), mode="linear", align_corners=True)
    for idx in [
        (0, 0, 0),
        (-1, -1, -1),
        (0, 0, -1),
        (-1, 0, 0),
        (0, -1, 0),
        (-1, -1, 0),
        (-1, 0, -1),
        (0, -1, -1),
    ]:
        assert np.isclose(out[idx], data[idx], atol=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_linear_align_corners_identity():
    data = make_data((32, 32, 32))
    ref = TorchReference.resample(data, (32, 32, 32), mode="linear", align_corners=True)
    out = volresample.resample(data, (32, 32, 32), mode="linear", align_corners=True)
    max_diff = np.max(np.abs(ref - out))
    assert max_diff < ATOL, f"identity: {max_diff:.2e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_linear_align_corners_single_voxel():
    data_3d = np.array([[[42.0]]], dtype=np.float32)
    ref = TorchReference.resample(data_3d, (4, 4, 4), mode="linear", align_corners=True)
    out = volresample.resample(data_3d, (4, 4, 4), mode="linear", align_corners=True)
    assert np.allclose(ref, 42.0)
    max_diff = np.max(np.abs(ref - out))
    assert max_diff < ATOL


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_linear_align_corners_extreme_scales():
    # Extreme upsampling
    data_small = make_data((2, 2, 2))
    ref = TorchReference.resample(data_small, (64, 64, 64), mode="linear", align_corners=True)
    out = volresample.resample(data_small, (64, 64, 64), mode="linear", align_corners=True)
    assert np.max(np.abs(ref - out)) < ATOL

    # Extreme downsampling
    data_large = make_data((64, 64, 64))
    ref = TorchReference.resample(data_large, (2, 2, 2), mode="linear", align_corners=True)
    out = volresample.resample(data_large, (2, 2, 2), mode="linear", align_corners=True)
    assert np.max(np.abs(ref - out)) < ATOL


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_linear_align_corners_size_one_dim():
    data = make_data((16, 16, 16))
    for out_size in [(1, 16, 16), (16, 1, 16), (16, 16, 1), (1, 1, 1)]:
        ref = TorchReference.resample(data, out_size, mode="linear", align_corners=True)
        out = volresample.resample(data, out_size, mode="linear", align_corners=True)
        assert ref.shape == out.shape == out_size
        assert np.max(np.abs(ref - out)) < ATOL


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_linear_align_corners_thread_safety():
    data = make_data((64, 64, 64))
    for threads in [1, 2, 4]:
        volresample.set_num_threads(threads)
        ref = TorchReference.resample(data, (32, 32, 32), mode="linear", align_corners=True)
        out = volresample.resample(data, (32, 32, 32), mode="linear", align_corners=True)
        assert np.max(np.abs(ref - out)) < ATOL
    volresample.set_num_threads(4)


# ============================================================================
# Edge cases
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_dimension_size_one_3d(mode):
    for shape, out in [
        ((1, 16, 16), (1, 8, 8)),
        ((16, 1, 16), (8, 1, 8)),
        ((16, 16, 1), (8, 8, 1)),
    ]:
        data = make_data(shape)
        ref = TorchReference.resample(data, out, mode=mode)
        res = volresample.resample(data, out, mode=mode)
        assert ref.shape == res.shape == out
        assert np.max(np.abs(ref - res)) < ATOL


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_dimension_size_one_4d(mode):
    for shape, out in [((1, 1, 16, 16), (1, 8, 8)), ((4, 16, 1, 16), (8, 1, 8))]:
        data = make_data(shape)
        ref = TorchReference.resample(data, out, mode=mode)
        res = volresample.resample(data, out, mode=mode)
        assert ref.shape == res.shape
        assert np.max(np.abs(ref - res)) < ATOL


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_single_voxel(mode):
    data_3d = np.array([[[42.0]]], dtype=np.float32)
    ref = TorchReference.resample(data_3d, (4, 4, 4), mode=mode)
    out = volresample.resample(data_3d, (4, 4, 4), mode=mode)
    assert ref.shape == out.shape == (4, 4, 4)
    assert np.allclose(ref, 42.0)
    assert np.max(np.abs(ref - out)) < ATOL

    data_4d = np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]]], dtype=np.float32)
    ref = TorchReference.resample(data_4d, (4, 4, 4), mode=mode)
    out = volresample.resample(data_4d, (4, 4, 4), mode=mode)
    assert ref.shape == out.shape == (3, 4, 4, 4)
    assert np.max(np.abs(ref - out)) < ATOL


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["nearest", "linear", "area"])
def test_identity_resample(mode):
    for shape in [(32, 32, 32), (4, 32, 32, 32), (2, 4, 32, 32, 32)]:
        data = make_data(shape)
        ref = TorchReference.resample(data, (32, 32, 32), mode=mode)
        out = volresample.resample(data, (32, 32, 32), mode=mode)
        assert ref.shape == out.shape
        assert np.max(np.abs(ref - out)) < ATOL


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_extreme_scale_factors(mode):
    # 32x upsampling
    data_small = make_data((2, 2, 2))
    ref = TorchReference.resample(data_small, (64, 64, 64), mode=mode)
    out = volresample.resample(data_small, (64, 64, 64), mode=mode)
    assert np.max(np.abs(ref - out)) < ATOL

    # 32x downsampling
    data_large = make_data((64, 64, 64))
    ref = TorchReference.resample(data_large, (2, 2, 2), mode=mode)
    out = volresample.resample(data_large, (2, 2, 2), mode=mode)
    assert np.max(np.abs(ref - out)) < ATOL


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["nearest", "linear", "area"])
def test_prime_number_dimensions(mode):
    for shape in [(17, 19, 23), (3, 17, 19, 23)]:
        data = make_data(shape)
        ref = TorchReference.resample(data, (11, 13, 7), mode=mode)
        out = volresample.resample(data, (11, 13, 7), mode=mode)
        assert ref.shape == out.shape
        assert np.max(np.abs(ref - out)) < ATOL


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_asymmetric_scaling(mode):
    for shape in [(32, 64, 16), (2, 32, 64, 16)]:
        data = make_data(shape)
        ref = TorchReference.resample(data, (64, 32, 32), mode=mode)
        out = volresample.resample(data, (64, 32, 32), mode=mode)
        assert ref.shape == out.shape
        assert np.max(np.abs(ref - out)) < ATOL


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_constant_value_arrays():
    for mode in ["nearest", "linear", "area"]:
        for val in [0.0, 1.0, 3.14159]:
            data = np.full((16, 16, 16), val, dtype=np.float32)
            out = volresample.resample(data, (8, 8, 8), mode=mode)
            assert np.allclose(out, val, rtol=1e-5), f"{mode}: constant {val}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_single_channel_4d():
    data_3d = make_data((32, 32, 32))
    data_4d = data_3d.reshape(1, 32, 32, 32)
    for mode in ["nearest", "linear", "area"]:
        r3 = volresample.resample(data_3d, (16, 16, 16), mode=mode)
        r4 = volresample.resample(data_4d, (16, 16, 16), mode=mode)
        assert r4.shape == (1, 16, 16, 16)
        assert np.max(np.abs(r3 - r4.squeeze(0))) < 1e-7


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_output_to_size_one(mode):
    data = make_data((16, 16, 16))
    for out_size in [(1, 16, 16), (16, 1, 16), (16, 16, 1), (1, 1, 1)]:
        ref = TorchReference.resample(data, out_size, mode=mode)
        out = volresample.resample(data, out_size, mode=mode)
        assert ref.shape == out.shape == out_size
        assert np.max(np.abs(ref - out)) < ATOL


# ============================================================================
# Memory layout tests
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_fortran_contiguous_nearest():
    data = np.asfortranarray(np.arange(27, dtype=np.float32).reshape(3, 3, 3))
    ref = TorchReference.resample(data, (2, 2, 2), mode="nearest")
    out = volresample.resample(data, (2, 2, 2), mode="nearest")
    assert np.max(np.abs(ref - out)) < ATOL


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_fortran_contiguous_linear():
    data = np.asfortranarray(np.random.randn(16, 16, 16).astype(np.float32))
    ref = TorchReference.resample(data, (8, 8, 8), mode="linear")
    out = volresample.resample(data, (8, 8, 8), mode="linear")
    assert np.max(np.abs(ref - out)) < ATOL


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_non_contiguous_sliced():
    data = np.random.randn(32, 32, 32).astype(np.float32)[::2, ::2, ::2]
    assert not data.flags["C_CONTIGUOUS"]
    ref = TorchReference.resample(data, (8, 8, 8), mode="linear")
    out = volresample.resample(data, (8, 8, 8), mode="linear")
    assert np.max(np.abs(ref - out)) < ATOL


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_transposed_array():
    data = np.random.randn(16, 16, 16).astype(np.float32).T
    assert not data.flags["C_CONTIGUOUS"]
    ref = TorchReference.resample(data, (8, 8, 8), mode="linear")
    out = volresample.resample(data, (8, 8, 8), mode="linear")
    assert np.max(np.abs(ref - out)) < ATOL


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_4d_fortran_contiguous():
    data = np.asfortranarray(np.random.randn(4, 16, 16, 16).astype(np.float32))
    ref = TorchReference.resample(data, (8, 8, 8), mode="linear")
    out = volresample.resample(data, (8, 8, 8), mode="linear")
    assert np.max(np.abs(ref - out)) < ATOL


# ============================================================================
# Area mode — mixed up/down scaling
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_area_mixed_up_down_scaling():
    np.random.seed(42)
    data = np.random.randn(8, 16, 32).astype(np.float32)
    ref = TorchReference.resample(data, (16, 8, 64), mode="area")
    out = volresample.resample(data, (16, 8, 64), mode="area")
    assert np.max(np.abs(ref - out)) < ATOL


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_area_one_dim_upsampling():
    np.random.seed(42)
    data = np.random.randn(16, 16, 16).astype(np.float32)
    ref = TorchReference.resample(data, (8, 8, 32), mode="area")
    out = volresample.resample(data, (8, 8, 32), mode="area")
    assert np.max(np.abs(ref - out)) < ATOL


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_area_two_dims_down_one_up():
    np.random.seed(42)
    data = np.random.randn(16, 16, 8).astype(np.float32)
    ref = TorchReference.resample(data, (8, 8, 16), mode="area")
    out = volresample.resample(data, (8, 8, 16), mode="area")
    assert np.max(np.abs(ref - out)) < ATOL


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_area_4d_mixed_scaling():
    np.random.seed(42)
    data = np.random.randn(4, 8, 16, 8).astype(np.float32)
    ref = TorchReference.resample(data, (16, 8, 16), mode="area")
    out = volresample.resample(data, (16, 8, 16), mode="area")
    assert np.max(np.abs(ref - out)) < ATOL
