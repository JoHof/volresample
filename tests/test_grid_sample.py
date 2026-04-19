"""Unit tests for 3D grid_sample implementation.

This module tests the volresample grid_sample implementation against PyTorch's
torch.nn.functional.grid_sample to verify correctness for 3D volumes.

volresample only supports 3D grid sampling with 5D input (N, C, D, H, W):
- N: batch size
- C: number of channels
- D, H, W: depth, height, width (spatial dimensions)

Tests cover:
- All interpolation modes: linear (trilinear) and nearest neighbor
- All padding modes: zeros, border, reflection
- Various grid configurations: identity, translation, scaling, rotation
- Edge cases: out-of-bound coordinates, small inputs, non-uniform sizes
- Exact match validation against PyTorch (max diff < 2e-5)
"""

import numpy as np
import pytest

import volresample

# Try to import torch
try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Tolerance for comparing results
TOLERANCE = 2e-5  # Slightly relaxed for floating point precision


# Mode mapping: volresample uses 'linear', PyTorch uses 'bilinear'
def torch_mode(mode):
    """Convert volresample mode to PyTorch mode."""
    return "bilinear" if mode == "linear" else mode


def generate_test_data(shape, seed=42):
    """Generate random test data with specified shape."""
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32)


def create_identity_grid_3d(N, D_out, H_out, W_out):
    """Create an identity grid for 3D (5D input) - grid that maps to same positions."""
    z = np.linspace(-1, 1, D_out, dtype=np.float32)
    y = np.linspace(-1, 1, H_out, dtype=np.float32)
    x = np.linspace(-1, 1, W_out, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    grid = np.stack([xx, yy, zz], axis=-1)  # Shape: (D_out, H_out, W_out, 3)
    grid = np.broadcast_to(grid, (N, D_out, H_out, W_out, 3)).copy()
    return grid


def create_random_grid_3d(N, D_out, H_out, W_out, seed=123, range_min=-1.5, range_max=1.5):
    """Create a random grid for 3D testing (may have out-of-bounds values)."""
    rng = np.random.RandomState(seed)
    grid = rng.uniform(range_min, range_max, (N, D_out, H_out, W_out, 3)).astype(np.float32)
    return grid


# =============================================================================
# Basic Functionality Tests (no torch dependency)
# =============================================================================


def test_grid_sample_3d_basic_shape():
    """Test basic 3D grid sample output shape."""
    input_data = generate_test_data((2, 3, 8, 8, 8))  # N=2, C=3, D=8, H=8, W=8
    grid = create_identity_grid_3d(2, 4, 4, 4)  # Output D=4, H=4, W=4

    output = volresample.grid_sample(input_data, grid)
    assert output.shape == (2, 3, 4, 4, 4), f"Expected (2, 3, 4, 4, 4), got {output.shape}"


def test_grid_sample_3d_modes():
    """Test that different modes produce outputs for 3D."""
    input_data = generate_test_data((1, 1, 4, 4, 4))
    grid = create_identity_grid_3d(1, 2, 2, 2)

    for mode in ["linear", "nearest"]:
        for padding in ["zeros", "border", "reflection"]:
            output = volresample.grid_sample(input_data, grid, mode=mode, padding_mode=padding)
            assert output.shape == (1, 1, 2, 2, 2)
            assert not np.isnan(output).any(), f"NaN in output for mode={mode}, padding={padding}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["linear", "nearest"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_3d_torch_match_identity_grid(mode, padding_mode):
    """Test 3D grid sample with identity grid matches PyTorch."""
    input_data = generate_test_data((2, 3, 8, 8, 8))
    grid = create_identity_grid_3d(2, 8, 8, 8)

    # PyTorch
    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch,
        grid_torch,
        mode=torch_mode(mode),
        padding_mode=padding_mode,
        align_corners=False,
    ).numpy()

    # Cython
    cython_output = volresample.grid_sample(input_data, grid, mode=mode, padding_mode=padding_mode)

    max_diff = np.max(np.abs(torch_output - cython_output))
    assert (
        max_diff < TOLERANCE
    ), f"3D identity grid, mode={mode}, padding={padding_mode}: max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["linear", "nearest"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_3d_torch_match_random_grid(mode, padding_mode):
    """Test 3D grid sample with random grid (including out-of-bounds) matches PyTorch."""
    input_data = generate_test_data((2, 3, 8, 8, 8))
    grid = create_random_grid_3d(2, 6, 6, 6, range_min=-1.5, range_max=1.5)

    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch,
        grid_torch,
        mode=torch_mode(mode),
        padding_mode=padding_mode,
        align_corners=False,
    ).numpy()

    cython_output = volresample.grid_sample(input_data, grid, mode=mode, padding_mode=padding_mode)

    max_diff = np.max(np.abs(torch_output - cython_output))
    assert (
        max_diff < TOLERANCE
    ), f"3D random grid, mode={mode}, padding={padding_mode}: max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["linear", "nearest"])
def test_3d_torch_match_upsampling(mode):
    """Test 3D grid sample for upsampling (larger output than input)."""
    input_data = generate_test_data((1, 2, 4, 4, 4))
    grid = create_identity_grid_3d(1, 8, 8, 8)  # 2x upsampling

    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode=torch_mode(mode), padding_mode="zeros", align_corners=False
    ).numpy()

    cython_output = volresample.grid_sample(input_data, grid, mode=mode, padding_mode="zeros")

    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, f"3D upsampling, mode={mode}: max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["linear", "nearest"])
def test_3d_torch_match_downsampling(mode):
    """Test 3D grid sample for downsampling (smaller output than input)."""
    input_data = generate_test_data((1, 2, 8, 8, 8))
    grid = create_identity_grid_3d(1, 4, 4, 4)  # 2x downsampling

    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode=torch_mode(mode), padding_mode="zeros", align_corners=False
    ).numpy()

    cython_output = volresample.grid_sample(input_data, grid, mode=mode, padding_mode="zeros")

    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, f"3D downsampling, mode={mode}: max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["linear", "nearest"])
def test_3d_torch_match_non_uniform_sizes(mode):
    """Test 3D grid sample with non-uniform input and output sizes."""
    input_data = generate_test_data((2, 3, 7, 9, 11))  # Non-cubic input
    grid = create_identity_grid_3d(2, 5, 6, 8)  # Non-cubic output

    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode=torch_mode(mode), padding_mode="zeros", align_corners=False
    ).numpy()

    cython_output = volresample.grid_sample(input_data, grid, mode=mode, padding_mode="zeros")

    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, f"3D non-uniform, mode={mode}: max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["linear", "nearest"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_3d_torch_match_extreme_out_of_bounds(mode, padding_mode):
    """Test 3D grid sample with extreme out-of-bounds coordinates."""
    input_data = generate_test_data((1, 1, 4, 4, 4))
    grid = create_random_grid_3d(1, 3, 3, 3, range_min=-5.0, range_max=5.0)

    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch,
        grid_torch,
        mode=torch_mode(mode),
        padding_mode=padding_mode,
        align_corners=False,
    ).numpy()

    cython_output = volresample.grid_sample(input_data, grid, mode=mode, padding_mode=padding_mode)

    max_diff = np.max(np.abs(torch_output - cython_output))
    assert (
        max_diff < TOLERANCE
    ), f"3D extreme OOB, mode={mode}, padding={padding_mode}: max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["linear", "nearest"])
def test_3d_torch_match_small_input(mode):
    """Test 3D grid sample with very small input."""
    input_data = generate_test_data((1, 1, 2, 2, 2))  # 2x2x2 input
    grid = create_identity_grid_3d(1, 4, 4, 4)

    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode=torch_mode(mode), padding_mode="zeros", align_corners=False
    ).numpy()

    cython_output = volresample.grid_sample(input_data, grid, mode=mode, padding_mode="zeros")

    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, f"3D small input, mode={mode}: max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_3d_torch_match_constant_input():
    """Test that constant input remains constant after 3D grid sampling."""
    input_data = np.full((1, 1, 4, 4, 4), 42.0, dtype=np.float32)
    grid = create_random_grid_3d(1, 3, 3, 3, range_min=-0.9, range_max=0.9)

    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode="bilinear", padding_mode="border", align_corners=False
    ).numpy()

    cython_output = volresample.grid_sample(
        input_data, grid, mode="bilinear", padding_mode="border"
    )

    assert np.allclose(cython_output, 42.0, rtol=1e-5), "Constant should remain constant"
    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, f"3D constant input: max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize(
    "input_shape,grid_shape",
    [
        ((2, 3, 8, 8, 8), (2, 12, 12, 12, 3)),  # Upsampling
        ((1, 4, 16, 16, 16), (1, 8, 8, 8, 3)),  # Downsampling
    ],
)
def test_torch_match_various_sizes(input_shape, grid_shape):
    """Test grid sample with various input and output sizes (3D only)."""
    input_data = generate_test_data(input_shape)

    # Only 3D configurations
    grid = create_random_grid_3d(
        input_shape[0], grid_shape[1], grid_shape[2], grid_shape[3], range_min=-1.0, range_max=1.0
    )

    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode="bilinear", padding_mode="zeros", align_corners=False
    ).numpy()

    cython_output = volresample.grid_sample(input_data, grid, mode="bilinear", padding_mode="zeros")

    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, f"Shape {input_shape}->{grid_shape}: max diff {max_diff:.6e}"


# =============================================================================
# Edge Cases and Corner Cases
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_3d_grid_at_corners():
    """Test 3D grid sample at exact corner positions."""
    input_data = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 2, 2)
    # Grid pointing exactly at corners (x, y, z)
    grid = np.array(
        [
            [
                [[[-1, -1, -1], [1, -1, -1]], [[-1, 1, -1], [1, 1, -1]]],
                [[[-1, -1, 1], [1, -1, 1]], [[-1, 1, 1], [1, 1, 1]]],
            ]
        ],
        dtype=np.float32,
    )

    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode="bilinear", padding_mode="zeros", align_corners=False
    ).numpy()

    cython_output = volresample.grid_sample(input_data, grid, mode="bilinear", padding_mode="zeros")

    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, f"3D corners: max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_3d_single_voxel_input():
    """Test 3D grid sample with 1x1x1 input."""
    input_data = np.array([[[[[42.0]]]]], dtype=np.float32)  # 1x1x1 input
    grid = create_identity_grid_3d(1, 2, 2, 2)

    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode="bilinear", padding_mode="border", align_corners=False
    ).numpy()

    cython_output = volresample.grid_sample(
        input_data, grid, mode="bilinear", padding_mode="border"
    )

    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, f"3D 1x1x1 input: max diff {max_diff:.6e}"


# =============================================================================
# Memory Layout Tests (non-C-contiguous arrays)
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_fortran_contiguous_input():
    """Fortran-contiguous input array should give correct result."""
    input_c = np.random.randn(1, 2, 8, 8, 8).astype(np.float32)
    input_f = np.asfortranarray(input_c.copy())
    grid = create_identity_grid_3d(1, 4, 4, 4)

    assert not input_f.flags["C_CONTIGUOUS"]

    input_torch = torch.from_numpy(input_f)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode="bilinear", padding_mode="zeros", align_corners=False
    ).numpy()

    cython_output = volresample.grid_sample(input_f, grid)

    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, f"Fortran input: max_diff={max_diff}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_fortran_contiguous_grid():
    """Fortran-contiguous grid array should give correct result."""
    input_data = np.random.randn(1, 2, 8, 8, 8).astype(np.float32)
    grid_c = create_identity_grid_3d(1, 4, 4, 4)
    grid_f = np.asfortranarray(grid_c.copy())

    assert not grid_f.flags["C_CONTIGUOUS"]

    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid_f)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode="bilinear", padding_mode="zeros", align_corners=False
    ).numpy()

    cython_output = volresample.grid_sample(input_data, grid_f)

    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, f"Fortran grid: max_diff={max_diff}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_sliced_non_contiguous_input():
    """Non-contiguous sliced input should give correct result."""
    input_full = np.random.randn(2, 4, 16, 16, 16).astype(np.float32)
    input_sliced = input_full[::2, ::2, ::2, ::2, ::2]
    grid = create_identity_grid_3d(1, 4, 4, 4)

    assert not input_sliced.flags["C_CONTIGUOUS"]

    input_torch = torch.from_numpy(input_sliced)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode="bilinear", padding_mode="zeros", align_corners=False
    ).numpy()

    cython_output = volresample.grid_sample(input_sliced, grid)

    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, f"Sliced input: max_diff={max_diff}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_both_non_contiguous():
    """Both input and grid non-contiguous should give correct result."""
    input_c = np.random.randn(1, 2, 8, 8, 8).astype(np.float32)
    input_f = np.asfortranarray(input_c.copy())
    grid_c = create_identity_grid_3d(1, 4, 4, 4)
    grid_f = np.asfortranarray(grid_c.copy())

    input_torch = torch.from_numpy(input_f)
    grid_torch = torch.from_numpy(grid_f)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode="bilinear", padding_mode="zeros", align_corners=False
    ).numpy()

    cython_output = volresample.grid_sample(input_f, grid_f)

    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, f"Both non-contiguous: max_diff={max_diff}"


# =============================================================================
# Singleton spatial dimensions (from issue audit)
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["nearest", "linear"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_grid_sample_singleton_spatial_dims_match_torch(mode, padding_mode):
    rng = np.random.default_rng(11)
    inp = rng.normal(size=(1, 2, 1, 2, 3)).astype(np.float32)
    grid = rng.uniform(-1.25, 1.25, size=(1, 3, 2, 4, 3)).astype(np.float32)
    ref = F.grid_sample(
        torch.from_numpy(inp),
        torch.from_numpy(grid),
        mode=torch_mode(mode),
        padding_mode=padding_mode,
        align_corners=False,
    ).numpy()
    out = volresample.grid_sample(inp, grid, mode=mode, padding_mode=padding_mode)
    atol = 2e-5 if mode == "linear" else 0.0
    assert np.allclose(out, ref, atol=atol)


# =============================================================================
# Native dtype tests for grid_sample nearest mode (uint8, int16)
# =============================================================================


class TestGridSampleNearestDtype:
    """Test native uint8 and int16 support for grid_sample with nearest mode."""

    def test_nearest_uint8_preserves_dtype(self):
        """grid_sample nearest should preserve uint8 dtype."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=(1, 2, 4, 4, 4), dtype=np.uint8)
        grid = create_identity_grid_3d(1, 4, 4, 4)
        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode="zeros")
        assert result.dtype == np.uint8
        assert result.shape == (1, 2, 4, 4, 4)

    def test_nearest_int16_preserves_dtype(self):
        """grid_sample nearest should preserve int16 dtype."""
        rng = np.random.default_rng(42)
        data = rng.integers(-32768, 32768, size=(1, 2, 4, 4, 4), dtype=np.int16)
        grid = create_identity_grid_3d(1, 4, 4, 4)
        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode="zeros")
        assert result.dtype == np.int16
        assert result.shape == (1, 2, 4, 4, 4)

    def test_nearest_float32_preserves_dtype(self):
        """grid_sample nearest should preserve float32 dtype."""
        data = generate_test_data((1, 2, 4, 4, 4))
        grid = create_identity_grid_3d(1, 4, 4, 4)
        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode="zeros")
        assert result.dtype == np.float32

    @pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
    def test_nearest_uint8_all_padding_modes(self, padding_mode):
        """uint8 grid_sample nearest should work with all padding modes."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=(1, 2, 4, 4, 4), dtype=np.uint8)
        grid = create_random_grid_3d(1, 3, 3, 3, range_min=-1.5, range_max=1.5)
        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode=padding_mode)
        assert result.dtype == np.uint8
        assert result.shape == (1, 2, 3, 3, 3)
        assert not np.isnan(result.astype(np.float32)).any()

    @pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
    def test_nearest_int16_all_padding_modes(self, padding_mode):
        """int16 grid_sample nearest should work with all padding modes."""
        rng = np.random.default_rng(42)
        data = rng.integers(-32768, 32768, size=(1, 2, 4, 4, 4), dtype=np.int16)
        grid = create_random_grid_3d(1, 3, 3, 3, range_min=-1.5, range_max=1.5)
        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode=padding_mode)
        assert result.dtype == np.int16
        assert result.shape == (1, 2, 3, 3, 3)

    def test_nearest_uint8_zeros_padding_oob(self):
        """uint8 nearest with zeros padding should return 0 for out-of-bounds."""
        data = np.full((1, 1, 2, 2, 2), 200, dtype=np.uint8)
        # Grid completely out of bounds
        grid = np.full((1, 1, 1, 1, 3), 5.0, dtype=np.float32)
        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode="zeros")
        assert result.dtype == np.uint8
        assert result[0, 0, 0, 0, 0] == 0

    def test_nearest_int16_zeros_padding_oob(self):
        """int16 nearest with zeros padding should return 0 for out-of-bounds."""
        data = np.full((1, 1, 2, 2, 2), -1000, dtype=np.int16)
        grid = np.full((1, 1, 1, 1, 3), 5.0, dtype=np.float32)
        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode="zeros")
        assert result.dtype == np.int16
        assert result[0, 0, 0, 0, 0] == 0

    def test_nearest_uint8_boundary_values(self):
        """uint8 grid_sample should handle boundary values (0 and 255)."""
        data = np.array([[[[[0, 255], [128, 64]], [[32, 16], [8, 4]]]]], dtype=np.uint8)
        grid = create_identity_grid_3d(1, 2, 2, 2)
        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode="border")
        assert result.dtype == np.uint8
        # Values should come from the original data
        unique_vals = set(result.flatten())
        assert unique_vals.issubset({0, 4, 8, 16, 32, 64, 128, 255})

    def test_nearest_int16_boundary_values(self):
        """int16 grid_sample should handle boundary values."""
        data = np.array([[[[[-32768, 32767], [0, 1000]], [[-1, 1], [100, -100]]]]], dtype=np.int16)
        grid = create_identity_grid_3d(1, 2, 2, 2)
        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode="border")
        assert result.dtype == np.int16

    def test_nearest_uint8_multichannel(self):
        """uint8 grid_sample should work with multiple channels."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=(2, 4, 8, 8, 8), dtype=np.uint8)
        grid = create_identity_grid_3d(2, 4, 4, 4)
        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode="zeros")
        assert result.dtype == np.uint8
        assert result.shape == (2, 4, 4, 4, 4)

    def test_nearest_int16_multichannel(self):
        """int16 grid_sample should work with multiple channels."""
        rng = np.random.default_rng(42)
        data = rng.integers(-32768, 32768, size=(2, 4, 8, 8, 8), dtype=np.int16)
        grid = create_identity_grid_3d(2, 4, 4, 4)
        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode="zeros")
        assert result.dtype == np.int16
        assert result.shape == (2, 4, 4, 4, 4)

    def test_linear_uint8_converts_to_float32(self):
        """Linear mode should convert uint8 input to float32."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=(1, 1, 4, 4, 4), dtype=np.uint8)
        grid = create_identity_grid_3d(1, 2, 2, 2)
        result = volresample.grid_sample(data, grid, mode="linear", padding_mode="zeros")
        assert result.dtype == np.float32

    def test_linear_int16_converts_to_float32(self):
        """Linear mode should convert int16 input to float32."""
        rng = np.random.default_rng(42)
        data = rng.integers(-32768, 32768, size=(1, 1, 4, 4, 4), dtype=np.int16)
        grid = create_identity_grid_3d(1, 2, 2, 2)
        result = volresample.grid_sample(data, grid, mode="linear", padding_mode="zeros")
        assert result.dtype == np.float32

    @pytest.mark.parametrize("dtype", [np.uint8, np.int16])
    @pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
    def test_native_matches_float32_roundtrip(self, dtype, padding_mode):
        """Native dtype path should produce identical results to float32 roundtrip."""
        rng = np.random.default_rng(99)
        data = (
            rng.integers(0, 256, size=(2, 3, 12, 12, 12), dtype=dtype)
            if dtype == np.uint8
            else rng.integers(-32768, 32768, size=(2, 3, 12, 12, 12), dtype=dtype)
        )
        grid = create_random_grid_3d(2, 8, 8, 8, range_min=-1.5, range_max=1.5)

        # Native path
        result_native = volresample.grid_sample(
            data, grid, mode="nearest", padding_mode=padding_mode
        )
        # Float32 roundtrip path
        result_f32 = volresample.grid_sample(
            data.astype(np.float32), grid, mode="nearest", padding_mode=padding_mode
        )

        assert result_native.dtype == dtype
        assert result_f32.dtype == np.float32
        np.testing.assert_array_equal(result_native, result_f32.astype(dtype))

    @pytest.mark.parametrize("dtype", [np.uint8, np.int16])
    def test_no_hidden_float32_allocation(self, dtype):
        """Verify that native dtype path does not allocate float32 temporaries."""
        import tracemalloc

        rng = np.random.default_rng(42)
        data = (
            rng.integers(0, 256, size=(1, 2, 32, 32, 32), dtype=dtype)
            if dtype == np.uint8
            else rng.integers(-32768, 32768, size=(1, 2, 32, 32, 32), dtype=dtype)
        )
        grid = create_random_grid_3d(1, 24, 24, 24, range_min=-1.0, range_max=1.0)

        # Expected output size in bytes (no float32 overhead)
        expected_output_bytes = 1 * 2 * 24 * 24 * 24 * np.dtype(dtype).itemsize

        tracemalloc.start()
        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode="zeros")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert result.dtype == dtype
        # Peak should be well under what a float32 conversion would require.
        # A float32 copy of input (1*2*32*32*32*4 = 262144) plus float32 output
        # (1*2*24*24*24*4 = 110592) = 372736 bytes.
        # Native: just the output array.
        float32_input_copy_bytes = data.size * 4
        assert peak < float32_input_copy_bytes, (
            f"Peak memory {peak} suggests hidden float32 conversion "
            f"(float32 input copy would be {float32_input_copy_bytes})"
        )

    @pytest.mark.parametrize("dtype", [np.uint8, np.int16])
    def test_fortran_contiguous_input(self, dtype):
        """Non-contiguous (Fortran-order) input should be handled correctly."""
        rng = np.random.default_rng(42)
        data_c = (
            rng.integers(0, 256, size=(1, 2, 6, 6, 6), dtype=dtype)
            if dtype == np.uint8
            else rng.integers(-32768, 32768, size=(1, 2, 6, 6, 6), dtype=dtype)
        )
        data_f = np.asfortranarray(data_c)
        grid = create_identity_grid_3d(1, 4, 4, 4)

        result_c = volresample.grid_sample(data_c, grid, mode="nearest", padding_mode="border")
        result_f = volresample.grid_sample(data_f, grid, mode="nearest", padding_mode="border")

        assert result_c.dtype == dtype
        assert result_f.dtype == dtype
        np.testing.assert_array_equal(result_c, result_f)

    @pytest.mark.parametrize("dtype", [np.uint8, np.int16])
    def test_larger_volume(self, dtype):
        """Native dtype should handle larger volumes correctly."""
        rng = np.random.default_rng(77)
        data = (
            rng.integers(0, 256, size=(2, 4, 32, 32, 32), dtype=dtype)
            if dtype == np.uint8
            else rng.integers(-32768, 32768, size=(2, 4, 32, 32, 32), dtype=dtype)
        )
        grid = create_random_grid_3d(2, 24, 24, 24, range_min=-1.2, range_max=1.2)

        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode="zeros")
        assert result.dtype == dtype
        assert result.shape == (2, 4, 24, 24, 24)

        # Verify via float32 roundtrip
        result_f32 = volresample.grid_sample(
            data.astype(np.float32), grid, mode="nearest", padding_mode="zeros"
        )
        np.testing.assert_array_equal(result, result_f32.astype(dtype))

    def test_uint8_known_values(self):
        """Verify exact values with manually constructed data and grid."""
        # 1x1x2x2x2 input with known values
        data = np.array([[[[[10, 20], [30, 40]], [[50, 60], [70, 80]]]]], dtype=np.uint8)
        # Grid pointing to center of each voxel (normalized coords)
        # For a 2-voxel dimension: centers at -0.5 and 0.5 in unnormalized,
        # which are -0.5 and 0.5 in normalized (for align_corners=False with size=2)
        grid = create_identity_grid_3d(1, 2, 2, 2)
        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode="border")
        assert result.dtype == np.uint8
        # With identity grid and border padding, should match float32 path exactly
        result_f32 = volresample.grid_sample(
            data.astype(np.float32), grid, mode="nearest", padding_mode="border"
        )
        np.testing.assert_array_equal(result, result_f32.astype(np.uint8))

    def test_int16_known_values(self):
        """Verify exact values with manually constructed int16 data."""
        data = np.array(
            [[[[[-1000, 1000], [500, -500]], [[0, 32767], [-32768, 100]]]]], dtype=np.int16
        )
        grid = create_identity_grid_3d(1, 2, 2, 2)
        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode="border")
        assert result.dtype == np.int16
        result_f32 = volresample.grid_sample(
            data.astype(np.float32), grid, mode="nearest", padding_mode="border"
        )
        np.testing.assert_array_equal(result, result_f32.astype(np.int16))

    @pytest.mark.parametrize("dtype", [np.uint8, np.int16])
    def test_singleton_spatial_dims(self, dtype):
        """Native dtype with singleton spatial dimensions (size=1)."""
        rng = np.random.default_rng(42)
        data = (
            rng.integers(0, 256, size=(1, 1, 1, 1, 4), dtype=dtype)
            if dtype == np.uint8
            else rng.integers(-32768, 32768, size=(1, 1, 1, 1, 4), dtype=dtype)
        )
        grid = create_random_grid_3d(1, 1, 1, 3, range_min=-1.0, range_max=1.0)
        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode="border")
        assert result.dtype == dtype
        assert result.shape == (1, 1, 1, 1, 3)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGridSampleNearestDtypeTorchParity:
    """Test that native dtype grid_sample matches PyTorch (via float32 round-trip)."""

    @pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
    def test_uint8_matches_torch_float32_nearest(self, padding_mode):
        """uint8 nearest grid_sample should match PyTorch float32 nearest."""
        rng = np.random.default_rng(42)
        data_u8 = rng.integers(0, 256, size=(2, 3, 8, 8, 8), dtype=np.uint8)
        grid = create_random_grid_3d(2, 6, 6, 6, range_min=-1.5, range_max=1.5)

        # PyTorch reference (always uses float32)
        data_f32 = data_u8.astype(np.float32)
        input_torch = torch.from_numpy(data_f32)
        grid_torch = torch.from_numpy(grid)
        torch_output = F.grid_sample(
            input_torch,
            grid_torch,
            mode="nearest",
            padding_mode=padding_mode,
            align_corners=False,
        ).numpy()

        # volresample with native uint8
        result = volresample.grid_sample(data_u8, grid, mode="nearest", padding_mode=padding_mode)
        assert result.dtype == np.uint8
        # Compare: torch output is float32, cast to uint8 for comparison
        np.testing.assert_array_equal(result, torch_output.astype(np.uint8))

    @pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
    def test_int16_matches_torch_float32_nearest(self, padding_mode):
        """int16 nearest grid_sample should match PyTorch float32 nearest."""
        rng = np.random.default_rng(42)
        data_i16 = rng.integers(-32768, 32768, size=(2, 3, 8, 8, 8), dtype=np.int16)
        grid = create_random_grid_3d(2, 6, 6, 6, range_min=-1.5, range_max=1.5)

        # PyTorch reference
        data_f32 = data_i16.astype(np.float32)
        input_torch = torch.from_numpy(data_f32)
        grid_torch = torch.from_numpy(grid)
        torch_output = F.grid_sample(
            input_torch,
            grid_torch,
            mode="nearest",
            padding_mode=padding_mode,
            align_corners=False,
        ).numpy()

        # volresample with native int16
        result = volresample.grid_sample(data_i16, grid, mode="nearest", padding_mode=padding_mode)
        assert result.dtype == np.int16
        np.testing.assert_array_equal(result, torch_output.astype(np.int16))

    def test_uint8_identity_grid_exact(self):
        """uint8 with identity grid should produce exact values from input."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=(1, 2, 8, 8, 8), dtype=np.uint8)
        grid = create_identity_grid_3d(1, 8, 8, 8)

        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode="border")

        # With identity grid and nearest, output should match input (via float32 reference)
        data_f32 = data.astype(np.float32)
        ref = F.grid_sample(
            torch.from_numpy(data_f32),
            torch.from_numpy(grid),
            mode="nearest",
            padding_mode="border",
            align_corners=False,
        ).numpy()
        np.testing.assert_array_equal(result, ref.astype(np.uint8))

    def test_int16_identity_grid_exact(self):
        """int16 with identity grid should produce exact values from input."""
        rng = np.random.default_rng(42)
        data = rng.integers(-32768, 32768, size=(1, 2, 8, 8, 8), dtype=np.int16)
        grid = create_identity_grid_3d(1, 8, 8, 8)

        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode="border")

        data_f32 = data.astype(np.float32)
        ref = F.grid_sample(
            torch.from_numpy(data_f32),
            torch.from_numpy(grid),
            mode="nearest",
            padding_mode="border",
            align_corners=False,
        ).numpy()
        np.testing.assert_array_equal(result, ref.astype(np.int16))

    def test_uint8_non_uniform_sizes(self):
        """uint8 nearest with non-uniform spatial sizes."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=(2, 3, 7, 9, 11), dtype=np.uint8)
        grid = create_identity_grid_3d(2, 5, 6, 8)

        data_f32 = data.astype(np.float32)
        ref = F.grid_sample(
            torch.from_numpy(data_f32),
            torch.from_numpy(grid),
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).numpy()

        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode="zeros")
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, ref.astype(np.uint8))

    def test_int16_extreme_out_of_bounds(self):
        """int16 nearest with extreme out-of-bounds coordinates."""
        rng = np.random.default_rng(42)
        data = rng.integers(-32768, 32768, size=(1, 1, 4, 4, 4), dtype=np.int16)
        grid = create_random_grid_3d(1, 3, 3, 3, range_min=-5.0, range_max=5.0)

        for padding_mode in ["zeros", "border", "reflection"]:
            data_f32 = data.astype(np.float32)
            ref = F.grid_sample(
                torch.from_numpy(data_f32),
                torch.from_numpy(grid),
                mode="nearest",
                padding_mode=padding_mode,
                align_corners=False,
            ).numpy()

            result = volresample.grid_sample(data, grid, mode="nearest", padding_mode=padding_mode)
            assert result.dtype == np.int16
            np.testing.assert_array_equal(result, ref.astype(np.int16))

    @pytest.mark.parametrize("dtype", [np.uint8, np.int16])
    @pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
    def test_larger_volume_torch_parity(self, dtype, padding_mode):
        """Native dtype on larger volumes should match PyTorch exactly."""
        rng = np.random.default_rng(55)
        data = (
            rng.integers(0, 256, size=(2, 4, 24, 24, 24), dtype=dtype)
            if dtype == np.uint8
            else rng.integers(-32768, 32768, size=(2, 4, 24, 24, 24), dtype=dtype)
        )
        grid = create_random_grid_3d(2, 16, 16, 16, range_min=-1.3, range_max=1.3)

        data_f32 = data.astype(np.float32)
        ref = F.grid_sample(
            torch.from_numpy(data_f32),
            torch.from_numpy(grid),
            mode="nearest",
            padding_mode=padding_mode,
            align_corners=False,
        ).numpy()

        result = volresample.grid_sample(data, grid, mode="nearest", padding_mode=padding_mode)
        assert result.dtype == dtype
        np.testing.assert_array_equal(result, ref.astype(dtype))
