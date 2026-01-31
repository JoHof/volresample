"""Unit tests for 3D grid_sample implementation.

This module tests the volresample grid_sample implementation against PyTorch's
torch.nn.functional.grid_sample to verify correctness for 3D volumes.

volresample only supports 3D grid sampling with 5D input (N, C, D, H, W):
- N: batch size
- C: number of channels
- D, H, W: depth, height, width (spatial dimensions)

Tests cover:
- All interpolation modes: bilinear/trilinear and nearest neighbor
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


def generate_test_data(shape, seed=42):
    """Generate random test data with specified shape."""
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32)


def create_identity_grid_3d(N, D_out, H_out, W_out):
    """Create an identity grid for 3D (5D input) - grid that maps to same positions."""
    z = np.linspace(-1, 1, D_out, dtype=np.float32)
    y = np.linspace(-1, 1, H_out, dtype=np.float32)
    x = np.linspace(-1, 1, W_out, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
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
    
    for mode in ["bilinear", "nearest"]:
        for padding in ["zeros", "border", "reflection"]:
            output = volresample.grid_sample(
                input_data, grid, mode=mode, padding_mode=padding
            )
            assert output.shape == (1, 1, 2, 2, 2)
            assert not np.isnan(output).any(), f"NaN in output for mode={mode}, padding={padding}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_3d_torch_match_identity_grid(mode, padding_mode):
    """Test 3D grid sample with identity grid matches PyTorch."""
    input_data = generate_test_data((2, 3, 8, 8, 8))
    grid = create_identity_grid_3d(2, 8, 8, 8)
    
    # PyTorch
    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode=mode, padding_mode=padding_mode, align_corners=False
    ).numpy()
    
    # Cython
    cython_output = volresample.grid_sample(
        input_data, grid, mode=mode, padding_mode=padding_mode
    )
    
    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, \
        f"3D identity grid, mode={mode}, padding={padding_mode}: max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_3d_torch_match_random_grid(mode, padding_mode):
    """Test 3D grid sample with random grid (including out-of-bounds) matches PyTorch."""
    input_data = generate_test_data((2, 3, 8, 8, 8))
    grid = create_random_grid_3d(2, 6, 6, 6, range_min=-1.5, range_max=1.5)
    
    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode=mode, padding_mode=padding_mode, align_corners=False
    ).numpy()
    
    cython_output = volresample.grid_sample(
        input_data, grid, mode=mode, padding_mode=padding_mode
    )
    
    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, \
        f"3D random grid, mode={mode}, padding={padding_mode}: max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_3d_torch_match_upsampling(mode):
    """Test 3D grid sample for upsampling (larger output than input)."""
    input_data = generate_test_data((1, 2, 4, 4, 4))
    grid = create_identity_grid_3d(1, 8, 8, 8)  # 2x upsampling
    
    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode=mode, padding_mode="zeros", align_corners=False
    ).numpy()
    
    cython_output = volresample.grid_sample(
        input_data, grid, mode=mode, padding_mode="zeros"
    )
    
    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, f"3D upsampling, mode={mode}: max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_3d_torch_match_downsampling(mode):
    """Test 3D grid sample for downsampling (smaller output than input)."""
    input_data = generate_test_data((1, 2, 8, 8, 8))
    grid = create_identity_grid_3d(1, 4, 4, 4)  # 2x downsampling
    
    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode=mode, padding_mode="zeros", align_corners=False
    ).numpy()
    
    cython_output = volresample.grid_sample(
        input_data, grid, mode=mode, padding_mode="zeros"
    )
    
    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, f"3D downsampling, mode={mode}: max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_3d_torch_match_non_uniform_sizes(mode):
    """Test 3D grid sample with non-uniform input and output sizes."""
    input_data = generate_test_data((2, 3, 7, 9, 11))  # Non-cubic input
    grid = create_identity_grid_3d(2, 5, 6, 8)  # Non-cubic output
    
    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode=mode, padding_mode="zeros", align_corners=False
    ).numpy()
    
    cython_output = volresample.grid_sample(
        input_data, grid, mode=mode, padding_mode="zeros"
    )
    
    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, f"3D non-uniform, mode={mode}: max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_3d_torch_match_extreme_out_of_bounds(mode, padding_mode):
    """Test 3D grid sample with extreme out-of-bounds coordinates."""
    input_data = generate_test_data((1, 1, 4, 4, 4))
    grid = create_random_grid_3d(1, 3, 3, 3, range_min=-5.0, range_max=5.0)
    
    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode=mode, padding_mode=padding_mode, align_corners=False
    ).numpy()
    
    cython_output = volresample.grid_sample(
        input_data, grid, mode=mode, padding_mode=padding_mode
    )
    
    max_diff = np.max(np.abs(torch_output - cython_output))
    assert max_diff < TOLERANCE, \
        f"3D extreme OOB, mode={mode}, padding={padding_mode}: max diff {max_diff:.6e}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_3d_torch_match_small_input(mode):
    """Test 3D grid sample with very small input."""
    input_data = generate_test_data((1, 1, 2, 2, 2))  # 2x2x2 input
    grid = create_identity_grid_3d(1, 4, 4, 4)
    
    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode=mode, padding_mode="zeros", align_corners=False
    ).numpy()
    
    cython_output = volresample.grid_sample(
        input_data, grid, mode=mode, padding_mode="zeros"
    )
    
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
@pytest.mark.parametrize("input_shape,grid_shape", [
    ((2, 3, 8, 8, 8), (2, 12, 12, 12, 3)),  # Upsampling
    ((1, 4, 16, 16, 16), (1, 8, 8, 8, 3)),  # Downsampling
])
def test_torch_match_various_sizes(input_shape, grid_shape):
    """Test grid sample with various input and output sizes (3D only)."""
    input_data = generate_test_data(input_shape)
    
    # Only 3D configurations
    grid = create_random_grid_3d(input_shape[0], grid_shape[1], grid_shape[2], grid_shape[3], range_min=-1.0, range_max=1.0)
    
    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode="bilinear", padding_mode="zeros", align_corners=False
    ).numpy()
    
    cython_output = volresample.grid_sample(
        input_data, grid, mode="bilinear", padding_mode="zeros"
    )
    
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
    grid = np.array([[[[[-1, -1, -1], [1, -1, -1]], [[-1, 1, -1], [1, 1, -1]]],
                      [[[-1, -1, 1], [1, -1, 1]], [[-1, 1, 1], [1, 1, 1]]]]], dtype=np.float32)
    
    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid)
    torch_output = F.grid_sample(
        input_torch, grid_torch, mode="bilinear", padding_mode="zeros", align_corners=False
    ).numpy()
    
    cython_output = volresample.grid_sample(
        input_data, grid, mode="bilinear", padding_mode="zeros"
    )
    
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
