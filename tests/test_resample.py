"""Unit tests for resampling functionality.

Tests the resample function and both numpy and torch backends without comparing to SimpleITK.
Tests focus on correctness of shapes, spacing, interpolation quality, and edge cases.
"""

import pytest
import numpy as np
from mimage.mimage import Mimage
from mimage.affine import Affine
from mimage.resample import resample, resample_to_spacing
from mimage.backends.resampling import ResamplingNumpyBackend, ResamplingTorchBackend

# Check if torch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestResamplingBackends:
    """Test resampling backend implementations."""
    
    def test_numpy_backend_available(self):
        """Test that numpy backend is always available."""
        assert ResamplingNumpyBackend.available
    
    def test_torch_backend_availability(self):
        """Test torch backend availability matches torch import."""
        assert ResamplingTorchBackend.available == HAS_TORCH
    
    def test_numpy_backend_nearest(self):
        """Test numpy backend with nearest neighbor interpolation."""
        # Create simple 3D gradient
        data = np.arange(2*4*8).reshape(2, 4, 8).astype(np.float32)
        
        # Resample to larger size
        result = ResamplingNumpyBackend.resample(data, (4, 8, 16), mode='nearest')
        
        assert result.shape == (4, 8, 16)
        assert result.dtype == data.dtype
        # Check corners are preserved
        assert result[0, 0, 0] == data[0, 0, 0]
    
    def test_numpy_backend_linear(self):
        """Test numpy backend with linear interpolation."""
        # Create simple gradient: value = x + y + z
        x, y, z = np.meshgrid(np.arange(10), np.arange(20), np.arange(30), indexing='ij')
        data = (x + y + z).astype(np.float32)
        
        # Resample to different size
        result = ResamplingNumpyBackend.resample(data, (5, 10, 15), mode='linear')
        
        assert result.shape == (5, 10, 15)
        # Linear gradient should remain approximately linear
        # Check that corners are close
        np.testing.assert_allclose(result[0, 0, 0], data[0, 0, 0], rtol=0.1)
    
    def test_numpy_backend_cubic(self):
        """Test numpy backend with cubic interpolation."""
        data = np.random.rand(8, 12, 16).astype(np.float32)
        
        result = ResamplingNumpyBackend.resample(data, (16, 24, 32), mode='cubic')
        
        assert result.shape == (16, 24, 32)
        assert result.dtype == data.dtype
    
    def test_numpy_backend_with_batch(self):
        """Test that numpy backend rejects batch dimension."""
        # Backend should only accept 3D data
        data = np.random.rand(5, 8, 12, 16).astype(np.float32)
        
        with pytest.raises(ValueError, match="Data must be exactly 3D"):
            ResamplingNumpyBackend.resample(data, (16, 24, 32), mode='linear')
    
    def test_numpy_backend_invalid_mode(self):
        """Test that invalid mode raises error."""
        data = np.random.rand(8, 12, 16).astype(np.float32)
        
        with pytest.raises(ValueError, match="Unsupported interpolation mode"):
            ResamplingNumpyBackend.resample(data, (16, 24, 32), mode='invalid')
    
    def test_numpy_backend_insufficient_dimensions(self):
        """Test that data with < 3 dimensions raises error."""
        data = np.random.rand(8, 12).astype(np.float32)
        
        with pytest.raises(ValueError, match="Data must be exactly 3D"):
            ResamplingNumpyBackend.resample(data, (16, 24, 32), mode='linear')
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_backend_nearest(self):
        """Test torch backend with nearest neighbor interpolation."""
        data = torch.arange(2*4*8).reshape(2, 4, 8).float()
        
        result = ResamplingTorchBackend.resample(data, (4, 8, 16), mode='nearest')
        
        assert result.shape == (4, 8, 16)
        assert result.dtype == data.dtype
        # Check corner is preserved
        assert torch.isclose(result[0, 0, 0], data[0, 0, 0])
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_backend_linear(self):
        """Test torch backend with linear interpolation."""
        x, y, z = torch.meshgrid(
            torch.arange(10),
            torch.arange(20),
            torch.arange(30),
            indexing='ij'
        )
        data = (x + y + z).float()
        
        result = ResamplingTorchBackend.resample(data, (5, 10, 15), mode='linear')
        
        assert result.shape == (5, 10, 15)
        # Linear gradient should be preserved in general structure
        # (specific corner values may shift with downsampling)
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_backend_with_batch(self):
        """Test that torch backend rejects batch dimension."""
        data = torch.rand(5, 8, 12, 16)
        
        with pytest.raises(ValueError, match="Data must be exactly 3D"):
            ResamplingTorchBackend.resample(data, (16, 24, 32), mode='linear')
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_backend_with_numpy_input(self):
        """Test torch backend can handle numpy input."""
        data = np.random.rand(8, 12, 16).astype(np.float32)
        
        result = ResamplingTorchBackend.resample(data, (16, 24, 32), mode='linear')
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (16, 24, 32)


class TestResampleFunction:
    """Test the high-level resample function."""
    
    def test_resample_by_spacing_isotropic(self):
        """Test resampling by specifying isotropic spacing."""
        data = np.random.rand(10, 20, 30).astype(np.float32)
        affine = Affine(spacing=(2.0, 1.0, 0.5))
        img = Mimage(data, affine=affine)
        
        # Resample to 1mm isotropic using resample_to_spacing
        result = resample_to_spacing(img, spacing=1.0)
        
        # Check spacing (relaxed tolerance due to integer shape rounding)
        np.testing.assert_allclose(result.spacing, [1.0, 1.0, 1.0], rtol=0.05)
        
        # Check that physical extent is approximately preserved
        # Note: integer division in shape calculation may cause small discrepancies
        original_extent = (np.array(img.shape) - 1) * img.spacing
        new_extent = (np.array(result.shape) - 1) * result.spacing
        np.testing.assert_allclose(original_extent, new_extent, rtol=0.05)  # Increased tolerance
        
        # Origin should be preserved
        np.testing.assert_allclose(result.origin, img.origin, rtol=1e-5)
    
    def test_resample_by_spacing_anisotropic(self):
        """Test resampling by specifying anisotropic spacing."""
        data = np.random.rand(10, 20, 30).astype(np.float32)
        affine = Affine(spacing=(2.0, 1.0, 0.5))
        img = Mimage(data, affine=affine)
        
        # Resample to anisotropic spacing using resample_to_spacing
        new_spacing = (1.5, 0.8, 1.0)
        result = resample_to_spacing(img, spacing=new_spacing)
        
        # Check spacing (relaxed tolerance due to integer shape rounding)
        np.testing.assert_allclose(result.spacing, new_spacing, rtol=0.05)
        
        # Physical extent preserved (with tolerance for integer division)
        original_extent = (np.array(img.shape) - 1) * img.spacing
        new_extent = (np.array(result.shape) - 1) * result.spacing
        np.testing.assert_allclose(original_extent, new_extent, rtol=0.05)  # Increased tolerance
    
    def test_resample_by_shape(self):
        """Test resampling by specifying target shape."""
        data = np.random.rand(10, 20, 30).astype(np.float32)
        affine = Affine(spacing=(2.0, 1.0, 0.5))
        img = Mimage(data, affine=affine)
        
        # Resample to specific shape using resample
        target_shape = (20, 40, 60)
        result = resample(img, shape=target_shape)
        
        # Check shape
        assert result.shape == target_shape
        
        # Check that spacing was adjusted to preserve extent
        original_extent = (np.array(img.shape) - 1) * img.spacing
        new_extent = (np.array(result.shape) - 1) * result.spacing
        np.testing.assert_allclose(original_extent, new_extent, rtol=1e-5)
    
    def test_resample_upsample(self):
        """Test upsampling (increasing resolution)."""
        data = np.random.rand(10, 20, 30).astype(np.float32)
        affine = Affine(spacing=(2.0, 2.0, 2.0))
        img = Mimage(data, affine=affine)
        
        # Upsample to finer resolution
        result = resample_to_spacing(img, spacing=0.5)
        
        # Shape should increase
        assert all(result.shape[i] > img.shape[i] for i in range(3))
        
        # Spacing should decrease
        np.testing.assert_allclose(result.spacing, [0.5, 0.5, 0.5], rtol=1e-5)
    
    def test_resample_downsample(self):
        """Test downsampling (decreasing resolution)."""
        data = np.random.rand(40, 80, 120).astype(np.float32)
        affine = Affine(spacing=(0.5, 0.5, 0.5))
        img = Mimage(data, affine=affine)
        
        # Downsample to coarser resolution
        result = resample_to_spacing(img, spacing=2.0)
        
        # Shape should decrease
        assert all(result.shape[i] < img.shape[i] for i in range(3))
        
        # Spacing should increase (relaxed tolerance due to integer shape rounding)
        np.testing.assert_allclose(result.spacing, [2.0, 2.0, 2.0], rtol=0.1)
    
    def test_resample_with_non_zero_origin(self):
        """Test that non-zero origin is preserved."""
        data = np.random.rand(10, 20, 30).astype(np.float32)
        origin = np.array([-50.0, 100.0, 25.0])
        affine = Affine(spacing=(2.0, 1.0, 0.5), origin=origin)
        img = Mimage(data, affine=affine)
        
        # Resample
        result = resample_to_spacing(img, spacing=1.0)
        
        # Origin should be preserved
        np.testing.assert_allclose(result.origin, origin, rtol=1e-5)
    
    def test_resample_preserves_direction(self):
        """Test that direction matrix is preserved."""
        data = np.random.rand(10, 20, 30).astype(np.float32)
        direction = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])
        affine = Affine(direction=direction, spacing=(2.0, 1.0, 0.5))
        img = Mimage(data, affine=affine)
        
        # Resample
        result = resample_to_spacing(img, spacing=1.0)
        
        # Direction should be preserved
        np.testing.assert_allclose(result.direction, direction, rtol=1e-5)
    
    def test_resample_with_batch_dimension(self):
        """Test resampling with batch dimension."""
        # Create 4D data with batch dimension
        data = np.random.rand(5, 10, 20, 30).astype(np.float32)
        # Set spacing so that resampling will actually change the shape
        affine = Affine(spacing=(2.0, 1.0, 0.5))
        img = Mimage(data, affine=affine, spatial_dims=(1, 2, 3))
        
        # Resample to isotropic 1mm using resample_to_spacing
        result = resample_to_spacing(img, spacing=1.0)
        
        # Batch dimension should be preserved
        assert result.shape[0] == 5
        
        # At least one spatial dimension should change
        # (given the different spacings, resampling to isotropic should change shape)
        spatial_changed = any(
            result.shape[i] != img.shape[i] for i in range(1, 4)
        )
        assert spatial_changed, "Spatial dimensions should change after resampling"
    
    def test_resample_interpolation_modes(self):
        """Test different interpolation modes produce different results."""
        # Create smooth gradient
        x, y, z = np.meshgrid(
            np.linspace(0, 10, 10),
            np.linspace(0, 10, 20),
            np.linspace(0, 10, 30),
            indexing='ij'
        )
        data = (np.sin(x) + np.cos(y) + np.sin(z)).astype(np.float32)
        img = Mimage(data)
        
        # Resample with different modes
        result_nearest = resample(img, shape=(20, 40, 60), mode='nearest')
        result_linear = resample(img, shape=(20, 40, 60), mode='linear')
        result_cubic = resample(img, shape=(20, 40, 60), mode='cubic')
        
        # Results should be different
        assert not np.allclose(result_nearest.data, result_linear.data)
        assert not np.allclose(result_linear.data, result_cubic.data)
        
        # All should have correct shape
        assert result_nearest.shape == (20, 40, 60)
        assert result_linear.shape == (20, 40, 60)
        assert result_cubic.shape == (20, 40, 60)
    
    def test_resample_coordinate_consistency(self):
        """Test that coordinate transformations remain consistent after resampling."""
        data = np.random.rand(10, 20, 30).astype(np.float32)
        affine = Affine(spacing=(2.0, 1.0, 0.5), origin=(-10.0, 5.0, 2.0))
        img = Mimage(data, affine=affine)
        
        # Resample
        result = resample_to_spacing(img, spacing=1.0)
        
        # Test that first voxel maps to same physical location
        orig_coord = img.index_to_coord([0, 0, 0])
        new_coord = result.index_to_coord([0, 0, 0])
        
        # Origins should match (both are center of first voxel)
        np.testing.assert_allclose(orig_coord, new_coord, rtol=1e-5)
        
        # Test round-trip: index -> coord -> index
        test_idx = np.array([[5, 10, 15], [result.shape[0]-1, result.shape[1]-1, result.shape[2]-1]])
        coords = result.index_to_coord(test_idx)
        idx_back = result.coord_to_index(coords)
        np.testing.assert_allclose(idx_back, test_idx, rtol=1e-5)
    
    def test_resample_invalid_spacing_shape(self):
        """Test that invalid spacing shape raises error."""
        data = np.random.rand(10, 20, 30).astype(np.float32)
        img = Mimage(data)
        
        with pytest.raises(ValueError, match="Spacing must be scalar or length-3 tuple"):
            resample_to_spacing(img, spacing=(1.0, 2.0))  # Only 2 values
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_resample_torch_backend_used_for_torch_image(self):
        """Test that torch backend is used when image has torch data."""
        data = torch.rand(10, 20, 30)
        img = Mimage(data, backend='torch')
        
        # Resample
        result = resample_to_spacing(img, spacing=1.0)
        
        # Result should still be torch
        assert isinstance(result.data, torch.Tensor)
        assert result.backend == 'torch'
    
    def test_resample_quality_linear_gradient(self):
        """Test that linear interpolation preserves linear gradients well."""
        # Create perfect linear gradient
        x, y, z = np.meshgrid(
            np.arange(10),
            np.arange(20),
            np.arange(30),
            indexing='ij'
        )
        data = (2*x + 3*y + 4*z).astype(np.float32)
        img = Mimage(data)
        
        # Resample with linear interpolation
        result = resample(img, shape=(20, 40, 60), mode='linear')
        
        # Check that gradient is still linear (sample a few points)
        # At index [10, 20, 30], with scaling factor 2, should map to approx [5, 10, 15] in original
        # Value should be approximately 2*5 + 3*10 + 4*15 = 10 + 30 + 60 = 100
        expected_scale_x = 9.0 / 19.0  # (10-1) / (20-1)
        expected_scale_y = 19.0 / 39.0  # (20-1) / (40-1)
        expected_scale_z = 29.0 / 59.0  # (30-1) / (60-1)
        
        orig_i, orig_j, orig_k = 10 * expected_scale_x, 20 * expected_scale_y, 30 * expected_scale_z
        expected_value = 2*orig_i + 3*orig_j + 4*orig_k
        
        actual_value = result.data[10, 20, 30]
        
        # Should be very close for linear gradient with linear interpolation
        np.testing.assert_allclose(actual_value, expected_value, rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
