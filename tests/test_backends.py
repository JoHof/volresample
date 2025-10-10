"""Unit tests for backend modules.

Tests both numpy and torch backends to ensure consistent behavior.
"""

import pytest
import numpy as np
from mimage.backends.numpy_backend import NumpyBackend

# Optional dependency
try:
    import torch
    from mimage.backends.torch_backend import TorchBackend
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    TorchBackend = None


class TestNumpyBackend:
    """Test NumpyBackend operations."""

    def test_name(self):
        """Test backend name."""
        assert NumpyBackend.name == 'numpy'

    def test_flip_single_axis(self):
        """Test flipping a single axis."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        flipped = NumpyBackend.flip(data, [0])
        
        expected = np.flip(data, axis=0)
        assert np.allclose(flipped, expected)

    def test_flip_multiple_axes(self):
        """Test flipping multiple axes."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        flipped = NumpyBackend.flip(data, [0, 2])
        
        expected = np.flip(data, axis=[0, 2])
        assert np.allclose(flipped, expected)

    def test_permute_identity(self):
        """Test identity permutation."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        permuted = NumpyBackend.permute(data, (0, 1, 2))
        
        assert np.allclose(permuted, data)

    def test_permute_reverse(self):
        """Test reversing axes."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        permuted = NumpyBackend.permute(data, (2, 1, 0))
        
        expected = np.transpose(data, (2, 1, 0))
        assert np.allclose(permuted, expected)
        assert permuted.shape == (4, 3, 2)

    def test_permute_swap_first_last(self):
        """Test swapping first and last axes."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        permuted = NumpyBackend.permute(data, (2, 1, 0))
        
        assert permuted.shape == (4, 3, 2)

    def test_asarray_from_numpy(self):
        """Test asarray with numpy input."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        result = NumpyBackend.asarray(data)
        
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, data)

    def test_asarray_from_list(self):
        """Test asarray with list input."""
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        result = NumpyBackend.asarray(data)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2, 2)

    def test_clone(self):
        """Test clone creates independent copy."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        cloned = NumpyBackend.clone(data)
        
        assert np.allclose(cloned, data)
        assert cloned is not data  # Different objects
        
        # Modify clone
        cloned[0, 0, 0] = 999
        assert data[0, 0, 0] != 999  # Original unchanged

    def test_clip_both_bounds(self):
        """Test clip with both min and max values."""
        data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]).astype(np.float32)
        clipped = NumpyBackend.clip(data, min_val=3, max_val=9)
        
        expected = np.clip(data, 3, 9)
        assert np.allclose(clipped, expected)
        assert clipped.min() >= 3
        assert clipped.max() <= 9

    def test_clip_min_only(self):
        """Test clip with only minimum value."""
        data = np.array([[[1, 2, 3], [4, 5, 6]]]).astype(np.float32)
        clipped = NumpyBackend.clip(data, min_val=3, max_val=None)
        
        expected = np.clip(data, 3, None)
        assert np.allclose(clipped, expected)
        assert clipped.min() >= 3

    def test_clip_max_only(self):
        """Test clip with only maximum value."""
        data = np.array([[[1, 2, 3], [4, 5, 6]]]).astype(np.float32)
        clipped = NumpyBackend.clip(data, min_val=None, max_val=4)
        
        expected = np.clip(data, None, 4)
        assert np.allclose(clipped, expected)
        assert clipped.max() <= 4

    def test_clip_no_change(self):
        """Test clip when all values are within range."""
        data = np.array([[[2, 3, 4], [5, 6, 7]]]).astype(np.float32)
        clipped = NumpyBackend.clip(data, min_val=1, max_val=10)
        
        assert np.allclose(clipped, data)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestTorchBackend:
    """Test TorchBackend operations."""

    def test_name(self):
        """Test backend name."""
        assert TorchBackend.name == 'torch'

    def test_flip_single_axis(self):
        """Test flipping a single axis."""
        data = torch.arange(24).reshape(2, 3, 4).float()
        flipped = TorchBackend.flip(data, [0])
        
        expected = torch.flip(data, dims=[0])
        assert torch.allclose(flipped, expected)

    def test_flip_multiple_axes(self):
        """Test flipping multiple axes."""
        data = torch.arange(24).reshape(2, 3, 4).float()
        flipped = TorchBackend.flip(data, [0, 2])
        
        expected = torch.flip(data, dims=[0, 2])
        assert torch.allclose(flipped, expected)

    def test_permute_identity(self):
        """Test identity permutation."""
        data = torch.arange(24).reshape(2, 3, 4).float()
        permuted = TorchBackend.permute(data, (0, 1, 2))
        
        assert torch.allclose(permuted, data)

    def test_permute_reverse(self):
        """Test reversing axes."""
        data = torch.arange(24).reshape(2, 3, 4).float()
        permuted = TorchBackend.permute(data, (2, 1, 0))
        
        expected = data.permute(2, 1, 0)
        assert torch.allclose(permuted, expected)
        assert permuted.shape == (4, 3, 2)

    def test_permute_swap_first_last(self):
        """Test swapping first and last axes."""
        data = torch.arange(24).reshape(2, 3, 4).float()
        permuted = TorchBackend.permute(data, (2, 1, 0))
        
        assert permuted.shape == torch.Size([4, 3, 2])

    def test_asarray_from_torch(self):
        """Test asarray with torch input."""
        data = torch.arange(24).reshape(2, 3, 4).float()
        result = TorchBackend.asarray(data)
        
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, data)

    def test_asarray_from_numpy(self):
        """Test asarray with numpy input."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        result = TorchBackend.asarray(data)
        
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, torch.from_numpy(data))

    def test_asarray_from_list(self):
        """Test asarray with list input."""
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        result = TorchBackend.asarray(data)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([2, 2, 2])

    def test_clone(self):
        """Test clone creates independent copy."""
        data = torch.arange(24).reshape(2, 3, 4).float()
        cloned = TorchBackend.clone(data)
        
        assert torch.allclose(cloned, data)
        assert cloned is not data  # Different tensors
        
        # Modify clone
        cloned[0, 0, 0] = 999
        assert data[0, 0, 0] != 999  # Original unchanged

    def test_clip_both_bounds(self):
        """Test clip with both min and max values."""
        data = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]).float()
        clipped = TorchBackend.clip(data, min_val=3, max_val=9)
        
        expected = torch.clamp(data, min=3, max=9)
        assert torch.allclose(clipped, expected)
        assert clipped.min() >= 3
        assert clipped.max() <= 9

    def test_clip_min_only(self):
        """Test clip with only minimum value."""
        data = torch.tensor([[[1, 2, 3], [4, 5, 6]]]).float()
        clipped = TorchBackend.clip(data, min_val=3, max_val=None)
        
        expected = torch.clamp(data, min=3)
        assert torch.allclose(clipped, expected)
        assert clipped.min() >= 3

    def test_clip_max_only(self):
        """Test clip with only maximum value."""
        data = torch.tensor([[[1, 2, 3], [4, 5, 6]]]).float()
        clipped = TorchBackend.clip(data, min_val=None, max_val=4)
        
        expected = torch.clamp(data, max=4)
        assert torch.allclose(clipped, expected)
        assert clipped.max() <= 4

    def test_clip_no_change(self):
        """Test clip when all values are within range."""
        data = torch.tensor([[[2, 3, 4], [5, 6, 7]]]).float()
        clipped = TorchBackend.clip(data, min_val=1, max_val=10)
        
        assert torch.allclose(clipped, data)


class TestBackendConsistency:
    """Test that numpy and torch backends produce consistent results."""

    def test_flip_consistency(self):
        """Test flip produces same results across backends."""
        data_np = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        flipped_np = NumpyBackend.flip(data_np, [0, 2])
        
        if HAS_TORCH:
            data_torch = torch.from_numpy(data_np)
            flipped_torch = TorchBackend.flip(data_torch, [0, 2])
            assert np.allclose(flipped_np, flipped_torch.numpy())

    def test_permute_consistency(self):
        """Test permute produces same results across backends."""
        data_np = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        permuted_np = NumpyBackend.permute(data_np, (2, 1, 0))
        
        if HAS_TORCH:
            data_torch = torch.from_numpy(data_np)
            permuted_torch = TorchBackend.permute(data_torch, (2, 1, 0))
            assert np.allclose(permuted_np, permuted_torch.numpy())

    def test_combined_operations_consistency(self):
        """Test combined operations produce consistent results."""
        data_np = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        result_np = NumpyBackend.flip(
            NumpyBackend.permute(data_np, (2, 1, 0)),
            [1]
        )
        
        if HAS_TORCH:
            data_torch = torch.from_numpy(data_np)
            result_torch = TorchBackend.flip(
                TorchBackend.permute(data_torch, (2, 1, 0)),
                [1]
            )
            assert np.allclose(result_np, result_torch.numpy())

    def test_clone_consistency(self):
        """Test clone behaves consistently across backends."""
        data_np = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        cloned_np = NumpyBackend.clone(data_np)
        
        assert np.allclose(data_np, cloned_np)
        
        if HAS_TORCH:
            data_torch = torch.from_numpy(data_np)
            cloned_torch = TorchBackend.clone(data_torch)
            assert torch.allclose(data_torch, cloned_torch)


class TestBackendEdgeCases:
    """Test edge cases and special scenarios."""

    def test_numpy_flip_empty_dims(self):
        """Test numpy flip with empty dims list."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        flipped = NumpyBackend.flip(data, [])
        assert np.allclose(flipped, data)

    def test_numpy_flip_all_axes(self):
        """Test numpy flip on all axes."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        flipped = NumpyBackend.flip(data, [0, 1, 2])
        
        expected = np.flip(np.flip(np.flip(data, 0), 1), 2)
        assert np.allclose(flipped, expected)

    def test_numpy_permute_4d(self):
        """Test numpy permute with 4D array."""
        data = np.arange(120).reshape(2, 3, 4, 5).astype(np.float32)
        permuted = NumpyBackend.permute(data, (3, 2, 1, 0))
        
        assert permuted.shape == (5, 4, 3, 2)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_flip_empty_dims(self):
        """Test torch flip with empty dims list."""
        data = torch.arange(24).reshape(2, 3, 4).float()
        flipped = TorchBackend.flip(data, [])
        assert torch.allclose(flipped, data)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_permute_4d(self):
        """Test torch permute with 4D tensor."""
        data = torch.arange(120).reshape(2, 3, 4, 5).float()
        permuted = TorchBackend.permute(data, (3, 2, 1, 0))
        
        assert permuted.shape == torch.Size([5, 4, 3, 2])

    def test_numpy_asarray_preserves_dtype(self):
        """Test that numpy asarray preserves data type."""
        data = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
        result = NumpyBackend.asarray(data)
        
        assert result.dtype == np.int32

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_asarray_preserves_dtype(self):
        """Test that torch asarray preserves data type."""
        data = torch.arange(24, dtype=torch.int32).reshape(2, 3, 4)
        result = TorchBackend.asarray(data)
        
        assert result.dtype == torch.int32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
