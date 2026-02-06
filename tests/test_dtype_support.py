"""Unit tests for dtype support in resampling.

Tests uint8 and int16 dtype support for nearest neighbor interpolation,
and float32 conversion for linear/area modes.
"""

import numpy as np

import volresample


class TestDtypeSupport3D:
    """Test dtype support for 3D resampling."""

    def test_nearest_uint8_preserves_dtype(self):
        """Nearest neighbor should preserve uint8 dtype."""
        data = np.array([[[1, 200], [50, 255]]], dtype=np.uint8)
        result = volresample.resample(data, (2, 2, 2), mode="nearest")

        assert result.dtype == np.uint8
        assert result.shape == (2, 2, 2)
        # Values should stay in uint8 range
        assert np.all(result >= 0) and np.all(result <= 255)

    def test_nearest_int16_preserves_dtype(self):
        """Nearest neighbor should preserve int16 dtype."""
        data = np.array([[[-1000, 500], [100, -32768]]], dtype=np.int16)
        result = volresample.resample(data, (2, 2, 2), mode="nearest")

        assert result.dtype == np.int16
        assert result.shape == (2, 2, 2)

    def test_nearest_float32_preserves_dtype(self):
        """Nearest neighbor should preserve float32 dtype."""
        data = np.array([[[1.5, 2.5], [3.5, 4.5]]], dtype=np.float32)
        result = volresample.resample(data, (2, 2, 2), mode="nearest")

        assert result.dtype == np.float32
        assert result.shape == (2, 2, 2)

    def test_linear_uint8_converts_to_float32(self):
        """Linear mode should convert uint8 to float32."""
        data = np.array([[[100, 200], [50, 150]]], dtype=np.uint8)
        result = volresample.resample(data, (2, 2, 2), mode="linear")

        assert result.dtype == np.float32
        assert result.shape == (2, 2, 2)

    def test_linear_int16_converts_to_float32(self):
        """Linear mode should convert int16 to float32."""
        data = np.array([[[1000, -500], [100, 2000]]], dtype=np.int16)
        result = volresample.resample(data, (2, 2, 2), mode="linear")

        assert result.dtype == np.float32
        assert result.shape == (2, 2, 2)

    def test_area_uint8_converts_to_float32(self):
        """Area mode should convert uint8 to float32."""
        data = np.ones((2, 4, 4), dtype=np.uint8) * 100
        result = volresample.resample(data, (2, 2, 2), mode="area")

        assert result.dtype == np.float32
        assert result.shape == (2, 2, 2)

    def test_nearest_uint8_values_correct(self):
        """Verify nearest neighbor correctly samples uint8 values."""
        # Create simple pattern
        data = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        result = volresample.resample(data, (2, 2, 2), mode="nearest")

        assert result.dtype == np.uint8
        # Verify values are from original data
        unique_values = set(result.flatten())
        assert unique_values.issubset({10, 20, 30, 40})

    def test_nearest_int16_negative_values(self):
        """Nearest neighbor should handle negative int16 values correctly."""
        data = np.array([[[-100, -200], [300, -400]]], dtype=np.int16)
        result = volresample.resample(data, (3, 3, 3), mode="nearest")

        assert result.dtype == np.int16
        # All values should be from original data
        unique_values = set(result.flatten())
        assert unique_values.issubset({-100, -200, 300, -400})

    def test_nearest_uint8_boundary_values(self):
        """Test uint8 boundary values (0 and 255)."""
        data = np.array([[[0, 255], [128, 64]]], dtype=np.uint8)
        result = volresample.resample(data, (2, 2, 2), mode="nearest")

        assert result.dtype == np.uint8
        assert 0 in result or 255 in result or 128 in result or 64 in result

    def test_nearest_int16_boundary_values(self):
        """Test int16 boundary values."""
        data = np.array([[[-32768, 32767], [0, 1000]]], dtype=np.int16)
        result = volresample.resample(data, (2, 2, 2), mode="nearest")

        assert result.dtype == np.int16
        # Values should be preserved from input
        unique_values = set(result.flatten())
        assert unique_values.issubset({-32768, 32767, 0, 1000})

    def test_upsampling_uint8(self):
        """Test upsampling with uint8."""
        data = np.array([[[100, 200]]], dtype=np.uint8)
        result = volresample.resample(data, (2, 4, 4), mode="nearest")

        assert result.dtype == np.uint8
        assert result.shape == (2, 4, 4)
        assert np.all((result == 100) | (result == 200))

    def test_downsampling_int16(self):
        """Test downsampling with int16."""
        data = np.ones((2, 8, 8), dtype=np.int16) * 500
        result = volresample.resample(data, (2, 2, 2), mode="nearest")

        assert result.dtype == np.int16
        assert result.shape == (2, 2, 2)
        assert np.all(result == 500)

    def test_large_uint8_array(self):
        """Test large uint8 array."""
        data = np.random.randint(0, 256, (4, 32, 32), dtype=np.uint8)
        result = volresample.resample(data, (8, 64, 64), mode="nearest")

        assert result.dtype == np.uint8
        assert result.shape == (8, 64, 64)

    def test_large_int16_array(self):
        """Test large int16 array."""
        data = np.random.randint(-32768, 32767, (4, 32, 32), dtype=np.int16)
        result = volresample.resample(data, (8, 64, 64), mode="nearest")

        assert result.dtype == np.int16
        assert result.shape == (8, 64, 64)


class TestDtypeSupport4D:
    """Test dtype support for 4D resampling."""

    def test_4d_nearest_uint8_preserves_dtype(self):
        """4D nearest neighbor should preserve uint8 dtype."""
        data = np.ones((2, 2, 2, 2), dtype=np.uint8) * 150
        result = volresample.resample(data, (3, 3, 3), mode="nearest")

        assert result.dtype == np.uint8
        assert result.shape == (2, 3, 3, 3)
        assert np.all(result == 150)

    def test_4d_nearest_int16_preserves_dtype(self):
        """4D nearest neighbor should preserve int16 dtype."""
        data = np.ones((3, 2, 2, 2), dtype=np.int16) * -1000
        result = volresample.resample(data, (3, 3, 3), mode="nearest")

        assert result.dtype == np.int16
        assert result.shape == (3, 3, 3, 3)
        assert np.all(result == -1000)

    def test_4d_nearest_float32_preserves_dtype(self):
        """4D nearest neighbor should preserve float32 dtype."""
        data = np.ones((2, 2, 2, 2), dtype=np.float32) * 3.14
        result = volresample.resample(data, (3, 3, 3), mode="nearest")

        assert result.dtype == np.float32
        assert result.shape == (2, 3, 3, 3)

    def test_4d_linear_uint8_converts_to_float32(self):
        """4D linear mode should convert uint8 to float32."""
        data = np.ones((2, 2, 2, 2), dtype=np.uint8) * 100
        result = volresample.resample(data, (3, 3, 3), mode="linear")

        assert result.dtype == np.float32
        assert result.shape == (2, 3, 3, 3)

    def test_4d_linear_int16_converts_to_float32(self):
        """4D linear mode should convert int16 to float32."""
        data = np.ones((2, 2, 2, 2), dtype=np.int16) * 500
        result = volresample.resample(data, (3, 3, 3), mode="linear")

        assert result.dtype == np.float32
        assert result.shape == (2, 3, 3, 3)

    def test_4d_multichannel_uint8(self):
        """Test 4D with multiple channels of uint8."""
        # 4 channels, different values
        data = np.zeros((4, 2, 2, 2), dtype=np.uint8)
        for c in range(4):
            data[c] = c * 50

        result = volresample.resample(data, (3, 3, 3), mode="nearest")

        assert result.dtype == np.uint8
        assert result.shape == (4, 3, 3, 3)
        # Each channel should have its values preserved
        for c in range(4):
            assert np.all(result[c] == c * 50)

    def test_4d_multichannel_int16(self):
        """Test 4D with multiple channels of int16."""
        data = np.zeros((3, 3, 3, 3), dtype=np.int16)
        for c in range(3):
            data[c] = (c - 1) * 1000  # -1000, 0, 1000

        result = volresample.resample(data, (4, 4, 4), mode="nearest")

        assert result.dtype == np.int16
        assert result.shape == (3, 4, 4, 4)
        for c in range(3):
            assert np.all(result[c] == (c - 1) * 1000)

    def test_4d_area_uint8_converts_to_float32(self):
        """4D area mode should convert uint8 to float32."""
        data = np.ones((2, 4, 4, 4), dtype=np.uint8) * 100
        result = volresample.resample(data, (2, 2, 2), mode="area")

        assert result.dtype == np.float32
        assert result.shape == (2, 2, 2, 2)


class TestDtypeEdgeCases:
    """Test edge cases and error conditions for dtype support."""

    def test_constant_value_uint8(self):
        """Constant value array with uint8."""
        data = np.full((2, 4, 4), 42, dtype=np.uint8)
        result = volresample.resample(data, (3, 5, 5), mode="nearest")

        assert result.dtype == np.uint8
        assert np.all(result == 42)

    def test_constant_value_int16(self):
        """Constant value array with int16."""
        data = np.full((2, 4, 4), -999, dtype=np.int16)
        result = volresample.resample(data, (3, 5, 5), mode="nearest")

        assert result.dtype == np.int16
        assert np.all(result == -999)

    def test_asymmetric_scaling_uint8(self):
        """Test asymmetric output sizes with uint8."""
        data = np.random.randint(0, 256, (3, 5, 5), dtype=np.uint8)
        result = volresample.resample(data, (4, 7, 6), mode="nearest")

        assert result.dtype == np.uint8
        assert result.shape == (4, 7, 6)

    def test_asymmetric_scaling_int16(self):
        """Test asymmetric output sizes with int16."""
        data = np.random.randint(-32768, 32767, (3, 5, 5), dtype=np.int16)
        result = volresample.resample(data, (4, 7, 6), mode="nearest")

        assert result.dtype == np.int16
        assert result.shape == (4, 7, 6)

    def test_single_element_upsampling_uint8(self):
        """Upsample single element with uint8."""
        data = np.array([[[200]]], dtype=np.uint8)
        result = volresample.resample(data, (3, 5, 5), mode="nearest")

        assert result.dtype == np.uint8
        assert result.shape == (3, 5, 5)
        assert np.all(result == 200)

    def test_single_element_upsampling_int16(self):
        """Upsample single element with int16."""
        data = np.array([[[-5000]]], dtype=np.int16)
        result = volresample.resample(data, (3, 5, 5), mode="nearest")

        assert result.dtype == np.int16
        assert result.shape == (3, 5, 5)
        assert np.all(result == -5000)

    def test_uint8_with_zeros(self):
        """Test uint8 array with zero values."""
        data = np.array([[[0, 0], [0, 0]]], dtype=np.uint8)
        result = volresample.resample(data, (2, 2, 2), mode="nearest")

        assert result.dtype == np.uint8
        assert np.all(result == 0)

    def test_int16_with_zeros(self):
        """Test int16 array with zero values."""
        data = np.array([[[0, 0], [0, 0]]], dtype=np.int16)
        result = volresample.resample(data, (2, 2, 2), mode="nearest")

        assert result.dtype == np.int16
        assert np.all(result == 0)

    def test_mixed_uint8_values(self):
        """Test uint8 with all different values."""
        # Use pattern that ensures different values
        data = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
        result = volresample.resample(data, (8, 8, 8), mode="nearest")

        assert result.dtype == np.uint8
        assert result.shape == (8, 8, 8)
        # All values should be from original range
        assert np.all(result < 64)


class TestDtypeConsistency:
    """Test consistency of dtype handling across different scenarios."""

    def test_nearest_consistency_across_shapes_uint8(self):
        """Nearest neighbor should handle various shapes consistently for uint8."""
        base_data = np.random.randint(0, 256, (2, 10, 10), dtype=np.uint8)

        # Test different output sizes
        for out_size in [(3, 5, 5), (10, 10, 10), (1, 20, 20)]:
            result = volresample.resample(base_data, out_size, mode="nearest")
            assert result.dtype == np.uint8
            assert result.shape == (out_size[0], out_size[1], out_size[2])

    def test_nearest_consistency_across_shapes_int16(self):
        """Nearest neighbor should handle various shapes consistently for int16."""
        base_data = np.random.randint(-32768, 32767, (2, 10, 10), dtype=np.int16)

        # Test different output sizes
        for out_size in [(3, 5, 5), (10, 10, 10), (1, 20, 20)]:
            result = volresample.resample(base_data, out_size, mode="nearest")
            assert result.dtype == np.int16
            assert result.shape == (out_size[0], out_size[1], out_size[2])

    def test_mode_consistency_uint8(self):
        """Test that mode affects output dtype correctly for uint8."""
        data = np.random.randint(0, 256, (2, 4, 4), dtype=np.uint8)

        # Nearest: should stay uint8
        result_nearest = volresample.resample(data, (3, 3, 3), mode="nearest")
        assert result_nearest.dtype == np.uint8

        # Linear: should become float32
        result_linear = volresample.resample(data, (3, 3, 3), mode="linear")
        assert result_linear.dtype == np.float32

    def test_mode_consistency_int16(self):
        """Test that mode affects output dtype correctly for int16."""
        data = np.random.randint(-32768, 32767, (2, 4, 4), dtype=np.int16)

        # Nearest: should stay int16
        result_nearest = volresample.resample(data, (3, 3, 3), mode="nearest")
        assert result_nearest.dtype == np.int16

        # Linear: should become float32
        result_linear = volresample.resample(data, (3, 3, 3), mode="linear")
        assert result_linear.dtype == np.float32
