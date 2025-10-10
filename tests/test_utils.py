"""Unit tests for utils.py module.

Tests utility functions for medical image processing.
"""

import pytest
import numpy as np
from mimage.utils import get_orientation_from_direction_cosines

# Optional dependency
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


class TestGetOrientationFromDirectionCosines:
    """Test get_orientation_from_direction_cosines function."""

    def test_identity_orientation(self):
        """Test that identity direction gives LPS orientation."""
        direction = np.eye(3)
        orientation = get_orientation_from_direction_cosines(direction)
        assert orientation == "LPS"

    def test_ras_orientation(self):
        """Test RAS (neurological) orientation."""
        # RAS: Right-Anterior-Superior
        direction = np.array([
            [-1, 0, 0],  # First axis points Right
            [0, -1, 0],  # Second axis points Anterior
            [0, 0, 1],   # Third axis points Superior
        ])
        orientation = get_orientation_from_direction_cosines(direction)
        assert orientation == "RAS"

    def test_lpi_orientation(self):
        """Test LPI orientation."""
        # LPI: Left-Posterior-Inferior
        direction = np.array([
            [1, 0, 0],   # First axis points Left
            [0, 1, 0],   # Second axis points Posterior
            [0, 0, -1],  # Third axis points Inferior
        ])
        orientation = get_orientation_from_direction_cosines(direction)
        assert orientation == "LPI"

    def test_permuted_axes(self):
        """Test with permuted axes."""
        # SPL: Superior-Posterior-Left (permuted from LPS)
        direction = np.array([
            [0, 0, 1],   # First axis points Superior
            [0, 1, 0],   # Second axis points Posterior
            [1, 0, 0],   # Third axis points Left
        ])
        orientation = get_orientation_from_direction_cosines(direction)
        assert orientation == "SPL"

    def test_invalid_shape(self):
        """Test that invalid direction shape raises ValueError."""
        with pytest.raises(ValueError, match="must be 3x3"):
            get_orientation_from_direction_cosines(np.eye(2))

    def test_non_orthogonal_matrix(self):
        """Test with non-orthogonal matrix (should still work but give approximate result)."""
        # Slightly non-orthogonal matrix
        direction = np.array([
            [1.0, 0.1, 0],
            [0, 1.0, 0],
            [0, 0, 1.0],
        ])
        # Should still determine primary orientations
        orientation = get_orientation_from_direction_cosines(direction)
        assert len(orientation) == 3
        assert orientation[0] in "LR"
        assert orientation[1] in "PA"
        assert orientation[2] in "SI"

    @pytest.mark.skipif(not HAS_SITK, reason="SimpleITK not installed")
    def test_vs_sitk_identity(self):
        """Test against SimpleITK for identity direction."""
        direction = np.eye(3).flatten()
        sitk_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            direction.tolist()
        )
        
        our_orientation = get_orientation_from_direction_cosines(np.eye(3))
        assert our_orientation == sitk_orientation

    @pytest.mark.skipif(not HAS_SITK, reason="SimpleITK not installed")
    def test_vs_sitk_ras(self):
        """Test against SimpleITK for RAS orientation."""
        direction = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ])
        direction_flat = direction.flatten()
        
        sitk_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            direction_flat.tolist()
        )
        
        our_orientation = get_orientation_from_direction_cosines(direction)
        assert our_orientation == sitk_orientation

    @pytest.mark.skipif(not HAS_SITK, reason="SimpleITK not installed")
    def test_vs_sitk_various_orientations(self):
        """Test against SimpleITK for various orientations."""
        test_cases = [
            "LPS", "RAS", "LPI", "RPI", "LAS", "RAI",
            "PSL", "ASL", "PIL", "AIL", "PSR", "ASR",
            "SLP", "SRP", "SLA", "SRA", "ILP", "IRP",
            "PLI", "PRI", "ALI", "ARI", "PLS", "PRS",
        ]
        
        for expected_orientation in test_cases:
            # Get direction from SimpleITK
            direction_flat = sitk.DICOMOrientImageFilter_GetDirectionCosinesFromOrientation(
                expected_orientation
            )
            direction = np.array(direction_flat).reshape(3, 3)
            
            # Test our function
            our_orientation = get_orientation_from_direction_cosines(direction)
            assert our_orientation == expected_orientation, \
                f"Expected {expected_orientation}, got {our_orientation}"

    def test_diagonal_negative_axes(self):
        """Test with diagonal matrix with negative axes."""
        # Flip all axes
        direction = -np.eye(3)
        orientation = get_orientation_from_direction_cosines(direction)
        assert orientation == "RAI"  # Right-Anterior-Inferior

    def test_45_degree_rotation(self):
        """Test with 45-degree rotation (should pick dominant axis)."""
        # Rotate 45 degrees around z-axis
        angle = np.pi / 4
        direction = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        orientation = get_orientation_from_direction_cosines(direction)
        # Should pick the dominant direction for each axis
        assert len(orientation) == 3
        assert orientation[2] == "S"  # Z-axis is unchanged

    def test_all_standard_orientations(self):
        """Test all 48 standard anatomical orientations."""
        # All possible 3-letter codes
        standard_orientations = [
            "LPS", "LPI", "LAS", "LAI", "LSP", "LSA", "LIP", "LIA",
            "RPS", "RPI", "RAS", "RAI", "RSP", "RSA", "RIP", "RIA",
            "PLS", "PLI", "PRS", "PRI", "PSL", "PSR", "PIL", "PIR",
            "ALS", "ALI", "ARS", "ARI", "ASL", "ASR", "AIL", "AIR",
            "SLP", "SLA", "SRP", "SRA", "SPL", "SPR", "SAL", "SAR",
            "ILP", "ILA", "IRP", "IRA", "IPL", "IPR", "IAL", "IAR",
        ]
        
        # Mapping from orientation code to direction
        axis_map = {
            'L': np.array([1, 0, 0]),
            'R': np.array([-1, 0, 0]),
            'P': np.array([0, 1, 0]),
            'A': np.array([0, -1, 0]),
            'S': np.array([0, 0, 1]),
            'I': np.array([0, 0, -1]),
        }
        
        for expected_orientation in standard_orientations:
            # Build direction matrix from orientation code
            direction = np.column_stack([
                axis_map[expected_orientation[0]],
                axis_map[expected_orientation[1]],
                axis_map[expected_orientation[2]],
            ])
            
            # Test our function
            result = get_orientation_from_direction_cosines(direction)
            assert result == expected_orientation, \
                f"Expected {expected_orientation}, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
