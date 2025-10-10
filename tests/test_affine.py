"""Unit tests for affine.py module.

Tests affine transformations against SimpleITK reference implementation.
"""

import pytest
import numpy as np
from mimage.affine import Affine

# Optional dependency
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


class TestAffineConstruction:
    """Test Affine object construction and initialization."""

    def test_default_affine(self):
        """Test creating affine with default identity transform."""
        affine = Affine()
        assert np.allclose(affine.matrix, np.eye(4))
        assert np.allclose(affine.direction, np.eye(3))
        assert np.allclose(affine.spacing, np.ones(3))
        assert np.allclose(affine.origin, np.zeros(3))

    def test_affine_from_matrix(self):
        """Test creating affine from a 4x4 matrix."""
        matrix = np.array([
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        affine = Affine(matrix=matrix)
        assert np.allclose(affine.matrix, matrix)
        assert np.allclose(affine.spacing, [1.0, 2.0, 3.0])
        assert np.allclose(affine.origin, [10.0, 20.0, 30.0])

    def test_affine_from_components(self):
        """Test creating affine from direction, spacing, and origin."""
        direction = np.eye(3)
        spacing = np.array([0.5, 1.0, 1.5])
        origin = np.array([-100.0, 50.0, 25.0])
        
        affine = Affine(direction=direction, spacing=spacing, origin=origin)
        assert np.allclose(affine.direction, direction)
        assert np.allclose(affine.spacing, spacing)
        assert np.allclose(affine.origin, origin)

    def test_invalid_matrix_shape(self):
        """Test that invalid matrix shape raises ValueError."""
        with pytest.raises(ValueError, match="must be 4x4"):
            Affine(matrix=np.eye(3))

    def test_invalid_direction_shape(self):
        """Test that invalid direction shape raises ValueError."""
        with pytest.raises(ValueError, match="must be 3x3"):
            Affine(direction=np.eye(2))

    def test_invalid_spacing_shape(self):
        """Test that invalid spacing shape raises ValueError."""
        with pytest.raises(ValueError, match="length 3"):
            Affine(spacing=[1.0, 2.0])

    def test_negative_spacing(self):
        """Test that negative spacing raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            Affine(spacing=[1.0, -2.0, 3.0])


class TestAffineProperties:
    """Test Affine property getters and setters."""

    def test_direction_getter(self):
        """Test that direction getter returns a copy."""
        affine = Affine()
        direction = affine.direction
        direction[0, 0] = 999
        assert affine.direction[0, 0] == 1.0  # unchanged

    def test_spacing_getter(self):
        """Test that spacing getter returns a copy."""
        affine = Affine()
        spacing = affine.spacing
        spacing[0] = 999
        assert affine.spacing[0] == 1.0  # unchanged

    def test_origin_getter(self):
        """Test that origin getter returns a copy."""
        affine = Affine()
        origin = affine.origin
        origin[0] = 999
        assert affine.origin[0] == 0.0  # unchanged

    def test_spacing_setter(self):
        """Test setting new spacing updates matrix."""
        affine = Affine()
        new_spacing = np.array([2.0, 3.0, 4.0])
        affine.spacing = new_spacing
        assert np.allclose(affine.spacing, new_spacing)
        assert np.allclose(affine.matrix[:3, :3], np.diag(new_spacing))

    def test_origin_setter(self):
        """Test setting new origin updates matrix."""
        affine = Affine()
        new_origin = np.array([10.0, 20.0, 30.0])
        affine.origin = new_origin
        assert np.allclose(affine.origin, new_origin)
        assert np.allclose(affine.matrix[:3, 3], new_origin)


@pytest.mark.skipif(not HAS_SITK, reason="SimpleITK not installed")
class TestAffineVsSimpleITK:
    """Test Affine transformations against SimpleITK reference."""

    @pytest.fixture
    def sitk_image(self):
        """Create a SimpleITK image with known affine parameters."""
        img = sitk.Image([10, 20, 30], sitk.sitkFloat32)
        img.SetSpacing([0.5, 1.0, 1.5])
        img.SetOrigin([-10.0, 20.0, 30.0])
        # Identity direction
        img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        return img

    @pytest.fixture
    def affine_from_sitk(self, sitk_image):
        """Create corresponding Affine object."""
        spacing = np.array(sitk_image.GetSpacing())
        origin = np.array(sitk_image.GetOrigin())
        direction = np.array(sitk_image.GetDirection()).reshape(3, 3)
        return Affine(direction=direction, spacing=spacing, origin=origin)

    def test_index_to_coord_single_point(self, sitk_image, affine_from_sitk):
        """Test index_to_coord matches SimpleITK TransformIndexToPhysicalPoint."""
        test_indices = [
            [0, 0, 0],
            [5, 10, 15],
            [9, 19, 29],
        ]
        
        for idx in test_indices:
            # SimpleITK expects (x, y, z) order
            sitk_coord = sitk_image.TransformIndexToPhysicalPoint(idx)
            affine_coord = affine_from_sitk.index_to_coord(np.array(idx))
            assert np.allclose(affine_coord, sitk_coord, atol=1e-6), \
                f"Index {idx}: Affine {affine_coord} != SimpleITK {sitk_coord}"

    def test_index_to_coord_multiple_points(self, sitk_image, affine_from_sitk):
        """Test index_to_coord with multiple points (N, 3) array."""
        test_indices = np.array([
            [0, 0, 0],
            [5, 10, 15],
            [9, 19, 29],
        ])
        
        # Test batch conversion
        affine_coords = affine_from_sitk.index_to_coord(test_indices)
        assert affine_coords.shape == (3, 3)
        
        # Verify each point
        for i, idx in enumerate(test_indices):
            sitk_coord = sitk_image.TransformIndexToPhysicalPoint(idx.tolist())
            assert np.allclose(affine_coords[i], sitk_coord, atol=1e-6)

    def test_coord_to_index_single_point(self, sitk_image, affine_from_sitk):
        """Test coord_to_index matches SimpleITK TransformPhysicalPointToIndex."""
        test_coords = [
            [-10.0, 20.0, 30.0],  # origin
            [-7.5, 30.0, 52.5],   # middle point
            [-5.5, 39.0, 73.5],   # near end
        ]
        
        for coord in test_coords:
            # SimpleITK continuous index
            sitk_idx = sitk_image.TransformPhysicalPointToContinuousIndex(coord)
            affine_idx = affine_from_sitk.coord_to_index(np.array(coord))
            assert np.allclose(affine_idx, sitk_idx, atol=1e-6), \
                f"Coord {coord}: Affine {affine_idx} != SimpleITK {sitk_idx}"

    def test_coord_to_index_multiple_points(self, sitk_image, affine_from_sitk):
        """Test coord_to_index with multiple points (N, 3) array."""
        test_coords = np.array([
            [-10.0, 20.0, 30.0],
            [-7.5, 30.0, 52.5],
            [-5.5, 39.0, 73.5],
        ])
        
        affine_indices = affine_from_sitk.coord_to_index(test_coords)
        assert affine_indices.shape == (3, 3)
        
        for i, coord in enumerate(test_coords):
            sitk_idx = sitk_image.TransformPhysicalPointToContinuousIndex(coord.tolist())
            assert np.allclose(affine_indices[i], sitk_idx, atol=1e-6)

    def test_roundtrip_index_coord_index(self, affine_from_sitk):
        """Test that index -> coord -> index roundtrip is identity."""
        indices = np.array([
            [0, 0, 0],
            [5, 10, 15],
            [9.5, 19.5, 29.5],  # fractional indices
        ])
        
        coords = affine_from_sitk.index_to_coord(indices)
        recovered_indices = affine_from_sitk.coord_to_index(coords)
        assert np.allclose(recovered_indices, indices, atol=1e-10)

    def test_roundtrip_coord_index_coord(self, affine_from_sitk):
        """Test that coord -> index -> coord roundtrip is identity."""
        coords = np.array([
            [-10.0, 20.0, 30.0],
            [-7.5, 30.0, 52.5],
            [-5.5, 39.0, 73.5],
        ])
        
        indices = affine_from_sitk.coord_to_index(coords)
        recovered_coords = affine_from_sitk.index_to_coord(indices)
        assert np.allclose(recovered_coords, coords, atol=1e-10)


@pytest.mark.skipif(not HAS_SITK, reason="SimpleITK not installed")
class TestAffinePermute:
    """Test permute_axes against SimpleITK PermuteAxes."""

    def test_permute_identity(self):
        """Test that identity permutation doesn't change affine."""
        affine = Affine(spacing=[1, 2, 3], origin=[10, 20, 30])
        permuted = affine.permute_axes((0, 1, 2))
        assert np.allclose(permuted.matrix, affine.matrix)

    def test_permute_reverse(self):
        """Test reversing axes (z, y, x) <- (x, y, z)."""
        affine = Affine(spacing=[1, 2, 3], origin=[10, 20, 30])
        permuted = affine.permute_axes((2, 1, 0))
        
        assert np.allclose(permuted.spacing, [3, 2, 1])
        assert np.allclose(permuted.origin, [10, 20, 30])  # origin unchanged

    def test_permute_vs_sitk(self):
        """Test permute_axes matches SimpleITK PermuteAxes behavior."""
        # Create SimpleITK image
        img = sitk.Image([10, 20, 30], sitk.sitkFloat32)
        img.SetSpacing([0.5, 1.0, 1.5])
        img.SetOrigin([-10.0, 20.0, 30.0])
        img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        
        # Create Affine
        affine = Affine(
            spacing=np.array(img.GetSpacing()),
            origin=np.array(img.GetOrigin()),
            direction=np.array(img.GetDirection()).reshape(3, 3)
        )
        
        # Test permutation (2, 1, 0) - reverse axes
        permuted_sitk = sitk.PermuteAxes(img, [2, 1, 0])
        permuted_affine = affine.permute_axes((2, 1, 0))
        
        # Compare transformations
        test_idx = [5, 10, 15]
        sitk_coord = permuted_sitk.TransformIndexToPhysicalPoint(test_idx)
        affine_coord = permuted_affine.index_to_coord(np.array(test_idx))
        
        assert np.allclose(affine_coord, sitk_coord, atol=1e-6)

    def test_permute_invalid_axes(self):
        """Test that invalid permutation raises ValueError."""
        affine = Affine()
        with pytest.raises(ValueError, match="Invalid axes"):
            affine.permute_axes((0, 1, 1))  # duplicate axis


@pytest.mark.skipif(not HAS_SITK, reason="SimpleITK not installed")
class TestAffineFlip:
    """Test flip_axes against SimpleITK Flip."""

    def test_flip_single_axis(self):
        """Test flipping a single axis."""
        affine = Affine(spacing=[1, 2, 3], origin=[10, 20, 30])
        shape = (10, 20, 30)
        
        flipped = affine.flip_axes([0], shape=shape)
        
        # Spacing sign should flip
        assert np.allclose(np.abs(flipped.spacing), [1, 2, 3])
        # Origin should be updated
        assert flipped.origin[0] != affine.origin[0]

    def test_flip_multiple_axes(self):
        """Test flipping multiple axes."""
        affine = Affine(spacing=[1, 2, 3], origin=[0, 0, 0])
        shape = (10, 20, 30)
        
        flipped = affine.flip_axes([0, 2], shape=shape)
        
        # Check that transformation is correct
        idx = np.array([0, 0, 0])
        coord_orig = affine.index_to_coord(idx)
        
        # After flip, index [0,0,0] should map to where [9,0,29] mapped originally
        idx_flipped = np.array([9, 0, 29])
        coord_expected = affine.index_to_coord(idx_flipped)
        coord_flipped = flipped.index_to_coord(idx)
        
        assert np.allclose(coord_flipped, coord_expected, atol=1e-6)

    def test_flip_vs_sitk(self):
        """Test flip_axes matches SimpleITK Flip behavior."""
        # Create SimpleITK image
        img = sitk.Image([10, 20, 30], sitk.sitkFloat32)
        img.SetSpacing([1.0, 2.0, 3.0])
        img.SetOrigin([0.0, 0.0, 0.0])
        img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        
        # Create Affine
        affine = Affine(
            spacing=np.array(img.GetSpacing()),
            origin=np.array(img.GetOrigin()),
            direction=np.array(img.GetDirection()).reshape(3, 3)
        )
        
        # Test flipping axes [True, False, True]
        flipped_sitk = sitk.Flip(img, [True, False, True])
        flipped_affine = affine.flip_axes([0, 2], shape=(10, 20, 30))
        
        # Compare transformations for corner points
        test_indices = [[0, 0, 0], [9, 19, 29], [5, 10, 15]]
        for idx in test_indices:
            sitk_coord = flipped_sitk.TransformIndexToPhysicalPoint(idx)
            affine_coord = flipped_affine.index_to_coord(np.array(idx))
            assert np.allclose(affine_coord, sitk_coord, atol=1e-6), \
                f"Mismatch at index {idx}"

    def test_flip_requires_shape(self):
        """Test that flip_axes requires shape parameter."""
        affine = Affine()
        with pytest.raises(ValueError, match="Shape must be provided"):
            affine.flip_axes([0])


class TestAffineMethods:
    """Test other Affine methods."""

    def test_clone(self):
        """Test that clone creates independent copy."""
        affine = Affine(spacing=[1, 2, 3], origin=[10, 20, 30])
        cloned = affine.clone()
        
        assert np.allclose(cloned.matrix, affine.matrix)
        
        # Modify clone
        cloned.spacing = [4, 5, 6]
        # Original unchanged
        assert np.allclose(affine.spacing, [1, 2, 3])

    def test_inverse(self):
        """Test inverse affine."""
        affine = Affine(spacing=[1, 2, 3], origin=[10, 20, 30])
        inv = affine.inverse()
        
        # Composition should give identity
        composed = affine.compose_with(inv)
        assert np.allclose(composed.matrix, np.eye(4), atol=1e-10)

    def test_compose_with(self):
        """Test composing two affines."""
        affine1 = Affine(spacing=[2, 2, 2], origin=[0, 0, 0])
        affine2 = Affine(spacing=[1, 1, 1], origin=[10, 20, 30])
        
        composed = affine1.compose_with(affine2)
        
        # Test that composition works correctly
        idx = np.array([1, 2, 3])
        coord1 = affine2.index_to_coord(idx)
        coord2 = affine1.index_to_coord(coord1)
        coord_direct = composed.index_to_coord(idx)
        
        assert np.allclose(coord2, coord_direct, atol=1e-10)

    def test_equality(self):
        """Test affine equality comparison."""
        affine1 = Affine(spacing=[1, 2, 3])
        affine2 = Affine(spacing=[1, 2, 3])
        affine3 = Affine(spacing=[1, 2, 4])
        
        assert affine1 == affine2
        assert affine1 != affine3

    def test_repr(self):
        """Test string representation."""
        affine = Affine(spacing=[1, 2, 3], origin=[10, 20, 30])
        repr_str = repr(affine)
        assert "Affine" in repr_str
        assert "origin" in repr_str
        assert "spacing" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
