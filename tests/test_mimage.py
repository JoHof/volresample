"""Unit tests for mimage.py module.

Tests Mimage class functionality against SimpleITK reference implementation.
"""

import pytest
import numpy as np
from mimage.mimage import Mimage
from mimage.affine import Affine

# Optional dependencies
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestMimageConstruction:
    """Test Mimage construction with different backends."""

    def test_numpy_backend_auto(self):
        """Test Mimage with numpy array (auto backend)."""
        data = np.random.rand(10, 20, 30).astype(np.float32)
        img = Mimage(data)
        assert img.backend == 'numpy'
        assert isinstance(img.data, np.ndarray)
        assert img.data.shape == (10, 20, 30)

    def test_numpy_backend_explicit(self):
        """Test Mimage with explicit numpy backend."""
        data = np.random.rand(10, 20, 30).astype(np.float32)
        img = Mimage(data, backend='numpy')
        assert img.backend == 'numpy'
        assert isinstance(img.data, np.ndarray)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_backend_auto(self):
        """Test Mimage with torch tensor (auto backend)."""
        data = torch.randn(10, 20, 30)
        img = Mimage(data)
        assert img.backend == 'torch'
        assert isinstance(img.data, torch.Tensor)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_backend_explicit(self):
        """Test Mimage with explicit torch backend."""
        data = np.random.rand(10, 20, 30).astype(np.float32)
        img = Mimage(data, backend='torch')
        assert img.backend == 'torch'
        assert isinstance(img.data, torch.Tensor)

    def test_with_affine(self):
        """Test Mimage with custom affine."""
        data = np.random.rand(10, 20, 30).astype(np.float32)
        affine = Affine(spacing=[1, 2, 3], origin=[10, 20, 30])
        img = Mimage(data, affine=affine)
        assert np.allclose(img.spacing, [1, 2, 3])
        assert np.allclose(img.origin, [10, 20, 30])

    def test_with_spatial_dims(self):
        """Test Mimage with explicit spatial_dims."""
        data = np.random.rand(5, 10, 20, 30).astype(np.float32)
        img = Mimage(data, spatial_dims=(1, 2, 3))
        assert img.spatial_dims == (1, 2, 3)

    def test_default_spatial_dims(self):
        """Test that spatial_dims defaults to last 3 dimensions."""
        data = np.random.rand(10, 20, 30).astype(np.float32)
        img = Mimage(data)
        assert img.spatial_dims == (0, 1, 2)

        data_4d = np.random.rand(5, 10, 20, 30).astype(np.float32)
        img_4d = Mimage(data_4d)
        assert img_4d.spatial_dims == (1, 2, 3)

    def test_invalid_backend(self):
        """Test that invalid backend raises ValueError."""
        data = np.random.rand(10, 20, 30).astype(np.float32)
        with pytest.raises(ValueError, match="backend must be"):
            Mimage(data, backend='invalid')


class TestMimageProperties:
    """Test Mimage property access."""

    def test_spacing_property(self):
        """Test spacing property."""
        affine = Affine(spacing=[1, 2, 3])
        img = Mimage(np.ones((10, 20, 30)), affine=affine)
        assert np.allclose(img.spacing, [1, 2, 3])

    def test_origin_property(self):
        """Test origin property."""
        affine = Affine(origin=[10, 20, 30])
        img = Mimage(np.ones((10, 20, 30)), affine=affine)
        assert np.allclose(img.origin, [10, 20, 30])

    def test_direction_property(self):
        """Test direction property."""
        direction = np.eye(3)
        affine = Affine(direction=direction)
        img = Mimage(np.ones((10, 20, 30)), affine=affine)
        assert np.allclose(img.direction, direction)

    def test_shape_property(self):
        """Test shape property."""
        img = Mimage(np.ones((10, 20, 30)))
        assert img.shape == (10, 20, 30)

    def test_repr(self):
        """Test string representation."""
        img = Mimage(np.ones((10, 20, 30)))
        repr_str = repr(img)
        assert "Mimage" in repr_str
        assert "shape" in repr_str


@pytest.mark.skipif(not HAS_SITK, reason="SimpleITK not installed")
class TestMimageSimpleITKConversion:
    """Test Mimage conversion to/from SimpleITK."""

    @pytest.fixture
    def sitk_image(self):
        """Create a SimpleITK test image."""
        img = sitk.Image([10, 20, 30], sitk.sitkFloat32)
        img.SetSpacing([0.5, 1.0, 1.5])
        img.SetOrigin([-10.0, 20.0, 30.0])
        img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        
        # Fill with test data
        arr = np.random.rand(30, 20, 10).astype(np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing([0.5, 1.0, 1.5])
        img.SetOrigin([-10.0, 20.0, 30.0])
        img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        return img

    def test_from_sitk(self, sitk_image):
        """Test creating Mimage from SimpleITK image."""
        img = Mimage.from_sitk(sitk_image)
        
        # Check shape (note: SimpleITK uses (z, y, x) order)
        sitk_arr = sitk.GetArrayFromImage(sitk_image)
        assert img.shape == sitk_arr.shape
        assert np.allclose(img.data, sitk_arr)

    def test_to_sitk(self, sitk_image):
        """Test converting Mimage to SimpleITK image."""
        img = Mimage.from_sitk(sitk_image)
        sitk_out = img.to_sitk()
        
        # Compare metadata
        assert np.allclose(sitk_out.GetSpacing(), sitk_image.GetSpacing(), atol=1e-6)
        assert np.allclose(sitk_out.GetOrigin(), sitk_image.GetOrigin(), atol=1e-6)
        assert np.allclose(sitk_out.GetDirection(), sitk_image.GetDirection(), atol=1e-6)
        
        # Compare data
        assert np.allclose(sitk.GetArrayFromImage(sitk_out), 
                          sitk.GetArrayFromImage(sitk_image))

    def test_roundtrip_sitk_mimage_sitk(self, sitk_image):
        """Test SimpleITK -> Mimage -> SimpleITK roundtrip."""
        img = Mimage.from_sitk(sitk_image)
        sitk_out = img.to_sitk()
        
        # Should be identical after roundtrip
        assert np.allclose(sitk_out.GetSpacing(), sitk_image.GetSpacing(), atol=1e-6)
        assert np.allclose(sitk_out.GetOrigin(), sitk_image.GetOrigin(), atol=1e-6)
        assert np.allclose(sitk_out.GetDirection(), sitk_image.GetDirection(), atol=1e-6)

    def test_index_to_coord_vs_sitk(self, sitk_image):
        """Test index_to_coord matches SimpleITK TransformIndexToPhysicalPoint."""
        img = Mimage.from_sitk(sitk_image)
        
        # Mimage uses (z, y, x) order (numpy convention)
        # SimpleITK uses (x, y, z) order
        test_indices_mimage = [
            [0, 0, 0],
            [5, 10, 15],
            [29, 19, 9],  # near max
        ]
        
        for idx_mimg in test_indices_mimage:
            # Mimage: (z, y, x) indexing
            mimage_coord = img.index_to_coord(idx_mimg)
            
            # SimpleITK: (x, y, z) indexing - reverse the order
            idx_sitk = [idx_mimg[2], idx_mimg[1], idx_mimg[0]]
            sitk_coord = sitk_image.TransformIndexToPhysicalPoint(idx_sitk)
            
            # Compare coordinates (both in physical space, same order)
            assert np.allclose(mimage_coord, sitk_coord, atol=1e-6), \
                f"Index {idx_mimg}: Mimage coord {mimage_coord} != SimpleITK coord {sitk_coord}"

    def test_coord_to_index_vs_sitk(self, sitk_image):
        """Test coord_to_index matches SimpleITK TransformPhysicalPointToContinuousIndex."""
        img = Mimage.from_sitk(sitk_image)
        
        # Test coordinates in physical space
        test_coords = [
            [-10.0, 20.0, 30.0],  # origin
            [-7.5, 30.0, 52.5],   # middle point
            [-5.5, 39.0, 73.5],   # near end
        ]
        
        for coord in test_coords:
            # Mimage coord_to_index
            mimage_idx = img.coord_to_index(coord)
            
            # SimpleITK expects same physical coordinates
            sitk_idx = sitk_image.TransformPhysicalPointToContinuousIndex(coord)
            
            # Convert SimpleITK (x,y,z) to Mimage (z,y,x) order for comparison
            sitk_idx_zyx = [sitk_idx[2], sitk_idx[1], sitk_idx[0]]
            
            assert np.allclose(mimage_idx, sitk_idx_zyx, atol=1e-6), \
                f"Coord {coord}: Mimage index {mimage_idx} != SimpleITK index {sitk_idx_zyx}"


@pytest.mark.skipif(not HAS_SITK, reason="SimpleITK not installed")
class TestMimagePermute:
    """Test Mimage permute against SimpleITK PermuteAxes."""

    @pytest.fixture
    def sitk_image(self):
        """Create a SimpleITK test image."""
        arr = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing([1.0, 2.0, 3.0])
        img.SetOrigin([10.0, 20.0, 30.0])
        img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        return img

    def test_permute_identity(self):
        """Test identity permutation."""
        img = Mimage(np.arange(24).reshape(2, 3, 4).astype(np.float32))
        permuted = img.permute(0, 1, 2)
        assert permuted.shape == img.shape
        assert np.allclose(permuted.data, img.data)

    def test_permute_reverse(self):
        """Test reversing all axes."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        img = Mimage(data)
        permuted = img.permute(2, 1, 0)
        
        assert permuted.shape == (4, 3, 2)
        assert np.allclose(permuted.data, np.transpose(data, (2, 1, 0)))

    def test_permute_vs_sitk(self, sitk_image):
        """Test permute matches SimpleITK PermuteAxes behavior."""
        img = Mimage.from_sitk(sitk_image)
        
        # Test permutation (2, 1, 0) - reverse axes
        permuted_mimage = img.permute(2, 1, 0)
        
        # Apply same permutation in SimpleITK
        # Note: SimpleITK PermuteAxes expects order as [x, y, z]
        # Mimage permute(2,1,0) means: new[0]=old[2], new[1]=old[1], new[2]=old[0]
        # In SimpleITK (x,y,z) this is [2,1,0]
        permuted_sitk = sitk.PermuteAxes(sitk_image, [2, 1, 0])
        
        # Compare by checking that index->coord transformation matches
        # Test several indices
        test_indices_mimage = [[0, 0, 0], [1, 2, 3]]
        
        for idx_m in test_indices_mimage:
            # Mimage coordinate
            mimage_coord = permuted_mimage.index_to_coord(idx_m)
            
            # SimpleITK coordinate (reverse index order)
            idx_sitk = [idx_m[2], idx_m[1], idx_m[0]]
            sitk_coord = permuted_sitk.TransformIndexToPhysicalPoint(idx_sitk)
            
            assert np.allclose(mimage_coord, sitk_coord, atol=1e-6), \
                f"Permuted index {idx_m}: Mimage coord {mimage_coord} != SimpleITK coord {sitk_coord}"
        
        # Also compare data shapes
        assert permuted_mimage.shape == sitk.GetArrayFromImage(permuted_sitk).shape

    def test_permute_updates_spatial_dims(self):
        """Test that permute updates spatial_dims correctly (always sorted)."""
        data = np.arange(120).reshape(5, 2, 3, 4).astype(np.float32)
        img = Mimage(data, spatial_dims=(1, 2, 3))
        
        # Test permute that keeps spatial dims at end
        permuted = img.permute(0, 3, 2, 1)
        # Original spatial_dims: (1, 2, 3)
        # After permute (0, 3, 2, 1): 
        #   - dim 1 goes to position 3
        #   - dim 2 goes to position 2
        #   - dim 3 goes to position 1
        # Unsorted: (3, 2, 1), but spatial_dims are always sorted -> (1, 2, 3)
        assert permuted.spatial_dims == (1, 2, 3)
        
        # Test permute that scatters spatial dims
        permuted2 = img.permute(3, 0, 2, 1)
        # After permute (3, 0, 2, 1):
        #   - dim 1 goes to position 3
        #   - dim 2 goes to position 2
        #   - dim 3 goes to position 0
        # Unsorted: (3, 2, 0), but spatial_dims are always sorted -> (0, 2, 3)
        assert permuted2.spatial_dims == (0, 2, 3)


@pytest.mark.skipif(not HAS_SITK, reason="SimpleITK not installed")
class TestMimageFlip:
    """Test Mimage flip against SimpleITK Flip."""

    @pytest.fixture
    def sitk_image(self):
        """Create a SimpleITK test image."""
        arr = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing([1.0, 2.0, 3.0])
        img.SetOrigin([0.0, 0.0, 0.0])
        img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        return img

    def test_flip_single_axis(self):
        """Test flipping a single axis."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        img = Mimage(data)
        flipped = img.flip([0])
        
        assert flipped.shape == img.shape
        assert np.allclose(flipped.data, np.flip(data, axis=0))

    def test_flip_multiple_axes(self):
        """Test flipping multiple axes."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        img = Mimage(data)
        flipped = img.flip([0, 2])
        
        expected = np.flip(np.flip(data, axis=0), axis=2)
        assert np.allclose(flipped.data, expected)

    def test_flip_vs_sitk(self, sitk_image):
        """Test flip matches SimpleITK Flip behavior."""
        img = Mimage.from_sitk(sitk_image)
        
        # Flip first and last axes in Mimage (axes 0 and 2)
        flipped_mimage = img.flip([0, 2])
        
        # SimpleITK Flip expects [flipX, flipY, flipZ]
        # Mimage axes [0,2] correspond to z and x, so flip[True, False, True]
        flipped_sitk = sitk.Flip(sitk_image, [True, False, True])
        
        # Compare transformations at several points
        test_indices_mimage = [[0, 0, 0], [1, 2, 3]]
        
        for idx_m in test_indices_mimage:
            # Mimage coordinate
            mimage_coord = flipped_mimage.index_to_coord(idx_m)
            
            # SimpleITK coordinate (reverse index order for x,y,z)
            idx_sitk = [idx_m[2], idx_m[1], idx_m[0]]
            sitk_coord = flipped_sitk.TransformIndexToPhysicalPoint(idx_sitk)
            
            assert np.allclose(mimage_coord, sitk_coord, atol=1e-6), \
                f"Flipped index {idx_m}: Mimage coord {mimage_coord} != SimpleITK coord {sitk_coord}"
        
        # Also compare data (should be flipped)
        mimage_arr = flipped_mimage.data
        sitk_arr = sitk.GetArrayFromImage(flipped_sitk)
        assert np.allclose(mimage_arr, sitk_arr)


class TestMimageIndexing:
    """Test Mimage indexing (__getitem__)."""

    def test_integer_indexing_spatial_error(self):
        """Test that integer indexing on spatial dimension raises error."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        img = Mimage(data)  # spatial_dims = (0, 1, 2)
        
        # Attempting to remove a spatial dimension should raise ValueError
        with pytest.raises(ValueError, match="Cannot remove spatial dimensions"):
            sliced = img[0]  # Dimension 0 is spatial

    def test_integer_indexing_nonspatial(self):
        """Test integer indexing on non-spatial dimension works."""
        data = np.arange(120).reshape(5, 2, 3, 4).astype(np.float32)
        img = Mimage(data)  # spatial_dims = (1, 2, 3), dim 0 is non-spatial
        
        sliced = img[0]
        assert sliced.shape == (2, 3, 4)
        assert sliced.spatial_dims == (0, 1, 2)  # Remapped after removing dim 0
        assert np.allclose(sliced.data, data[0])

    def test_slice_indexing(self):
        """Test slice indexing keeps dimension."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        img = Mimage(data)
        
        sliced = img[0:1]
        assert sliced.shape == (1, 3, 4)
        assert np.allclose(sliced.data, data[0:1])

    def test_slice_with_start(self):
        """Test that slice start updates affine origin."""
        affine = Affine(spacing=[1, 2, 3], origin=[0, 0, 0])
        img = Mimage(np.ones((10, 20, 30)), affine=affine)
        
        sliced = img[5:]
        # Origin should be shifted by 5 * affine_column_0
        assert np.allclose(sliced.origin[0], 5 * 1)

        """Test that slice start updates affine origin after permute."""
        affine = Affine(spacing=[1, 2, 3], origin=[0, 0, 0])
        img = Mimage(np.ones((10, 20, 30)), affine=affine).permute(2, 1, 0)
        
        sliced = img[5:]
        # After permute(2,1,0), column 0 is [0, 0, 3], so shift is [0, 0, 15]
        assert np.allclose(sliced.origin[2], 5 * 3)

    def test_newaxis_indexing(self):
        """Test None/newaxis adds dimension."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        img = Mimage(data)
        
        expanded = img[None]
        assert expanded.shape == (1, 2, 3, 4)

    def test_newaxis_updates_spatial_dims(self):
        """Test that newaxis updates spatial_dims."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        img = Mimage(data, spatial_dims=(0, 1, 2))
        
        expanded = img[None]
        # Spatial dims should shift by 1
        assert expanded.spatial_dims == (1, 2, 3)

        expanded = img[:,None]
        assert expanded.spatial_dims == (0, 2, 3)

    def test_ellipsis_indexing(self):
        """Test ellipsis indexing keeps dimensions via slicing."""
        data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        img = Mimage(data)
        
        # Use slicing to keep dimension instead of removing it
        sliced = img[..., 0:1]
        assert sliced.shape == (2, 3, 1)
        assert np.allclose(sliced.data, data[..., 0:1])

    def test_mixed_indexing(self):
        """Test mixed integer and slice indexing."""
        data = np.arange(120).reshape(5, 2, 3, 4).astype(np.float32)
        img = Mimage(data, spatial_dims=(1, 2, 3))
        
        sliced = img[0, :, 1:, :]
        assert sliced.shape == (2, 2, 4)

    def test_scalar_return(self):
        """Test that indexing a scalar returns the value, not Mimage."""
        data = np.arange(120).reshape(5, 2, 3, 4).astype(np.float32)
        img = Mimage(data)  # spatial_dims = (1, 2, 3), dim 0 is non-spatial
        
        # Index all dimensions to get scalar
        value = img[0, 0, 0, 0]
        assert isinstance(value, (int, float, np.number))
        assert value == data[0, 0, 0, 0]


class TestMimageClone:
    """Test Mimage clone method."""

    def test_clone_numpy(self):
        """Test cloning numpy-backed Mimage."""
        data = np.random.rand(10, 20, 30).astype(np.float32)
        img = Mimage(data)
        cloned = img.clone()
        
        assert np.allclose(cloned.data, img.data)
        assert cloned.data is not img.data  # Different objects
        
        # Modify clone
        cloned.data[0, 0, 0] = 999
        assert img.data[0, 0, 0] != 999  # Original unchanged

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_clone_torch(self):
        """Test cloning torch-backed Mimage."""
        data = torch.randn(10, 20, 30)
        img = Mimage(data, backend='torch')
        cloned = img.clone()
        
        assert torch.allclose(cloned.data, img.data)
        assert cloned.data is not img.data
        
        # Modify clone
        cloned.data[0, 0, 0] = 999
        assert img.data[0, 0, 0] != 999


class TestMimageCoordTransforms:
    """Test coordinate transformations."""

    def test_index_to_coord_single(self):
        """Test index_to_coord with single point."""
        affine = Affine(spacing=[1, 2, 3], origin=[10, 20, 30])
        img = Mimage(np.ones((10, 20, 30)), affine=affine)
        
        coord = img.index_to_coord([0, 0, 0])
        assert np.allclose(coord, [10, 20, 30])

    def test_index_to_coord_multiple(self):
        """Test index_to_coord with multiple points."""
        affine = Affine(spacing=[1, 2, 3], origin=[10, 20, 30])
        img = Mimage(np.ones((10, 20, 30)), affine=affine)
        
        indices = np.array([[0, 0, 0], [1, 1, 1]])
        coords = img.index_to_coord(indices)
        
        assert coords.shape == (2, 3)
        assert np.allclose(coords[0], [10, 20, 30])
        assert np.allclose(coords[1], [11, 22, 33])

    def test_coord_to_index_single(self):
        """Test coord_to_index with single point."""
        affine = Affine(spacing=[1, 2, 3], origin=[10, 20, 30])
        img = Mimage(np.ones((10, 20, 30)), affine=affine)
        
        idx = img.coord_to_index([10, 20, 30])
        assert np.allclose(idx, [0, 0, 0])

    def test_roundtrip(self):
        """Test index -> coord -> index roundtrip."""
        affine = Affine(spacing=[1, 2, 3], origin=[10, 20, 30])
        img = Mimage(np.ones((10, 20, 30)), affine=affine)
        
        indices = np.array([[0, 0, 0], [5, 10, 15]])
        coords = img.index_to_coord(indices)
        recovered = img.coord_to_index(coords)
        
        assert np.allclose(recovered, indices, atol=1e-10)


class TestMimageClip:
    """Test clip functionality."""

    def test_clip_both_bounds_numpy(self):
        """Test clip with both min and max values (numpy backend)."""
        data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]).astype(np.float32)
        img = Mimage(data)
        
        clipped = img.clip(min_val=3, max_val=9)
        
        assert clipped.data.min() >= 3
        assert clipped.data.max() <= 9
        assert np.allclose(clipped.data, np.clip(data, 3, 9))

    def test_clip_min_only(self):
        """Test clip with only minimum value."""
        data = np.array([[[1, 2, 3], [4, 5, 6]]]).astype(np.float32)
        img = Mimage(data)
        
        clipped = img.clip(min_val=3)
        
        assert clipped.data.min() >= 3
        assert np.allclose(clipped.data, np.clip(data, 3, None))

    def test_clip_max_only(self):
        """Test clip with only maximum value."""
        data = np.array([[[1, 2, 3], [4, 5, 6]]]).astype(np.float32)
        img = Mimage(data)
        
        clipped = img.clip(max_val=4)
        
        assert clipped.data.max() <= 4
        assert np.allclose(clipped.data, np.clip(data, None, 4))

    def test_clip_preserves_affine(self):
        """Test that clip preserves affine transform."""
        affine = Affine(spacing=[1, 2, 3], origin=[10, 20, 30])
        data = np.array([[[1, 2, 3], [4, 5, 6]]]).astype(np.float32)
        img = Mimage(data, affine=affine)
        
        clipped = img.clip(min_val=2, max_val=5)
        
        assert np.allclose(clipped.spacing, img.spacing)
        assert np.allclose(clipped.origin, img.origin)
        assert np.allclose(clipped.direction, img.direction)

    def test_clip_immutable(self):
        """Test that clip returns new instance without modifying original."""
        data = np.array([[[1, 2, 3], [4, 5, 6]]]).astype(np.float32)
        img = Mimage(data)
        original_min = img.data.min()
        
        clipped = img.clip(min_val=3)
        
        assert img.data.min() == original_min  # Original unchanged
        assert clipped.data.min() >= 3
        assert img is not clipped

    def test_clip_no_change(self):
        """Test clip when all values are within range."""
        data = np.array([[[3, 4, 5], [6, 7, 8]]]).astype(np.float32)
        img = Mimage(data)
        
        clipped = img.clip(min_val=2, max_val=10)
        
        assert np.allclose(clipped.data, data)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_clip_both_bounds_torch(self):
        """Test clip with both min and max values (torch backend)."""
        data = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]).float()
        img = Mimage(data, backend='torch')
        
        clipped = img.clip(min_val=3, max_val=9)
        
        assert clipped.data.min() >= 3
        assert clipped.data.max() <= 9
        assert torch.allclose(clipped.data, torch.clamp(data, min=3, max=9))

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_clip_preserves_backend(self):
        """Test that clip preserves backend type."""
        data = torch.randn(2, 3, 4)
        img = Mimage(data, backend='torch')
        
        clipped = img.clip(min_val=-1, max_val=1)
        
        assert isinstance(clipped.data, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
