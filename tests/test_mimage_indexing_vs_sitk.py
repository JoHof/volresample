"""Comprehensive unit tests for Mimage.__getitem__ vs SimpleITK indexing.

Tests verify that Mimage indexing matches SimpleITK's behavior for:
- Simple slicing with start/stop
- Strided slicing (::step)
- Reverse slicing (::-1)
- Combined start:stop:step
- Multi-dimensional slicing
- Negative indices
"""

import pytest
import numpy as np
from mimage.mimage import Mimage
from mimage.affine import Affine

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


@pytest.mark.skipif(not HAS_SITK, reason="SimpleITK not installed")
class TestMimageIndexingVsSimpleITK:
    """Test Mimage indexing matches SimpleITK behavior."""
    
    def create_test_images(self):
        """Create matching SimpleITK and Mimage test images."""
        # Create SimpleITK image
        arr = np.arange(10*20*30, dtype=np.float32).reshape(30, 20, 10)
        sitk_img = sitk.GetImageFromArray(arr)
        sitk_img.SetSpacing([1.5, 2.0, 2.5])
        sitk_img.SetOrigin([10.0, 20.0, 30.0])
        sitk_img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        
        # Create matching Mimage
        mimg = Mimage.from_sitk(sitk_img)
        
        return sitk_img, mimg
    
    def assert_images_match(self, sitk_img, mimg, msg=""):
        """Assert that SimpleITK and Mimage have matching properties."""
        # Size (SimpleITK uses x,y,z order, Mimage uses z,y,x for data)
        sitk_size = sitk_img.GetSize()
        mimg_size = mimg.shape
        assert mimg_size == tuple(reversed(sitk_size)), f"{msg}: Size mismatch"
        
        # Spacing (need to account for axis order)
        sitk_spacing = np.array(sitk_img.GetSpacing())
        mimg_spacing = mimg.spacing
        # After from_sitk, spacing is permuted to match array order (z,y,x)
        expected_spacing = sitk_spacing[::-1]
        assert np.allclose(mimg_spacing, expected_spacing), f"{msg}: Spacing mismatch"
        
        # Origin
        sitk_origin = np.array(sitk_img.GetOrigin())
        mimg_origin = mimg.origin
        assert np.allclose(mimg_origin, sitk_origin), f"{msg}: Origin mismatch"
        
        # Direction (3x3 matrix)
        sitk_direction = np.array(sitk_img.GetDirection()).reshape(3, 3)
        mimg_direction = mimg.affine.permute_axes((2, 1, 0)).direction  # Convert back to SimpleITK order
        assert np.allclose(mimg_direction, sitk_direction), f"{msg}: Direction mismatch"
        
        # Data
        sitk_arr = sitk.GetArrayFromImage(sitk_img)
        mimg_arr = mimg.data
        assert np.allclose(sitk_arr, mimg_arr), f"{msg}: Data mismatch"
    
    def test_simple_slice_start_stop(self):
        """Test simple slicing img[start:stop, :, :]."""
        sitk_img, mimg = self.create_test_images()
        
        # Test slicing first dimension
        sitk_sliced = sitk_img[5:15, :, :]
        mimg_sliced = mimg[:, :, 5:15]  # Remember: Mimage uses z,y,x order
        
        self.assert_images_match(sitk_sliced, mimg_sliced, "Simple slice [5:15]")
    
    def test_simple_slice_with_start_offset(self):
        """Test that origin shifts correctly with start index."""
        sitk_img, mimg = self.create_test_images()
        
        # Various start offsets
        for start in [0, 3, 7]:
            sitk_sliced = sitk_img[start:, :, :]
            mimg_sliced = mimg[:, :, start:]
            
            self.assert_images_match(sitk_sliced, mimg_sliced, f"Slice [{start}:]")
    
    def test_reverse_slice_single_axis(self):
        """Test reverse slicing img[::-1, :, :] flips direction."""
        sitk_img, mimg = self.create_test_images()
        
        # Reverse first axis
        sitk_sliced = sitk_img[::-1, :, :]
        mimg_sliced = mimg[:, :, ::-1]
        
        self.assert_images_match(sitk_sliced, mimg_sliced, "Reverse slice [::-1]")
    
    def test_reverse_slice_multiple_axes(self):
        """Test reversing multiple axes."""
        sitk_img, mimg = self.create_test_images()
        
        # Reverse all three axes
        sitk_sliced = sitk_img[::-1, ::-1, ::-1]
        mimg_sliced = mimg[::-1, ::-1, ::-1]
        
        self.assert_images_match(sitk_sliced, mimg_sliced, "Reverse all [::-1, ::-1, ::-1]")
    
    def test_stride_slice_step_2(self):
        """Test strided slicing img[::2, :, :] scales spacing."""
        sitk_img, mimg = self.create_test_images()
        
        # Stride by 2 on first axis
        sitk_sliced = sitk_img[::2, :, :]
        mimg_sliced = mimg[:, :, ::2]
        
        self.assert_images_match(sitk_sliced, mimg_sliced, "Stride [::2]")
    
    def test_stride_slice_step_3(self):
        """Test stride of 3."""
        sitk_img, mimg = self.create_test_images()
        
        sitk_sliced = sitk_img[::3, ::2, :]
        mimg_sliced = mimg[:, ::2, ::3]
        
        self.assert_images_match(sitk_sliced, mimg_sliced, "Stride [::3, ::2, :]")
    
    def test_combined_start_stop_step(self):
        """Test img[start:stop:step] with all parameters."""
        sitk_img, mimg = self.create_test_images()
        
        # Combined: start, stop, step
        sitk_sliced = sitk_img[2:8:2, :, :]
        mimg_sliced = mimg[:, :, 2:8:2]
        
        self.assert_images_match(sitk_sliced, mimg_sliced, "Combined [2:8:2]")
    
    def test_combined_with_reverse(self):
        """Test img[start:stop:-1] reverse with boundaries."""
        sitk_img, mimg = self.create_test_images()
        
        # Reverse with boundaries
        sitk_sliced = sitk_img[8:2:-1, :, :]
        mimg_sliced = mimg[:, :, 8:2:-1]
        
        self.assert_images_match(sitk_sliced, mimg_sliced, "Reverse with bounds [8:2:-1]")
    
    def test_multidimensional_slicing(self):
        """Test complex multi-dimensional slicing."""
        sitk_img, mimg = self.create_test_images()
        
        sitk_sliced = sitk_img[2:8:2, 5:15, ::3]
        mimg_sliced = mimg[::3, 5:15, 2:8:2]
        
        self.assert_images_match(sitk_sliced, mimg_sliced, "Multi-dim [2:8:2, 5:15, ::3]")
    
    def test_negative_start_index(self):
        """Test negative indices in start position."""
        sitk_img, mimg = self.create_test_images()
        
        sitk_sliced = sitk_img[-5:, :, :]
        mimg_sliced = mimg[:, :, -5:]
        
        self.assert_images_match(sitk_sliced, mimg_sliced, "Negative start [-5:]")
    
    def test_negative_stop_index(self):
        """Test negative indices in stop position."""
        sitk_img, mimg = self.create_test_images()
        
        sitk_sliced = sitk_img[:-5, :, :]
        mimg_sliced = mimg[:, :, :-5]
        
        self.assert_images_match(sitk_sliced, mimg_sliced, "Negative stop [:-5]")
    
    def test_coordinate_transform_after_stride(self):
        """Test that coordinate transforms work correctly after strided slicing."""
        sitk_img, mimg = self.create_test_images()
        
        # Slice with stride
        sitk_sliced = sitk_img[::2, ::3, :]
        mimg_sliced = mimg[:, ::3, ::2]
        
        # Test a few coordinate transformations
        test_indices = [[0, 0, 0], [1, 2, 3], [2, 5, 10]]
        
        for idx in test_indices:
            # SimpleITK order: x, y, z
            sitk_coord = sitk_sliced.TransformIndexToPhysicalPoint(idx)
            # Mimage order: z, y, x (reversed)
            mimg_coord = mimg_sliced.index_to_coord(idx[::-1])
            
            assert np.allclose(mimg_coord, sitk_coord), \
                f"Coordinate mismatch for index {idx}"
    
    def test_coordinate_transform_after_reverse(self):
        """Test coordinate transforms after reversing."""
        sitk_img, mimg = self.create_test_images()
        
        # Reverse first axis
        sitk_sliced = sitk_img[::-1, :, :]
        mimg_sliced = mimg[:, :, ::-1]
        
        # Test coordinate transformation
        idx = [5, 10, 15]
        sitk_coord = sitk_sliced.TransformIndexToPhysicalPoint(idx)
        mimg_coord = mimg_sliced.index_to_coord(idx[::-1])
        
        assert np.allclose(mimg_coord, sitk_coord), \
            f"Coordinate mismatch after reverse"
    
    def test_full_slice_no_change(self):
        """Test that img[:, :, :] doesn't change properties."""
        sitk_img, mimg = self.create_test_images()
        
        sitk_sliced = sitk_img[:, :, :]
        mimg_sliced = mimg[:, :, :]
        
        self.assert_images_match(sitk_sliced, mimg_sliced, "Full slice [:, :, :]")
