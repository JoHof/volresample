"""Image resampling functionality for Mimage.

This module provides resampling (interpolation) to change image resolution.
Resampling creates a NEW grid with different spacing, using interpolation to
estimate values, as opposed to stride slicing which simply subsamples existing voxels.
"""

import numpy as np
from typing import Tuple, Union
from mimage.affine import Affine
from mimage.backends.resampling import ResamplingNumpyBackend, ResamplingTorchBackend
from mimage.mimage import Mimage


def resample(
    image: "Mimage",
    shape: Tuple[int, int, int],
    mode: str = "linear"
) -> "Mimage":
    """Resample image to a new shape using interpolation.
    
    This creates a NEW grid with the specified shape, using interpolation to
    estimate values. The spacing is automatically adjusted to maintain the
    same physical extent.
    
    Args:
        image: Input Mimage to resample
        shape: Target spatial shape (D, H, W)
        mode: Interpolation mode:
              - 'nearest': Nearest neighbor (no interpolation)
              - 'linear': Trilinear interpolation (smooth, default)
              - 'cubic': Tricubic interpolation (highest quality)
              
    Returns:
        Mimage: Resampled image with new shape and updated affine.
        
    Example:
        >>> # Original: (128, 256, 256) with spacing (2.0, 1.0, 1.0) mm
        >>> img = Mimage(data, affine=Affine(spacing=[2.0, 1.0, 1.0]))
        >>> 
        >>> # Downsample to specific shape
        >>> img_small = resample(img, shape=(64, 128, 128))
        >>> # Result: (64, 128, 128) with spacing (4.0, 2.0, 2.0) mm
    """
    
    # # Compute target spacing from shape to maintain physical extent
    physical_size = np.array(image.shape_spatial) * image.spacing
    target_spacing = physical_size / np.array(shape)

    # Choose backend (prefer torch if available)
    if ResamplingTorchBackend.available:
        backend = ResamplingTorchBackend
    else:
        backend = ResamplingNumpyBackend
    
    # Prepare data for resampling
    data = image.data
    spatial_dims = image.spatial_dims
    
    # If spatial dims are not at the end, we need to permute
    expected_spatial_dims = tuple(range(data.ndim - 3, data.ndim))
    if spatial_dims != expected_spatial_dims:
        # Permute to move spatial dims to the end
        non_spatial_dims = [i for i in range(data.ndim) if i not in spatial_dims]
        perm = non_spatial_dims + list(spatial_dims)
        data = image._backend.permute(data, perm)
        
        # After resampling, we'll need to permute back
        inv_perm = [perm.index(i) for i in range(data.ndim)]
    else:
        inv_perm = None
    
    # Iterate over non-spatial dimensions and resample each 3D volume
    if data.ndim == 3:
        resampled_data = backend.resample(data, shape, mode)
    else:
        # Complex case: iterate over non-spatial dimensions
        non_spatial_shape = data.shape[:-3]
        batch_size = int(np.prod(non_spatial_shape))
        
        # Reshape to (batch, D, H, W)
        data_reshaped = data.reshape(batch_size, *data.shape[-3:])
        
        # Resample each 3D volume
        resampled_data = backend.resample(data_reshaped, shape, mode)
        resampled_data = resampled_data.reshape(*non_spatial_shape, *shape)
    
    # Permute back if needed
    if inv_perm is not None:
        resampled_data = image._backend.permute(resampled_data, inv_perm)
    
    # Create new affine with updated origin
    target_origin = image.index_to_coord(.5 * (target_spacing / image.spacing) - .5)

    new_affine = Affine(
        direction=image.direction,
        spacing=target_spacing,
        origin=target_origin
    )
    
    # Create new Mimage
    return Mimage(
        resampled_data,
        affine=new_affine,
        spatial_dims=image.spatial_dims,
        backend=image.backend
    )


def resample_to_spacing(
    image: "Mimage",
    spacing: Union[float, Tuple[float, float, float]],
    mode: str = "linear"
) -> "Mimage":
    """Resample image to a new voxel spacing using interpolation.
    
    This is a convenience wrapper around `resample()` that computes the target
    shape from the desired spacing while maintaining the physical extent.
    
    Args:
        image: Input Mimage to resample
        spacing: Target voxel spacing (mm). Can be:
                - float: Isotropic spacing (same for all axes)
                - tuple: Anisotropic spacing
        mode: Interpolation mode:
              - 'nearest': Nearest neighbor
              - 'linear': Trilinear interpolation
              - 'cubic': Tricubic interpolation
              
    Returns:
        Mimage: Resampled image with new spacing and updated affine.
        
    Example:
        >>> # Original: (128, 256, 256) with spacing (2.0, 1.0, 1.0) mm
        >>> img = Mimage(data, affine=Affine(spacing=[2.0, 1.0, 1.0]))
        >>> 
        >>> # Resample to isotropic 1mm spacing
        >>> img_1mm = resample_to_spacing(img, spacing=1.0)
        >>> # Result: (256, 256, 256) with spacing (1.0, 1.0, 1.0) mm
        >>>
        >>> # Resample to anisotropic spacing
        >>> img_aniso = resample_to_spacing(img, spacing=(1.5, 0.8, 1.0))
    """
    # Get current spatial shape and spacing
    current_shape = tuple(image.data.shape[dim] for dim in image.spatial_dims)
    current_spacing = image.spacing
    
    # Convert spacing to array
    if isinstance(spacing, (int, float)):
        target_spacing = np.array([float(spacing)] * 3)
    else:
        target_spacing = np.array(spacing)
    
    if target_spacing.shape != (3,):
        raise ValueError(
            f"Spacing must be scalar or length-3 tuple, got shape {target_spacing.shape}"
        )
    
    # Compute target shape to cover same physical extent
    # physical_size = (shape - 1) * spacing
    physical_size = np.array(current_shape) * current_spacing
    target_shape = tuple((physical_size / target_spacing).astype(int))
    
    # Call the main resample function
    return resample(image, target_shape, mode)
