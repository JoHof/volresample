"""Numpy backend for image resampling using scipy.ndimage."""

import numpy as np
from scipy.ndimage import map_coordinates
from typing import Tuple

import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def resample_linear_3d_numba(arr, new_shape):
    """Resample a 3D array using trilinear interpolation (Numba-accelerated)."""
    z, y, x = arr.shape
    new_z, new_y, new_x = new_shape
    out = np.empty((new_z, new_y, new_x), dtype=arr.dtype)

    scale_z = (z - 1) / (new_z - 1) if new_z > 1 else 0
    scale_y = (y - 1) / (new_y - 1) if new_y > 1 else 0
    scale_x = (x - 1) / (new_x - 1) if new_x > 1 else 0

    for i in prange(new_z):
        zf = i * scale_z
        z0 = int(np.floor(zf))
        z1 = min(z0 + 1, z - 1)
        dz = zf - z0

        for j in range(new_y):
            yf = j * scale_y
            y0 = int(np.floor(yf))
            y1 = min(y0 + 1, y - 1)
            dy = yf - y0

            for k in range(new_x):
                xf = k * scale_x
                x0 = int(np.floor(xf))
                x1 = min(x0 + 1, x - 1)
                dx = xf - x0

                # Fetch 8 corner values
                c000 = arr[z0, y0, x0]
                c001 = arr[z0, y0, x1]
                c010 = arr[z0, y1, x0]
                c011 = arr[z0, y1, x1]
                c100 = arr[z1, y0, x0]
                c101 = arr[z1, y0, x1]
                c110 = arr[z1, y1, x0]
                c111 = arr[z1, y1, x1]

                # Trilinear interpolation
                c00 = c000 * (1 - dx) + c001 * dx
                c01 = c010 * (1 - dx) + c011 * dx
                c10 = c100 * (1 - dx) + c101 * dx
                c11 = c110 * (1 - dx) + c111 * dx

                c0 = c00 * (1 - dy) + c01 * dy
                c1 = c10 * (1 - dy) + c11 * dy

                out[i, j, k] = c0 * (1 - dz) + c1 * dz

    return out


class ResamplingNumpyBackend:
    """Numpy-based resampling backend using scipy.ndimage.map_coordinates."""
    
    available = True
    
    @staticmethod
    def resample(
        data: np.ndarray,
        size: Tuple[int, int, int],
        mode: str = "linear"
    ) -> np.ndarray:
        """Resample a 3D volume to a new size using interpolation.
        
        This function operates on 3D arrays only. For arrays with additional
        dimensions, the caller must iterate over non-spatial dimensions.
        
        Args:
            data: Input 3D array with shape (D, H, W)
            size: Target size (new_D, new_H, new_W) for spatial dimensions
            mode: Interpolation mode:
                  - 'nearest': Nearest neighbor (order=0)
                  - 'linear': Trilinear interpolation (order=1)
                  - 'cubic': Tricubic interpolation (order=3)
                  
        Returns:
            Resampled array with shape (new_D, new_H, new_W)
            
        Raises:
            ValueError: If mode is not supported or data is not 3D
        """
        if data.ndim != 3:
            raise ValueError(f"Data must be exactly 3D, got {data.ndim} dimensions")
        
        # Map mode to scipy order parameter
        order_map = {
            'nearest': 0,
            'linear': 1,
            'cubic': 3
        }
        
        if mode not in order_map:
            raise ValueError(
                f"Unsupported interpolation mode: {mode}. "
                f"Use 'nearest', 'linear', or 'cubic'."
            )
        
        order = order_map[mode]
        
        # Get current spatial shape
        current_shape = data.shape
        new_D, new_H, new_W = size
        
        # Create sampling coordinates in the INPUT space
        # We map from output coordinates to input coordinates
        scale_factors = np.array([
            (current_shape[0] - 1) / max(new_D - 1, 1),
            (current_shape[1] - 1) / max(new_H - 1, 1),
            (current_shape[2] - 1) / max(new_W - 1, 1)
        ])
        
        # Create coordinate grid for output volume
        out_coords = np.meshgrid(
            np.arange(new_D),
            np.arange(new_H),
            np.arange(new_W),
            indexing='ij'
        )
        
        # Map output coordinates to input coordinates
        input_coords = np.stack([
            out_coords[0] * scale_factors[0],
            out_coords[1] * scale_factors[1],
            out_coords[2] * scale_factors[2]
        ], axis=0)
        
        # Resample
        output = map_coordinates(
            data,
            input_coords,
            order=order,
            mode='constant',
            cval=0.0,
            prefilter=order > 1  # Use prefilter for cubic
        )
        
        return output
