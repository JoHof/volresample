"""Python wrapper for Cython resampling backend."""

from __future__ import annotations
from typing import Any, Tuple
import numpy as np

try:
    from . import resampling_cython
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


class ResamplingCythonBackend:
    """Cython-based resampling backend with AVX optimization."""

    available = CYTHON_AVAILABLE

    @staticmethod
    def resample(
        data: Any,
        size: Tuple[int, int, int],
        mode: str = "linear",
        parallel_threads: int = 0,
    ) -> Any:
        """Resample a 3D or 4D volume to a new size using specified interpolation.

        Supports 3D (D, H, W) and 4D (C, D, H, W) inputs. Matches PyTorch
        F.interpolate behavior.

        Args:
            data: Input numpy array of shape (D, H, W) or (C, D, H, W).
                  Supported dtypes: uint8, int16, float32.
            size: Target size (new_D, new_H, new_W) for spatial dimensions.
            mode: Interpolation mode: 'nearest', 'linear', or 'area'.
            parallel_threads: Number of threads for parallel execution (default 0 = library default)

        Returns:
            Resampled numpy array. Shape is (new_D, new_H, new_W) for 3D inputs
            and (C, new_D, new_H, new_W) for 4D inputs.
            - For nearest neighbor: output dtype matches input dtype
            - For linear/area: output dtype is always float32

        Raises:
            ValueError: If mode is not supported or data is not 3D/4D.
            ImportError: If Cython extension is not available.
            TypeError: If data is not a numpy array.
        """
        if not CYTHON_AVAILABLE:
            raise ImportError(
                "Cython extension not available. "
                "Build it with: python dev_cythonized.py build_ext --inplace"
            )

        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")

        if mode not in ("nearest", "linear", "area"):
            raise ValueError(
                f"Unsupported interpolation mode '{mode}'. "
                "CythonBackend supports 'nearest', 'linear', and 'area'."
            )

        if data.ndim not in (3, 4):
            raise ValueError(
                f"Data must be 3D or 4D (got {data.ndim}). "
                "Supported shapes: (D,H,W) or (C,D,H,W)."
            )

        # For nearest neighbor with uint8/int16, preserve dtype
        # For other modes, convert to float32
        if mode != "nearest" or data.dtype not in (np.uint8, np.int16):
            data = data.astype(np.float32, copy=False)

        # Ensure contiguous C-order array
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        
        # Call unified resample function (handles both 3D and 4D)
        return resampling_cython.resample(data, size, mode, parallel_threads)

    @staticmethod
    def grid_sample(
        input_data: Any,
        grid: Any,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        parallel_threads: int = 0,
    ) -> Any:
        """Grid sample operation matching PyTorch's F.grid_sample with align_corners=False.

        Given an input and a flow-field grid, computes the output using input values
        and pixel locations from grid.

        Args:
            input_data: Input numpy array of shape (N, C, H, W) for 2D or (N, C, D, H, W) for 3D.
            grid: Flow-field numpy array of shape (N, H_out, W_out, 2) for 2D or 
                  (N, D_out, H_out, W_out, 3) for 3D. Values should be in the range [-1, 1],
                  where -1 corresponds to the left/top/front edge and 1 corresponds to the 
                  right/bottom/back edge (align_corners=False behavior).
            mode: Interpolation mode - 'bilinear' (trilinear for 3D) or 'nearest'. Default: 'bilinear'.
            padding_mode: Padding mode for out-of-bound grid values - 'zeros', 'border', or 'reflection'.
                          Default: 'zeros'.
            parallel_threads: Number of threads for parallel execution (default 0 = library default).

        Returns:
            Output numpy array of shape (N, C, H_out, W_out) for 2D or (N, C, D_out, H_out, W_out) for 3D.

        Raises:
            ValueError: If mode or padding_mode is not supported, or if input/grid shapes are invalid.
            ImportError: If Cython extension is not available.
            TypeError: If input_data or grid is not a numpy array.

        Note:
            This implementation uses align_corners=False (PyTorch default).
            Grid coordinate x=-1 maps to the left edge of the leftmost pixel,
            x=1 maps to the right edge of the rightmost pixel.
        """
        if not CYTHON_AVAILABLE:
            raise ImportError(
                "Cython extension not available. "
                "Build it with: python dev_cythonize.py build_ext --inplace"
            )

        if not isinstance(input_data, np.ndarray):
            raise TypeError("input_data must be a numpy.ndarray")

        if not isinstance(grid, np.ndarray):
            raise TypeError("grid must be a numpy.ndarray")

        if mode not in ("bilinear", "nearest"):
            raise ValueError(
                f"Unsupported interpolation mode '{mode}'. "
                "Supported: 'bilinear', 'nearest'."
            )

        if padding_mode not in ("zeros", "border", "reflection"):
            raise ValueError(
                f"Unsupported padding mode '{padding_mode}'. "
                "Supported: 'zeros', 'border', 'reflection'."
            )

        if input_data.ndim not in (4, 5):
            raise ValueError(
                f"Input must be 4D or 5D (got {input_data.ndim}). "
                "Supported shapes: (N,C,H,W) or (N,C,D,H,W)."
            )

        # Ensure contiguous arrays
        if not input_data.flags['C_CONTIGUOUS']:
            input_data = np.ascontiguousarray(input_data)
        if not grid.flags['C_CONTIGUOUS']:
            grid = np.ascontiguousarray(grid)

        return resampling_cython.grid_sample(
            input_data, grid, mode, padding_mode, parallel_threads
        )
