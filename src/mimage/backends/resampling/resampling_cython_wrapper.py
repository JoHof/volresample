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
    ) -> Any:
        """Resample a 3D volume to a new size using specified interpolation.

        Supports 3D (D, H, W) and 4D (C, D, H, W) inputs. Matches PyTorch
        F.interpolate behavior.

        Behavior:
        - Input is cast to float32 before interpolation.
        - Supports 'nearest', 'linear' (trilinear), and 'area' modes.
        - Output is always float32.

        Args:
            data: Input numpy array of shape (D, H, W) or (C, D, H, W).
            size: Target size (new_D, new_H, new_W) for spatial dimensions.
            mode: Interpolation mode: 'nearest', 'linear', or 'area'.

        Returns:
            Resampled numpy array. Shape is (new_D, new_H, new_W) for 3D inputs
            and (C, new_D, new_H, new_W) for 4D inputs. Output dtype is float32.

        Raises:
            ValueError: If mode is not supported or data is not 3D/4D.
            ImportError: If Cython extension is not available.
            TypeError: If data is not a numpy array.
        """
        if not CYTHON_AVAILABLE:
            raise ImportError(
                "Cython extension not available. "
                "Build it with: python setup_cython.py build_ext --inplace"
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

        # Convert to float32 for interpolation
        data_f32 = data.astype(np.float32, copy=False)

        # Call appropriate Cython function
        if data.ndim == 3:
            # Ensure contiguous C-order array
            if not data_f32.flags['C_CONTIGUOUS']:
                data_f32 = np.ascontiguousarray(data_f32)
            return resampling_cython.resample_3d(data_f32, size, mode)
        else:  # ndim == 4
            # Ensure contiguous C-order array
            if not data_f32.flags['C_CONTIGUOUS']:
                data_f32 = np.ascontiguousarray(data_f32)
            return resampling_cython.resample_4d(data_f32, size, mode)
