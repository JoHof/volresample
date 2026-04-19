"""Type stubs for volresample._resample Cython module."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

def resample(
    data: NDArray[np.float32] | NDArray[np.uint8] | NDArray[np.int16],
    size: tuple[int, int, int],
    mode: Literal["nearest", "linear", "area", "cubic"] = "linear",
    align_corners: bool = False,
) -> NDArray[np.float32] | NDArray[np.uint8] | NDArray[np.int16]:
    """Resample a 3D, 4D, or 5D volume to a new size.

    Args:
        data: Input array of shape (D, H, W), (C, D, H, W), or (N, C, D, H, W).
            Supported dtypes: uint8, int16, float32.
        size: Target spatial size (D_out, H_out, W_out).
        mode: Interpolation mode:
            - 'nearest': Nearest neighbor (works with all dtypes).
              Corresponds to PyTorch's 'nearest-exact'.
            - 'linear': Trilinear interpolation (float32 only).
              Corresponds to PyTorch's 'trilinear'.
            - 'area': Area-based averaging (float32 only, suited for downsampling).
            - 'cubic': Cubic B-spline interpolation (float32 only).
              Matches scipy.ndimage.zoom(order=3, mode='reflect').
        align_corners: If True, corner voxels of input and output are aligned,
            preserving values at the corners. Only supported for 'linear' and
            'cubic' modes.
            - For 'linear': matches PyTorch trilinear with align_corners=True.
            - For 'cubic': matches scipy.ndimage.zoom(order=3, mode='reflect',
              grid_mode=False).
            - For 'cubic' with align_corners=False: matches scipy.ndimage.zoom(
              order=3, mode='reflect', grid_mode=True).

    Returns:
        Resampled array with the same number of dimensions as the input.
        - For nearest mode: preserves input dtype.
        - For linear, area, and cubic modes: returns float32.

    Note:
        Thread count is controlled globally via volresample.set_num_threads().
        Default is min(cpu_count, 4).
    """
    ...

def grid_sample(
    input: NDArray[np.float32] | NDArray[np.uint8] | NDArray[np.int16],
    grid: NDArray[np.float32],
    mode: Literal["linear", "nearest"] = "linear",
    padding_mode: Literal["zeros", "border", "reflection"] = "zeros",
) -> NDArray[np.float32] | NDArray[np.uint8] | NDArray[np.int16]:
    """Sample input at arbitrary locations specified by a grid.

    Similar to PyTorch's F.grid_sample for 3D volumes.

    Args:
        input: Input array of shape (N, C, D, H, W).
            Supported dtypes: uint8, int16, float32.
        grid: Sampling grid of shape (N, D_out, H_out, W_out, 3).
            Values in range [-1, 1] where -1 maps to the first voxel
            and 1 maps to the last voxel.
        mode: Interpolation mode:
            - 'linear': Trilinear interpolation (float32 only)
            - 'nearest': Nearest neighbor (works with all dtypes)
        padding_mode: How to handle out-of-bounds grid values:
            - 'zeros': Use 0 for out-of-bounds samples
            - 'border': Use border values for out-of-bounds samples
            - 'reflection': Reflect coordinates at boundaries

    Returns:
        Sampled array of shape (N, C, D_out, H_out, W_out).
        - For nearest mode: preserves input dtype.
        - For linear mode: returns float32.

    Note:
        The behavior matches PyTorch's grid_sample with align_corners=False.
        Thread count is controlled globally via volresample.set_num_threads().
        Default is min(cpu_count, 4).

    Examples:
        >>> import numpy as np
        >>> import volresample
        >>> input = np.random.rand(1, 2, 32, 32, 32).astype(np.float32)
        >>> grid = np.random.uniform(-1, 1, (1, 24, 24, 24, 3)).astype(np.float32)
        >>> output = volresample.grid_sample(input, grid, mode='linear')
        >>> output.shape
        (1, 2, 24, 24, 24)
    """
    ...
