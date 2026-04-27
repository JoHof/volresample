# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""Cython implementation for fast 3D volume resampling."""

import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.math cimport floor, ceil
from libc.stdlib cimport malloc, free

# Import global thread configuration
from volresample._config import get_num_threads

cdef extern from "omp.h":
    int omp_set_num_threads(int)
    int omp_get_thread_num() noexcept nogil
    int omp_get_max_threads() noexcept nogil

cnp.import_array()

# Include the implementation files directly
include "cython_src/utils.pyx"
include "cython_src/nearest.pyx"
include "cython_src/linear.pyx"
include "cython_src/area.pyx"
include "cython_src/cubic.pyx"
include "cython_src/grid_sample.pyx"


cdef inline void _apply_thread_settings() noexcept:
    """Apply global thread settings to OpenMP."""
    cdef int num_threads = get_num_threads()
    omp_set_num_threads(num_threads)


# Dispatch wrappers for dtype support in nearest neighbor
cdef object _resample_nearest_dispatch(
    object data,
    tuple size,
    bint align_corners,
):
    """Dispatch nearest neighbor resampling based on input dtype."""
    cdef int in_d = data.shape[0]
    cdef int in_h = data.shape[1]
    cdef int in_w = data.shape[2]
    cdef int out_d = size[0]
    cdef int out_h = size[1]
    cdef int out_w = size[2]

    cdef float scale_d = <float>in_d / <float>out_d
    cdef float scale_h = <float>in_h / <float>out_h
    cdef float scale_w = <float>in_w / <float>out_w
    
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] data_u8
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] output_u8
    cdef uint8_t* data_ptr_u8
    cdef uint8_t* output_ptr_u8
    
    cdef cnp.ndarray[cnp.int16_t, ndim=3] data_i16
    cdef cnp.ndarray[cnp.int16_t, ndim=3] output_i16
    cdef int16_t* data_ptr_i16
    cdef int16_t* output_ptr_i16
    
    cdef cnp.ndarray[cnp.float32_t, ndim=3] data_f32
    cdef cnp.ndarray[cnp.float32_t, ndim=3] output_f32
    cdef float* data_ptr_f32
    cdef float* output_ptr_f32

    # Dispatch based on dtype - ensure C-contiguous memory layout
    if data.dtype == np.uint8:
        data_u8 = np.ascontiguousarray(data, dtype=np.uint8)
        output_u8 = np.empty((out_d, out_h, out_w), dtype=np.uint8)
        data_ptr_u8 = <uint8_t*>cnp.PyArray_DATA(data_u8)
        output_ptr_u8 = <uint8_t*>cnp.PyArray_DATA(output_u8)
        
        # Release GIL for parallel execution
        with nogil:
            _resample_nearest(data_ptr_u8, output_ptr_u8, in_d, in_h, in_w, out_d, out_h, out_w, scale_d, scale_h, scale_w, align_corners)
        return output_u8
    
    elif data.dtype == np.int16:
        data_i16 = np.ascontiguousarray(data, dtype=np.int16)
        output_i16 = np.empty((out_d, out_h, out_w), dtype=np.int16)
        data_ptr_i16 = <int16_t*>cnp.PyArray_DATA(data_i16)
        output_ptr_i16 = <int16_t*>cnp.PyArray_DATA(output_i16)
        
        # Release GIL for parallel execution
        with nogil:
            _resample_nearest(data_ptr_i16, output_ptr_i16, in_d, in_h, in_w, out_d, out_h, out_w, scale_d, scale_h, scale_w, align_corners)
        return output_i16
    
    else:  # float32
        data_f32 = np.ascontiguousarray(data, dtype=np.float32)
        output_f32 = np.empty((out_d, out_h, out_w), dtype=np.float32)
        data_ptr_f32 = <float*>cnp.PyArray_DATA(data_f32)
        output_ptr_f32 = <float*>cnp.PyArray_DATA(output_f32)
        
        # Release GIL for parallel execution
        with nogil:
            _resample_nearest(data_ptr_f32, output_ptr_f32, in_d, in_h, in_w, out_d, out_h, out_w, scale_d, scale_h, scale_w, align_corners)
        return output_f32

def resample(
    data,
    tuple size,
    str mode="linear",
    bint align_corners=False
):
    """Resample 3D, 4D or 5D volume using the specified interpolation mode.

    Args:
        data: Input array, shape (D, H, W), (C, D, H, W), or (N, C, D, H, W).
            Supports uint8, int16, float32.
        size: Output size (D, H, W).
        mode: Interpolation mode - 'nearest', 'linear', 'area', or 'cubic'.
        align_corners: If True, corner voxels of input and output are aligned,
            preserving values at the corners. Only supported for 'nearest',
            'linear', and 'cubic' modes.
            - For 'nearest': maps output corner indices to input corner indices;
            - For 'linear': matches PyTorch trilinear interpolate with
              align_corners=True.
            - For 'cubic': matches scipy.ndimage.zoom(order=3, mode='reflect',
              grid_mode=False).
            - With align_corners=False, 'cubic' matches scipy.ndimage.zoom(
              order=3, mode='reflect', grid_mode=True).

    Returns:
        Resampled array with the same number of dimensions as the input.

    Note:
        Thread count is controlled globally via volresample.set_num_threads().
        Default is min(cpu_count, 4).

    Examples:
        >>> import numpy as np
        >>> import volresample
        >>> volresample.set_num_threads(4)
        >>> data = np.random.rand(64, 64, 64).astype(np.float32)
        >>> resampled = volresample.resample(data, (32, 32, 32), mode='linear')
        >>> resampled.shape
        (32, 32, 32)
    """
    cdef int ndim
    cdef int n_batch, n_channels
    cdef cnp.ndarray data_np
    cdef cnp.ndarray output
    cdef cnp.ndarray channel_output
    cdef int b, c
    cdef list batch_outputs, channel_outputs
    # Multi-channel linear fast path variables
    cdef cnp.ndarray data_f32_mc
    cdef cnp.ndarray output_mc
    cdef float* mc_data_ptr
    cdef float* mc_out_ptr
    cdef int mc_in_d, mc_in_h, mc_in_w
    cdef int mc_out_d, mc_out_h, mc_out_w
    cdef int mc_total_ch
    cdef float mc_scale_d, mc_scale_h, mc_scale_w
    
    # Validate size tuple
    if len(size) != 3:
        raise ValueError(
            f"size must be a 3-tuple (D, H, W), got {len(size)} elements"
        )
    if size[0] <= 0 or size[1] <= 0 or size[2] <= 0:
        raise ValueError(
            f"All output dimensions must be positive, got size={size}"
        )

    # Validate align_corners
    if align_corners and mode not in ("nearest", "linear", "cubic"):
        raise ValueError(
            f"align_corners=True is only supported for 'nearest', 'linear', and 'cubic' modes, got '{mode}'"
        )
    
    # Apply global thread settings
    _apply_thread_settings()
    
    # Ensure numpy array with C-contiguous memory layout
    data_np = np.ascontiguousarray(data)
    ndim = data_np.ndim
    
    if ndim not in (3, 4, 5):
        raise ValueError(f"Data must be 3D, 4D, or 5D, got {ndim}D")
    
    # Handle 5D: iterate over batch and channels
    if ndim == 5:
        n_batch = data_np.shape[0]
        n_channels = data_np.shape[1]

        # Fast multi-channel path for linear: process all N*C channels at once
        if mode == "linear":
            mc_total_ch = n_batch * n_channels
            data_f32_mc = np.ascontiguousarray(data_np, dtype=np.float32)
            mc_in_d = data_f32_mc.shape[2]
            mc_in_h = data_f32_mc.shape[3]
            mc_in_w = data_f32_mc.shape[4]
            mc_out_d = size[0]
            mc_out_h = size[1]
            mc_out_w = size[2]
            output_mc = np.empty((n_batch, n_channels, mc_out_d, mc_out_h, mc_out_w), dtype=np.float32)
            mc_data_ptr = <float*>cnp.PyArray_DATA(data_f32_mc)
            mc_out_ptr = <float*>cnp.PyArray_DATA(output_mc)
            mc_scale_d = <float>mc_in_d / <float>mc_out_d
            mc_scale_h = <float>mc_in_h / <float>mc_out_h
            mc_scale_w = <float>mc_in_w / <float>mc_out_w
            with nogil:
                _resample_linear_multi(
                    mc_data_ptr, mc_out_ptr, mc_total_ch,
                    mc_in_d, mc_in_h, mc_in_w,
                    mc_out_d, mc_out_h, mc_out_w,
                    mc_scale_d, mc_scale_h, mc_scale_w, align_corners
                )
            return output_mc

        batch_outputs = []
        
        for b in range(n_batch):
            channel_outputs = []
            for c in range(n_channels):
                channel_output = _resample_channel(data_np[b, c], size, mode, align_corners)
                channel_outputs.append(channel_output)
            batch_outputs.append(np.stack(channel_outputs, axis=0))
        
        return np.stack(batch_outputs, axis=0)
    
    # Handle 4D: iterate over channels
    elif ndim == 4:
        n_channels = data_np.shape[0]

        # Fast multi-channel path for linear
        if mode == "linear":
            data_f32_mc = np.ascontiguousarray(data_np, dtype=np.float32)
            mc_in_d = data_f32_mc.shape[1]
            mc_in_h = data_f32_mc.shape[2]
            mc_in_w = data_f32_mc.shape[3]
            mc_out_d = size[0]
            mc_out_h = size[1]
            mc_out_w = size[2]
            output_mc = np.empty((n_channels, mc_out_d, mc_out_h, mc_out_w), dtype=np.float32)
            mc_data_ptr = <float*>cnp.PyArray_DATA(data_f32_mc)
            mc_out_ptr = <float*>cnp.PyArray_DATA(output_mc)
            mc_scale_d = <float>mc_in_d / <float>mc_out_d
            mc_scale_h = <float>mc_in_h / <float>mc_out_h
            mc_scale_w = <float>mc_in_w / <float>mc_out_w
            with nogil:
                _resample_linear_multi(
                    mc_data_ptr, mc_out_ptr, n_channels,
                    mc_in_d, mc_in_h, mc_in_w,
                    mc_out_d, mc_out_h, mc_out_w,
                    mc_scale_d, mc_scale_h, mc_scale_w, align_corners
                )
            return output_mc

        channel_outputs = []
        
        for c in range(n_channels):
            channel_output = _resample_channel(data_np[c], size, mode, align_corners)
            channel_outputs.append(channel_output)
        
        return np.stack(channel_outputs, axis=0)
    
    # 3D case
    return _resample_channel(data_np, size, mode, align_corners)

cdef object _resample_channel(
    cnp.ndarray data,
    tuple size,
    str mode,
    bint align_corners
):
    """Resample a single 3D volume."""
    cdef int in_d = data.shape[0]
    cdef int in_h = data.shape[1]
    cdef int in_w = data.shape[2]
    cdef int out_d = size[0]
    cdef int out_h = size[1]
    cdef int out_w = size[2]
    
    cdef float scale_d = <float>in_d / <float>out_d
    cdef float scale_h = <float>in_h / <float>out_h
    cdef float scale_w = <float>in_w / <float>out_w
    
    cdef cnp.ndarray[cnp.float32_t, ndim=3] data_f32
    cdef cnp.ndarray[cnp.float32_t, ndim=3] output
    cdef float* data_ptr
    cdef float* output_ptr
    cdef int cubic_nt
    
    # Mode dispatch
    if mode == "nearest":
        return _resample_nearest_dispatch(data, size, align_corners)
    
    elif mode == "linear":
        # Linear always uses float32, ensure C-contiguous
        data_f32 = np.ascontiguousarray(data, dtype=np.float32)
        output = np.empty((out_d, out_h, out_w), dtype=np.float32)
        data_ptr = <float*>cnp.PyArray_DATA(data_f32)
        output_ptr = <float*>cnp.PyArray_DATA(output)
        
        # Release GIL for parallel execution
        with nogil:
            _resample_linear(data_ptr, output_ptr, in_d, in_h, in_w, out_d, out_h, out_w, scale_d, scale_h, scale_w, align_corners)
        return output
    
    elif mode == "area":
        # Area mode: handles both downsampling (averaging) and upsampling (replication)
        # per-dimension independently, matching PyTorch's F.interpolate behavior
        data_f32 = np.ascontiguousarray(data, dtype=np.float32)
        output = np.empty((out_d, out_h, out_w), dtype=np.float32)
        data_ptr = <float*>cnp.PyArray_DATA(data_f32)
        output_ptr = <float*>cnp.PyArray_DATA(output)
        
        # Release GIL for parallel execution
        with nogil:
            _resample_area(data_ptr, output_ptr, in_d, in_h, in_w, out_d, out_h, out_w, scale_d, scale_h, scale_w)
        return output
    
    elif mode == "cubic":
        # Cubic B-spline matching scipy.ndimage.zoom(order=3, mode='reflect'):
        # - grid_mode=True when align_corners=False
        # - grid_mode=False when align_corners=True
        data_f32 = np.ascontiguousarray(data, dtype=np.float32)
        
        # Identity fast-path: when output size == input size, just copy
        # (matches scipy behavior: prefilter + eval at integer coords = identity)
        if in_d == out_d and in_h == out_h and in_w == out_w:
            return data_f32.copy()
        
        output = np.empty((out_d, out_h, out_w), dtype=np.float32)
        data_ptr = <float*>cnp.PyArray_DATA(data_f32)
        output_ptr = <float*>cnp.PyArray_DATA(output)
        cubic_nt = <int>get_num_threads()
        
        # Release GIL for parallel execution
        with nogil:
            _resample_cubic(data_ptr, output_ptr, in_d, in_h, in_w, out_d, out_h, out_w, scale_d, scale_h, scale_w, cubic_nt, align_corners)
        return output
    
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'nearest', 'linear', 'area', or 'cubic'.")


def grid_sample(
    input,
    grid,
    str mode="linear",
    str padding_mode="zeros",
    fill_value=0,
):
    """Sample input using a sampling grid (similar to PyTorch's grid_sample).
    
    Args:
        input: Input array, shape (N, C, D, H, W).
            Supports uint8, int16, float32 for nearest mode; float32 for linear.
        grid: Sampling grid, shape (N, D_out, H_out, W_out, 3).
              Values in range [-1, 1] where -1 is the start and 1 is the end.
        mode: Interpolation mode - 'linear' or 'nearest'.
        padding_mode: Padding mode for out-of-bounds values - 'zeros', 'border',
            'reflection', or 'constant'.
        fill_value: Fill value for out-of-bounds samples when
            padding_mode='constant'. Defaults to 0. For integer dtypes in
            nearest mode, the value is clamped to the valid dtype range.
        
    Returns:
        Sampled array of shape (N, C, D_out, H_out, W_out).
        For nearest mode, preserves input dtype.
        For linear mode, returns float32.
        
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
    # Grid is always float32
    cdef cnp.ndarray grid_np = np.ascontiguousarray(grid, dtype=np.float32)
    # Input: preserve dtype for nearest mode, convert to float32 for linear
    cdef cnp.ndarray input_np
    if mode == "nearest":
        input_np = np.ascontiguousarray(input)
    else:
        input_np = np.ascontiguousarray(input, dtype=np.float32)
    
    if input_np.ndim != 5:
        raise ValueError(f"Input must be 5D (N, C, D, H, W), got {input_np.ndim}D")
    if grid_np.ndim != 5:
        raise ValueError(f"Grid must be 5D (N, D_out, H_out, W_out, 3), got {grid_np.ndim}D")
    if grid_np.shape[4] != 3:
        raise ValueError(f"Grid last dimension must be 3, got {grid_np.shape[4]}")
    if grid_np.shape[0] != input_np.shape[0]:
        raise ValueError(
            f"Batch size of input ({input_np.shape[0]}) and grid ({grid_np.shape[0]}) must match"
        )
    
    cdef int N = input_np.shape[0]
    cdef int C = input_np.shape[1]
    cdef int in_d = input_np.shape[2]
    cdef int in_h = input_np.shape[3]
    cdef int in_w = input_np.shape[4]
    cdef int out_d = grid_np.shape[1]
    cdef int out_h = grid_np.shape[2]
    cdef int out_w = grid_np.shape[3]
    
    cdef cnp.ndarray output
    cdef float* input_ptr
    cdef float* grid_ptr
    cdef float* output_ptr
    cdef uint8_t* input_ptr_u8
    cdef uint8_t* output_ptr_u8
    cdef int16_t* input_ptr_i16
    cdef int16_t* output_ptr_i16
    cdef float fill_f32
    cdef uint8_t fill_u8
    cdef int16_t fill_i16
    cdef int fill_int
    
    # Apply global thread settings
    _apply_thread_settings()
    
    # Accept both 'linear' and 'bilinear' for compatibility
    if mode not in ("nearest", "linear", "bilinear"):
        raise ValueError(f"Unsupported mode: {mode}. Use 'nearest' or 'linear'.")
    
    if padding_mode not in ("zeros", "border", "reflection", "constant"):
        raise ValueError(f"Unsupported padding_mode: {padding_mode}")
    
    # Treat "zeros" as "constant" with fill_value=0
    if padding_mode == "zeros":
        padding_mode = "constant"
        fill_value = 0
    
    # Grid pointer is always float32
    grid_ptr = <float*>cnp.PyArray_DATA(grid_np)
    
    # Dispatch based on mode and dtype
    if mode == "nearest":
        if input_np.dtype == np.uint8:
            output = np.empty((N, C, out_d, out_h, out_w), dtype=np.uint8)
            input_ptr_u8 = <uint8_t*>cnp.PyArray_DATA(input_np)
            output_ptr_u8 = <uint8_t*>cnp.PyArray_DATA(output)
            if padding_mode == "constant":
                fill_int = max(0, min(255, int(round(fill_value))))
                fill_u8 = <uint8_t>fill_int
                with nogil:
                    _grid_sample_nearest_zeros(input_ptr_u8, grid_ptr, output_ptr_u8,
                                             N, C, in_d, in_h, in_w, out_d, out_h, out_w,
                                             fill_u8)
            elif padding_mode == "border":
                with nogil:
                    _grid_sample_nearest_border(input_ptr_u8, grid_ptr, output_ptr_u8,
                                              N, C, in_d, in_h, in_w, out_d, out_h, out_w)
            elif padding_mode == "reflection":
                with nogil:
                    _grid_sample_nearest_reflection(input_ptr_u8, grid_ptr, output_ptr_u8,
                                                  N, C, in_d, in_h, in_w, out_d, out_h, out_w)
        elif input_np.dtype == np.int16:
            output = np.empty((N, C, out_d, out_h, out_w), dtype=np.int16)
            input_ptr_i16 = <int16_t*>cnp.PyArray_DATA(input_np)
            output_ptr_i16 = <int16_t*>cnp.PyArray_DATA(output)
            if padding_mode == "constant":
                fill_int = max(-32768, min(32767, int(round(fill_value))))
                fill_i16 = <int16_t>fill_int
                with nogil:
                    _grid_sample_nearest_zeros(input_ptr_i16, grid_ptr, output_ptr_i16,
                                             N, C, in_d, in_h, in_w, out_d, out_h, out_w,
                                             fill_i16)
            elif padding_mode == "border":
                with nogil:
                    _grid_sample_nearest_border(input_ptr_i16, grid_ptr, output_ptr_i16,
                                              N, C, in_d, in_h, in_w, out_d, out_h, out_w)
            elif padding_mode == "reflection":
                with nogil:
                    _grid_sample_nearest_reflection(input_ptr_i16, grid_ptr, output_ptr_i16,
                                                  N, C, in_d, in_h, in_w, out_d, out_h, out_w)
        else:
            input_np = np.ascontiguousarray(input_np, dtype=np.float32)
            output = np.empty((N, C, out_d, out_h, out_w), dtype=np.float32)
            input_ptr = <float*>cnp.PyArray_DATA(input_np)
            output_ptr = <float*>cnp.PyArray_DATA(output)
            if padding_mode == "constant":
                fill_f32 = <float>fill_value
                with nogil:
                    _grid_sample_nearest_zeros(input_ptr, grid_ptr, output_ptr,
                                             N, C, in_d, in_h, in_w, out_d, out_h, out_w,
                                             fill_f32)
            elif padding_mode == "border":
                with nogil:
                    _grid_sample_nearest_border(input_ptr, grid_ptr, output_ptr,
                                              N, C, in_d, in_h, in_w, out_d, out_h, out_w)
            elif padding_mode == "reflection":
                with nogil:
                    _grid_sample_nearest_reflection(input_ptr, grid_ptr, output_ptr,
                                                  N, C, in_d, in_h, in_w, out_d, out_h, out_w)
    else:  # linear (or bilinear for compatibility)
        output = np.empty((N, C, out_d, out_h, out_w), dtype=np.float32)
        input_ptr = <float*>cnp.PyArray_DATA(input_np)
        output_ptr = <float*>cnp.PyArray_DATA(output)
        if padding_mode == "constant":
            fill_f32 = <float>fill_value
            with nogil:
                _grid_sample_bilinear_zeros(input_ptr, grid_ptr, output_ptr,
                                          N, C, in_d, in_h, in_w, out_d, out_h, out_w,
                                          fill_f32)
        elif padding_mode == "border":
            with nogil:
                _grid_sample_bilinear_border(input_ptr, grid_ptr, output_ptr,
                                           N, C, in_d, in_h, in_w, out_d, out_h, out_w)
        elif padding_mode == "reflection":
            with nogil:
                _grid_sample_bilinear_reflection(input_ptr, grid_ptr, output_ptr,
                                               N, C, in_d, in_h, in_w, out_d, out_h, out_w)
    
    return output
