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

cnp.import_array()

# Include the implementation files directly
include "cython_src/utils.pyx"
include "cython_src/nearest.pyx"
include "cython_src/linear.pyx"
include "cython_src/area.pyx"
include "cython_src/grid_sample.pyx"


cdef inline void _apply_thread_settings() noexcept:
    """Apply global thread settings to OpenMP."""
    cdef int num_threads = get_num_threads()
    omp_set_num_threads(num_threads)


# Dispatch wrappers for dtype support in nearest neighbor
cdef object _resample_nearest_dispatch(
    object data,
    tuple size,
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
            _resample_nearest(data_ptr_u8, output_ptr_u8, in_d, in_h, in_w, out_d, out_h, out_w, scale_d, scale_h, scale_w)
        return output_u8
    
    elif data.dtype == np.int16:
        data_i16 = np.ascontiguousarray(data, dtype=np.int16)
        output_i16 = np.empty((out_d, out_h, out_w), dtype=np.int16)
        data_ptr_i16 = <int16_t*>cnp.PyArray_DATA(data_i16)
        output_ptr_i16 = <int16_t*>cnp.PyArray_DATA(output_i16)
        
        # Release GIL for parallel execution
        with nogil:
            _resample_nearest(data_ptr_i16, output_ptr_i16, in_d, in_h, in_w, out_d, out_h, out_w, scale_d, scale_h, scale_w)
        return output_i16
    
    else:  # float32
        data_f32 = np.ascontiguousarray(data, dtype=np.float32)
        output_f32 = np.empty((out_d, out_h, out_w), dtype=np.float32)
        data_ptr_f32 = <float*>cnp.PyArray_DATA(data_f32)
        output_ptr_f32 = <float*>cnp.PyArray_DATA(output_f32)
        
        # Release GIL for parallel execution
        with nogil:
            _resample_nearest(data_ptr_f32, output_ptr_f32, in_d, in_h, in_w, out_d, out_h, out_w, scale_d, scale_h, scale_w)
        return output_f32

def resample(
    data,
    tuple size,
    str mode="linear"
):
    """Resample 3D or 4D volume using specified interpolation mode.
    
    Args:
        data: Input array, shape (D, H, W) or (C, D, H, W). Supports uint8, int16, float32.
        size: Output size (D, H, W).
        mode: Interpolation mode - 'nearest', 'linear', 'area'.
        
    Returns:
        Resampled array with same number of dimensions as input.
        
    Note:
        Thread count is controlled globally via volresample.set_num_threads().
        Default is min(cpu_count, 4).
        
    Examples:
        >>> import numpy as np
        >>> import volresample
        >>> volresample.set_num_threads(4)  # Optional: set thread count
        >>> data = np.random.rand(64, 64, 64).astype(np.float32)
        >>> resampled = volresample.resample(data, (32, 32, 32), mode='linear')
        >>> resampled.shape
        (32, 32, 32)
    """
    cdef bint is_3d = data.ndim == 3
    cdef int n_channels
    cdef cnp.ndarray data_np
    cdef cnp.ndarray output
    cdef cnp.ndarray channel_output
    cdef int c
    
    # Apply global thread settings
    _apply_thread_settings()
    
    # Ensure numpy array with C-contiguous memory layout
    data_np = np.ascontiguousarray(data)
    
    if data_np.ndim not in (3, 4):
        raise ValueError(f"Data must be 3D or 4D, got {data_np.ndim}D")
    
    # Handle 4D: iterate over channels
    if not is_3d:
        n_channels = data_np.shape[0]
        channel_outputs = []
        
        for c in range(n_channels):
            channel_output = _resample_channel(data_np[c], size, mode)
            channel_outputs.append(channel_output)
        
        return np.stack(channel_outputs, axis=0)
    
    # 3D case
    return _resample_channel(data_np, size, mode)

cdef object _resample_channel(
    cnp.ndarray data,
    tuple size,
    str mode
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
    
    # Mode dispatch
    if mode == "nearest":
        return _resample_nearest_dispatch(data, size)
    
    elif mode == "linear":
        # Linear always uses float32, ensure C-contiguous
        data_f32 = np.ascontiguousarray(data, dtype=np.float32)
        output = np.empty((out_d, out_h, out_w), dtype=np.float32)
        data_ptr = <float*>cnp.PyArray_DATA(data_f32)
        output_ptr = <float*>cnp.PyArray_DATA(output)
        
        # Release GIL for parallel execution
        with nogil:
            _resample_linear(data_ptr, output_ptr, in_d, in_h, in_w, out_d, out_h, out_w, scale_d, scale_h, scale_w)
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
    
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'nearest', 'linear', or 'area'.")


def grid_sample(
    input,
    grid,
    str mode="linear",
    str padding_mode="zeros",
):
    """Sample input using a sampling grid (similar to PyTorch's grid_sample).
    
    Args:
        input: Input array, shape (N, C, D, H, W).
        grid: Sampling grid, shape (N, D_out, H_out, W_out, 3).
              Values in range [-1, 1] where -1 is the start and 1 is the end.
        mode: Interpolation mode - 'linear' or 'nearest'.
        padding_mode: Padding mode for out-of-bounds values - 'zeros', 'border', 'reflection'.
        
    Returns:
        Sampled array of shape (N, C, D_out, H_out, W_out).
        
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
    # Ensure C-contiguous memory layout for both input and grid
    cdef cnp.ndarray input_np = np.ascontiguousarray(input, dtype=np.float32)
    cdef cnp.ndarray grid_np = np.ascontiguousarray(grid, dtype=np.float32)
    
    if input_np.ndim != 5:
        raise ValueError(f"Input must be 5D (N, C, D, H, W), got {input_np.ndim}D")
    if grid_np.ndim != 5:
        raise ValueError(f"Grid must be 5D (N, D_out, H_out, W_out, 3), got {grid_np.ndim}D")
    if grid_np.shape[4] != 3:
        raise ValueError(f"Grid last dimension must be 3, got {grid_np.shape[4]}")
    
    cdef int N = input_np.shape[0]
    cdef int C = input_np.shape[1]
    cdef int in_d = input_np.shape[2]
    cdef int in_h = input_np.shape[3]
    cdef int in_w = input_np.shape[4]
    cdef int out_d = grid_np.shape[1]
    cdef int out_h = grid_np.shape[2]
    cdef int out_w = grid_np.shape[3]
    
    cdef cnp.ndarray[cnp.float32_t, ndim=5] output = np.empty((N, C, out_d, out_h, out_w), dtype=np.float32)
    cdef float* input_ptr
    cdef float* grid_ptr
    cdef float* output_ptr
    
    # Apply global thread settings
    _apply_thread_settings()
    
    # Determine which grid_sample function to call
    # Accept both 'linear' and 'bilinear' for compatibility
    cdef int padding_id = 0  # 0=zeros, 1=border, 2=reflection
    
    if mode not in ("nearest", "linear", "bilinear"):
        raise ValueError(f"Unsupported mode: {mode}. Use 'nearest' or 'linear'.")
    
    if padding_mode == "zeros":
        padding_id = 0
    elif padding_mode == "border":
        padding_id = 1
    elif padding_mode == "reflection":
        padding_id = 2
    else:
        raise ValueError(f"Unsupported padding_mode: {padding_mode}")
    
    # Get pointers to the data
    input_ptr = <float*>cnp.PyArray_DATA(input_np)
    grid_ptr = <float*>cnp.PyArray_DATA(grid_np)
    output_ptr = <float*>cnp.PyArray_DATA(output)
    
    # Call appropriate grid_sample function
    if mode == "nearest":
        if padding_mode == "zeros":
            with nogil:
                _grid_sample_nearest_zeros(input_ptr, grid_ptr, output_ptr,
                                         N, C, in_d, in_h, in_w, out_d, out_h, out_w)
        elif padding_mode == "border":
            with nogil:
                _grid_sample_nearest_border(input_ptr, grid_ptr, output_ptr,
                                          N, C, in_d, in_h, in_w, out_d, out_h, out_w)
        elif padding_mode == "reflection":
            with nogil:
                _grid_sample_nearest_reflection(input_ptr, grid_ptr, output_ptr,
                                              N, C, in_d, in_h, in_w, out_d, out_h, out_w)
    else:  # linear (or bilinear for compatibility)
        if padding_mode == "zeros":
            with nogil:
                _grid_sample_bilinear_zeros(input_ptr, grid_ptr, output_ptr,
                                          N, C, in_d, in_h, in_w, out_d, out_h, out_w)
        elif padding_mode == "border":
            with nogil:
                _grid_sample_bilinear_border(input_ptr, grid_ptr, output_ptr,
                                           N, C, in_d, in_h, in_w, out_d, out_h, out_w)
        elif padding_mode == "reflection":
            with nogil:
                _grid_sample_bilinear_reflection(input_ptr, grid_ptr, output_ptr,
                                               N, C, in_d, in_h, in_w, out_d, out_h, out_w)
    
    return output
