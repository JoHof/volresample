# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""Cython backend for trilinear interpolation with AVX optimization."""

import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.math cimport floor, ceil
from libc.stdlib cimport malloc, free

cdef extern from "omp.h":
    int omp_set_num_threads(int)

cnp.import_array()

# Include the implementation files directly
include "cython_src/utils.pyx"
include "cython_src/nearest.pyx"
include "cython_src/linear.pyx"
include "cython_src/area.pyx"
include "cython_src/grid_sample.pyx"

# Dispatch wrappers for dtype support in nearest neighbor
cdef object _resample_nearest_dispatch(
    object data,
    tuple size,
    str mode="nearest",
    int parallel_threads=0
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

    if parallel_threads > 0:
        omp_set_num_threads(parallel_threads)

    # Dispatch based on dtype
    if data.dtype == np.uint8:
        data_u8 = np.asarray(data, dtype=np.uint8)
        output_u8 = np.empty((out_d, out_h, out_w), dtype=np.uint8)
        data_ptr_u8 = <uint8_t*>cnp.PyArray_DATA(data_u8)
        output_ptr_u8 = <uint8_t*>cnp.PyArray_DATA(output_u8)
        _resample_nearest(data_ptr_u8, output_ptr_u8, in_d, in_h, in_w, out_d, out_h, out_w, scale_d, scale_h, scale_w)
        return output_u8
    
    elif data.dtype == np.int16:
        data_i16 = np.asarray(data, dtype=np.int16)
        output_i16 = np.empty((out_d, out_h, out_w), dtype=np.int16)
        data_ptr_i16 = <int16_t*>cnp.PyArray_DATA(data_i16)
        output_ptr_i16 = <int16_t*>cnp.PyArray_DATA(output_i16)
        _resample_nearest(data_ptr_i16, output_ptr_i16, in_d, in_h, in_w, out_d, out_h, out_w, scale_d, scale_h, scale_w)
        return output_i16
    
    else:  # float32
        data_f32 = np.asarray(data, dtype=np.float32)
        output_f32 = np.empty((out_d, out_h, out_w), dtype=np.float32)
        data_ptr_f32 = <float*>cnp.PyArray_DATA(data_f32)
        output_ptr_f32 = <float*>cnp.PyArray_DATA(output_f32)
        _resample_nearest(data_ptr_f32, output_ptr_f32, in_d, in_h, in_w, out_d, out_h, out_w, scale_d, scale_h, scale_w)
        return output_f32

def resample(
    data,
    tuple size,
    str mode="linear",
    int parallel_threads=0
):
    """Resample 3D or 4D volume using specified interpolation mode.
    
    Args:
        data: Input array, shape (D, H, W) or (C, D, H, W). Supports uint8, int16, float32.
        size: Output size (D, H, W).
        mode: Interpolation mode - 'nearest', 'linear', 'area'.
        parallel_threads: Number of threads (0 = default).
    
    Returns:
        Resampled array with same dtype as input (for nearest) or float32 (for linear/area).
        Output shape matches input: (D, H, W) for 3D input, (C, D, H, W) for 4D input.
    """
    cdef int n_channels, c, in_d, in_h, in_w, out_d, out_h, out_w
    cdef float scale_d, scale_h, scale_w
    cdef Py_ssize_t in_channel_stride, out_channel_stride
    cdef cnp.ndarray[cnp.float32_t, ndim=4] data_f32
    cdef cnp.ndarray[cnp.float32_t, ndim=4] output
    cdef float* data_ptr
    cdef float* output_ptr
    cdef list output_channels
    cdef bint is_3d = data.ndim == 3
    
    # Handle 3D input by adding a channel dimension
    if is_3d:
        data = data[np.newaxis, ...]  # Add channel dimension: (D,H,W) -> (1,D,H,W)
    
    # For nearest neighbor with uint8/int16, process per-channel
    if mode == "nearest" and data.dtype in (np.uint8, np.int16):
        n_channels = data.shape[0]
        output_channels = []
        for c in range(n_channels):
            output_channels.append(_resample_nearest_dispatch(data[c], size, mode, parallel_threads))
        output_result = np.stack(output_channels, axis=0)
        # Remove channel dimension for 3D input
        if is_3d:
            return output_result[0]
        return output_result
    
    
    # Float32 path for all modes
    data_f32 = np.asarray(data, dtype=np.float32)
    
    n_channels = data_f32.shape[0]
    in_d = data_f32.shape[1]
    in_h = data_f32.shape[2]
    in_w = data_f32.shape[3]

    out_d = size[0]
    out_h = size[1]
    out_w = size[2]

    output = np.empty((n_channels, out_d, out_h, out_w), dtype=np.float32)

    scale_d = <float>in_d / <float>out_d
    scale_h = <float>in_h / <float>out_h
    scale_w = <float>in_w / <float>out_w

    in_channel_stride = in_d * in_h * in_w
    out_channel_stride = out_d * out_h * out_w
    data_ptr = <float*>cnp.PyArray_DATA(data_f32)
    output_ptr = <float*>cnp.PyArray_DATA(output)

    if parallel_threads > 0:
        omp_set_num_threads(parallel_threads)

    if mode == "nearest":
        for c in range(n_channels):
            _resample_nearest(
                data_ptr + c * in_channel_stride,
                output_ptr + c * out_channel_stride,
                in_d, in_h, in_w, out_d, out_h, out_w,
                scale_d, scale_h, scale_w
            )
    elif mode == "linear":
        for c in range(n_channels):
            _resample_linear(
                data_ptr + c * in_channel_stride,
                output_ptr + c * out_channel_stride,
                in_d, in_h, in_w, out_d, out_h, out_w,
                scale_d, scale_h, scale_w
            )
    elif mode == "area":
        if scale_d >= 1.0 and scale_h >= 1.0 and scale_w >= 1.0:
            for c in range(n_channels):
                _resample_area(
                    data_ptr + c * in_channel_stride,
                    output_ptr + c * out_channel_stride,
                    in_d, in_h, in_w, out_d, out_h, out_w,
                    scale_d, scale_h, scale_w
                )
        else:
            # Upsampling fallback to nearest
            for c in range(n_channels):
                _resample_nearest(
                    data_ptr + c * in_channel_stride,
                    output_ptr + c * out_channel_stride,
                    in_d, in_h, in_w, out_d, out_h, out_w,
                    scale_d, scale_h, scale_w
                )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Remove channel dimension for 3D input
    if is_3d:
        return output[0]
    return output


def grid_sample(
    input_data,
    grid,
    str mode="bilinear",
    str padding_mode="zeros",
    int parallel_threads=0
):
    """Grid sample operation matching PyTorch's F.grid_sample with align_corners=False.
    
    Given an input and a flow-field grid, computes the output using input values
    and pixel locations from grid.
    
    Args:
        input_data: Input array of shape (N, C, H, W) for 2D or (N, C, D, H, W) for 3D.
        grid: Flow-field of shape (N, H_out, W_out, 2) for 2D or (N, D_out, H_out, W_out, 3) for 3D.
              Values should be in the range [-1, 1], where -1 corresponds to the left/top/front
              edge and 1 corresponds to the right/bottom/back edge (align_corners=False behavior).
        mode: Interpolation mode - 'bilinear' (trilinear for 3D) or 'nearest'. Default: 'bilinear'.
        padding_mode: Padding mode for out-of-bound grid values - 'zeros', 'border', or 'reflection'.
                      Default: 'zeros'.
        parallel_threads: Number of threads for parallel execution (0 = library default).
    
    Returns:
        Output array of shape (N, C, H_out, W_out) for 2D or (N, C, D_out, H_out, W_out) for 3D.
    
    Note:
        This implementation uses align_corners=False (PyTorch default).
        Grid coordinate x=-1 maps to the left edge of the leftmost pixel,
        x=1 maps to the right edge of the rightmost pixel.
    """
    cdef cnp.ndarray[cnp.float32_t, ndim=4] input_4d
    cdef cnp.ndarray[cnp.float32_t, ndim=5] input_5d
    cdef cnp.ndarray[cnp.float32_t, ndim=4] grid_4d
    cdef cnp.ndarray[cnp.float32_t, ndim=5] grid_5d
    cdef cnp.ndarray[cnp.float32_t, ndim=4] output_4d
    cdef cnp.ndarray[cnp.float32_t, ndim=5] output_5d
    cdef float* input_ptr
    cdef float* grid_ptr
    cdef float* output_ptr
    cdef int N, C, D_in, H_in, W_in, D_out, H_out, W_out
    
    if mode not in ("bilinear", "nearest"):
        raise ValueError(f"Unsupported mode: {mode}. Supported: 'bilinear', 'nearest'")
    
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError(f"Unsupported padding_mode: {padding_mode}. Supported: 'zeros', 'border', 'reflection'")
    
    if parallel_threads > 0:
        omp_set_num_threads(parallel_threads)
    
    # Handle 4D input (2D spatial)
    if input_data.ndim == 4:
        if grid.ndim != 4 or grid.shape[3] != 2:
            raise ValueError(f"For 4D input, grid must be 4D with shape (N, H_out, W_out, 2), got {grid.shape}")
        
        input_4d = np.ascontiguousarray(input_data, dtype=np.float32)
        grid_4d = np.ascontiguousarray(grid, dtype=np.float32)
        
        N = input_4d.shape[0]
        C = input_4d.shape[1]
        H_in = input_4d.shape[2]
        W_in = input_4d.shape[3]
        H_out = grid_4d.shape[1]
        W_out = grid_4d.shape[2]
        
        if grid_4d.shape[0] != N:
            raise ValueError(f"Batch size mismatch: input has {N}, grid has {grid_4d.shape[0]}")
        
        output_4d = np.empty((N, C, H_out, W_out), dtype=np.float32)
        
        input_ptr = <float*>cnp.PyArray_DATA(input_4d)
        grid_ptr = <float*>cnp.PyArray_DATA(grid_4d)
        output_ptr = <float*>cnp.PyArray_DATA(output_4d)
        
        if mode == "bilinear":
            if padding_mode == "zeros":
                _grid_sample_2d_bilinear_zeros(input_ptr, grid_ptr, output_ptr, N, C, H_in, W_in, H_out, W_out)
            elif padding_mode == "border":
                _grid_sample_2d_bilinear_border(input_ptr, grid_ptr, output_ptr, N, C, H_in, W_in, H_out, W_out)
            else:  # reflection
                _grid_sample_2d_bilinear_reflection(input_ptr, grid_ptr, output_ptr, N, C, H_in, W_in, H_out, W_out)
        else:  # nearest
            if padding_mode == "zeros":
                _grid_sample_2d_nearest_zeros(input_ptr, grid_ptr, output_ptr, N, C, H_in, W_in, H_out, W_out)
            elif padding_mode == "border":
                _grid_sample_2d_nearest_border(input_ptr, grid_ptr, output_ptr, N, C, H_in, W_in, H_out, W_out)
            else:  # reflection
                _grid_sample_2d_nearest_reflection(input_ptr, grid_ptr, output_ptr, N, C, H_in, W_in, H_out, W_out)
        
        return output_4d
    
    # Handle 5D input (3D spatial)
    elif input_data.ndim == 5:
        if grid.ndim != 5 or grid.shape[4] != 3:
            raise ValueError(f"For 5D input, grid must be 5D with shape (N, D_out, H_out, W_out, 3), got {grid.shape}")
        
        input_5d = np.ascontiguousarray(input_data, dtype=np.float32)
        grid_5d = np.ascontiguousarray(grid, dtype=np.float32)
        
        N = input_5d.shape[0]
        C = input_5d.shape[1]
        D_in = input_5d.shape[2]
        H_in = input_5d.shape[3]
        W_in = input_5d.shape[4]
        D_out = grid_5d.shape[1]
        H_out = grid_5d.shape[2]
        W_out = grid_5d.shape[3]
        
        if grid_5d.shape[0] != N:
            raise ValueError(f"Batch size mismatch: input has {N}, grid has {grid_5d.shape[0]}")
        
        output_5d = np.empty((N, C, D_out, H_out, W_out), dtype=np.float32)
        
        input_ptr = <float*>cnp.PyArray_DATA(input_5d)
        grid_ptr = <float*>cnp.PyArray_DATA(grid_5d)
        output_ptr = <float*>cnp.PyArray_DATA(output_5d)
        
        if mode == "bilinear":
            if padding_mode == "zeros":
                _grid_sample_bilinear_zeros(input_ptr, grid_ptr, output_ptr, N, C, D_in, H_in, W_in, D_out, H_out, W_out)
            elif padding_mode == "border":
                _grid_sample_bilinear_border(input_ptr, grid_ptr, output_ptr, N, C, D_in, H_in, W_in, D_out, H_out, W_out)
            else:  # reflection
                _grid_sample_bilinear_reflection(input_ptr, grid_ptr, output_ptr, N, C, D_in, H_in, W_in, D_out, H_out, W_out)
        else:  # nearest
            if padding_mode == "zeros":
                _grid_sample_nearest_zeros(input_ptr, grid_ptr, output_ptr, N, C, D_in, H_in, W_in, D_out, H_out, W_out)
            elif padding_mode == "border":
                _grid_sample_nearest_border(input_ptr, grid_ptr, output_ptr, N, C, D_in, H_in, W_in, D_out, H_out, W_out)
            else:  # reflection
                _grid_sample_nearest_reflection(input_ptr, grid_ptr, output_ptr, N, C, D_in, H_in, W_in, D_out, H_out, W_out)
        
        return output_5d
    
    else:
        raise ValueError(f"Input must be 4D or 5D (got {input_data.ndim}). "
                         "Supported shapes: (N,C,H,W) or (N,C,D,H,W).")