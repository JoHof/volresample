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

cnp.import_array()


cdef inline float clip(float val, float min_val, float max_val) nogil:
    """Clip value to [min_val, max_val]."""
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val


cdef inline void nearest_sample(
    float* data,
    int in_d, int in_h, int in_w,
    float d, float h, float w,
    float* result
) noexcept nogil:
    """Sample single value using nearest neighbor interpolation.
    
    Args:
        data: Input data pointer (contiguous C-order array)
        in_d, in_h, in_w: Input dimensions
        d, h, w: Floating-point coordinates to sample
        result: Output pointer (single float value)
    """
    # Round to nearest integer coordinates
    cdef int d_idx = <int>(d + 0.5)
    cdef int h_idx = <int>(h + 0.5)
    cdef int w_idx = <int>(w + 0.5)
    
    # Clip to valid range
    d_idx = <int>clip(d_idx, 0, in_d - 1)
    h_idx = <int>clip(h_idx, 0, in_h - 1)
    w_idx = <int>clip(w_idx, 0, in_w - 1)
    
    # Get value at nearest location
    cdef int idx = d_idx * in_h * in_w + h_idx * in_w + w_idx
    result[0] = data[idx]




def resample_3d(
    cnp.ndarray[cnp.float32_t, ndim=3] data,
    tuple size,
    str mode = "linear"
):
    """Resample 3D volume using specified interpolation mode.
    
    Args:
        data: Input array of shape (D, H, W), float32
        size: Target size (new_D, new_H, new_W)
        mode: Interpolation mode: 'nearest', 'linear', or 'area'
        
    Returns:
        Resampled array of shape (new_D, new_H, new_W), float32
    """
    cdef int in_d = data.shape[0]
    cdef int in_h = data.shape[1]
    cdef int in_w = data.shape[2]
    
    cdef int out_d = size[0]
    cdef int out_h = size[1]
    cdef int out_w = size[2]
    
    # Create output array
    cdef cnp.ndarray[cnp.float32_t, ndim=3] output = np.empty(
        (out_d, out_h, out_w), dtype=np.float32
    )
    
    # Compute scale factors (PyTorch F.interpolate behavior)
    cdef float scale_d = <float>in_d / <float>out_d
    cdef float scale_h = <float>in_h / <float>out_h
    cdef float scale_w = <float>in_w / <float>out_w
    
    cdef int od, oh, ow
    cdef float src_d, src_h, src_w
    cdef float* data_ptr = <float*>cnp.PyArray_DATA(data)
    cdef float* output_ptr = <float*>cnp.PyArray_DATA(output)
    cdef int idx_out, idx_in
    cdef int d_idx, h_idx, w_idx, d1_idx, h1_idx, w1_idx
    cdef int idx000, idx001, idx010, idx011, idx100, idx101, idx110, idx111
    cdef float d_floor, h_floor, w_floor
    cdef float fd, fh, fw, fd1, fh1, fw1
    cdef float v000, v001, v010, v011, v100, v101, v110, v111
    cdef float interp_w00, interp_w01, interp_w10, interp_w11, interp_h0, interp_h1
    
    if mode == "nearest":
        # Nearest neighbor interpolation - OPTIMIZED: inlined for performance
        for od in prange(out_d, nogil=True, schedule='static'):
            src_d = (od + 0.5) * scale_d - 0.5
            # Compute and clamp d index once per depth slice
            d_idx = <int>(src_d + 0.5)
            if d_idx < 0:
                d_idx = 0
            elif d_idx >= in_d:
                d_idx = in_d - 1
                
            for oh in range(out_h):
                src_h = (oh + 0.5) * scale_h - 0.5
                # Compute and clamp h index once per row
                h_idx = <int>(src_h + 0.5)
                if h_idx < 0:
                    h_idx = 0
                elif h_idx >= in_h:
                    h_idx = in_h - 1
                    
                for ow in range(out_w):
                    src_w = (ow + 0.5) * scale_w - 0.5
                    # Compute and clamp w index
                    w_idx = <int>(src_w + 0.5)
                    if w_idx < 0:
                        w_idx = 0
                    elif w_idx >= in_w:
                        w_idx = in_w - 1
                    
                    # Direct memory access - no function call
                    idx_in = d_idx * in_h * in_w + h_idx * in_w + w_idx
                    idx_out = od * out_h * out_w + oh * out_w + ow
                    output_ptr[idx_out] = data_ptr[idx_in]
    
    elif mode == "linear":
        # Trilinear interpolation - OPTIMIZED: inlined for performance
        for od in prange(out_d, nogil=True, schedule='static'):
            src_d = (od + 0.5) * scale_d - 0.5
            for oh in range(out_h):
                src_h = (oh + 0.5) * scale_h - 0.5
                for ow in range(out_w):
                    src_w = (ow + 0.5) * scale_w - 0.5
                    
                    # Compute integer coordinates (C floor, nogil safe)
                    d_floor = floor(src_d)
                    h_floor = floor(src_h)
                    w_floor = floor(src_w)
                    
                    d_idx = <int>d_floor
                    h_idx = <int>h_floor
                    w_idx = <int>w_floor
                    
                    d1_idx = d_idx + 1
                    h1_idx = h_idx + 1
                    w1_idx = w_idx + 1
                    
                    # Clamp to valid range (inline clip)
                    if d_idx < 0: d_idx = 0
                    elif d_idx >= in_d: d_idx = in_d - 1
                    if d1_idx < 0: d1_idx = 0
                    elif d1_idx >= in_d: d1_idx = in_d - 1
                    
                    if h_idx < 0: h_idx = 0
                    elif h_idx >= in_h: h_idx = in_h - 1
                    if h1_idx < 0: h1_idx = 0
                    elif h1_idx >= in_h: h1_idx = in_h - 1
                    
                    if w_idx < 0: w_idx = 0
                    elif w_idx >= in_w: w_idx = in_w - 1
                    if w1_idx < 0: w1_idx = 0
                    elif w1_idx >= in_w: w1_idx = in_w - 1
                    
                    # Compute fractional parts (reuse floor results)
                    fd = src_d - d_floor
                    fh = src_h - h_floor
                    fw = src_w - w_floor
                    
                    # Complementary weights
                    fd1 = 1.0 - fd
                    fh1 = 1.0 - fh
                    fw1 = 1.0 - fw
                    
                    # Compute 8 corner indices
                    idx000 = d_idx * in_h * in_w + h_idx * in_w + w_idx
                    idx001 = d_idx * in_h * in_w + h_idx * in_w + w1_idx
                    idx010 = d_idx * in_h * in_w + h1_idx * in_w + w_idx
                    idx011 = d_idx * in_h * in_w + h1_idx * in_w + w1_idx
                    idx100 = d1_idx * in_h * in_w + h_idx * in_w + w_idx
                    idx101 = d1_idx * in_h * in_w + h_idx * in_w + w1_idx
                    idx110 = d1_idx * in_h * in_w + h1_idx * in_w + w_idx
                    idx111 = d1_idx * in_h * in_w + h1_idx * in_w + w1_idx
                    
                    # Fetch 8 corner values
                    v000 = data_ptr[idx000]
                    v001 = data_ptr[idx001]
                    v010 = data_ptr[idx010]
                    v011 = data_ptr[idx011]
                    v100 = data_ptr[idx100]
                    v101 = data_ptr[idx101]
                    v110 = data_ptr[idx110]
                    v111 = data_ptr[idx111]
                    
                    # Trilinear interpolation
                    interp_w00 = v000 * fw1 + v001 * fw
                    interp_w01 = v010 * fw1 + v011 * fw
                    interp_w10 = v100 * fw1 + v101 * fw
                    interp_w11 = v110 * fw1 + v111 * fw
                    
                    interp_h0 = interp_w00 * fh1 + interp_w01 * fh
                    interp_h1 = interp_w10 * fh1 + interp_w11 * fh
                    
                    idx_out = od * out_h * out_w + oh * out_w + ow
                    output_ptr[idx_out] = interp_h0 * fd1 + interp_h1 * fd
    
    elif mode == "area":
        # Area interpolation (adaptive averaging for downsampling)
        # For upsampling, falls back to nearest (matching PyTorch behavior)
        if scale_d >= 1.0 and scale_h >= 1.0 and scale_w >= 1.0:
            # Downsampling: use area averaging
            output = _resample_area_3d(data, size, scale_d, scale_h, scale_w)
        else:
            # Upsampling: fall back to nearest (PyTorch's behavior)
            for od in prange(out_d, nogil=True, schedule='static'):
                src_d = (od + 0.5) * scale_d - 0.5
                for oh in range(out_h):
                    src_h = (oh + 0.5) * scale_h - 0.5
                    for ow in range(out_w):
                        src_w = (ow + 0.5) * scale_w - 0.5
                        
                        idx_out = od * out_h * out_w + oh * out_w + ow
                        
                        nearest_sample(
                            data_ptr, in_d, in_h, in_w,
                            src_d, src_h, src_w,
                            &output_ptr[idx_out]
                        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    return output


def resample_4d(
    cnp.ndarray[cnp.float32_t, ndim=4] data,
    tuple size,
    str mode = "linear"
):
    """Resample 4D volume (C, D, H, W) using specified interpolation mode.
    
    Args:
        data: Input array of shape (C, D, H, W), float32
        size: Target size (new_D, new_H, new_W) for spatial dimensions
        mode: Interpolation mode: 'nearest', 'linear', or 'area'
        
    Returns:
        Resampled array of shape (C, new_D, new_H, new_W), float32
    """
    cdef int n_channels = data.shape[0]
    cdef int in_d = data.shape[1]
    cdef int in_h = data.shape[2]
    cdef int in_w = data.shape[3]
    
    cdef int out_d = size[0]
    cdef int out_h = size[1]
    cdef int out_w = size[2]
    
    # Create output array
    cdef cnp.ndarray[cnp.float32_t, ndim=4] output = np.empty(
        (n_channels, out_d, out_h, out_w), dtype=np.float32
    )
    
    # Compute scale factors
    cdef float scale_d = <float>in_d / <float>out_d
    cdef float scale_h = <float>in_h / <float>out_h
    cdef float scale_w = <float>in_w / <float>out_w
    
    cdef int c, od, oh, ow
    cdef float src_d, src_h, src_w
    cdef int channel_size_in = in_d * in_h * in_w
    cdef int channel_size_out = out_d * out_h * out_w
    cdef int idx_out, idx_in, channel_in_offset, channel_out_offset
    cdef int d_idx, h_idx, w_idx, d1_idx, h1_idx, w1_idx
    cdef int idx000, idx001, idx010, idx011, idx100, idx101, idx110, idx111
    cdef float d_floor, h_floor, w_floor
    cdef float fd, fh, fw, fd1, fh1, fw1
    cdef float v000, v001, v010, v011, v100, v101, v110, v111
    cdef float interp_w00, interp_w01, interp_w10, interp_w11, interp_h0, interp_h1
    cdef float* channel_data_ptr
    # Tiling variables
    cdef int tile_d, tile_h, tile_w, d_start, d_end, h_start, h_end, w_start, w_end
    
    cdef float* data_ptr = <float*>cnp.PyArray_DATA(data)
    cdef float* output_ptr = <float*>cnp.PyArray_DATA(output)
    
    if mode == "nearest":
        # Nearest neighbor interpolation - OPTIMIZED: inlined for performance
        for c in prange(n_channels, nogil=True, schedule='static'):
            # Compute channel offsets ONCE per channel (not per voxel!)
            channel_in_offset = c * channel_size_in
            channel_out_offset = c * channel_size_out
            
            for od in range(out_d):
                src_d = (od + 0.5) * scale_d - 0.5
                # Compute and clamp d index once per depth slice
                d_idx = <int>(src_d + 0.5)
                if d_idx < 0:
                    d_idx = 0
                elif d_idx >= in_d:
                    d_idx = in_d - 1
                    
                for oh in range(out_h):
                    src_h = (oh + 0.5) * scale_h - 0.5
                    # Compute and clamp h index once per row
                    h_idx = <int>(src_h + 0.5)
                    if h_idx < 0:
                        h_idx = 0
                    elif h_idx >= in_h:
                        h_idx = in_h - 1
                        
                    for ow in range(out_w):
                        src_w = (ow + 0.5) * scale_w - 0.5
                        # Compute and clamp w index
                        w_idx = <int>(src_w + 0.5)
                        if w_idx < 0:
                            w_idx = 0
                        elif w_idx >= in_w:
                            w_idx = in_w - 1
                        
                        # Direct memory access - no function call
                        idx_in = d_idx * in_h * in_w + h_idx * in_w + w_idx
                        idx_out = channel_out_offset + od * out_h * out_w + oh * out_w + ow
                        output_ptr[idx_out] = data_ptr[channel_in_offset + idx_in]
    
    elif mode == "linear":
        # Trilinear interpolation - OPTIMIZED: inlined for performance
        for c in prange(n_channels, nogil=True, schedule='static'):
            # Compute channel offset ONCE per channel (not per voxel!)
            channel_data_ptr = data_ptr + c * channel_size_in
            channel_out_offset = c * channel_size_out
            
            for od in range(out_d):
                src_d = (od + 0.5) * scale_d - 0.5
                d_floor = floor(src_d)
                d_idx = <int>d_floor
                d1_idx = d_idx + 1
                
                # Clamp D indices once per depth slice
                if d_idx < 0: d_idx = 0
                elif d_idx >= in_d: d_idx = in_d - 1
                if d1_idx < 0: d1_idx = 0
                elif d1_idx >= in_d: d1_idx = in_d - 1
                
                fd = src_d - d_floor
                fd1 = 1.0 - fd
                
                for oh in range(out_h):
                    src_h = (oh + 0.5) * scale_h - 0.5
                    h_floor = floor(src_h)
                    h_idx = <int>h_floor
                    h1_idx = h_idx + 1
                    
                    # Clamp H indices once per row
                    if h_idx < 0: h_idx = 0
                    elif h_idx >= in_h: h_idx = in_h - 1
                    if h1_idx < 0: h1_idx = 0
                    elif h1_idx >= in_h: h1_idx = in_h - 1
                    
                    fh = src_h - h_floor
                    fh1 = 1.0 - fh
                    
                    for ow in range(out_w):
                        src_w = (ow + 0.5) * scale_w - 0.5
                        w_floor = floor(src_w)
                        w_idx = <int>w_floor
                        w1_idx = w_idx + 1
                        
                        # Clamp W indices
                        if w_idx < 0: w_idx = 0
                        elif w_idx >= in_w: w_idx = in_w - 1
                        if w1_idx < 0: w1_idx = 0
                        elif w1_idx >= in_w: w1_idx = in_w - 1
                        
                        fw = src_w - w_floor
                        fw1 = 1.0 - fw
                        
                        # Compute 8 corner indices
                        idx000 = d_idx * in_h * in_w + h_idx * in_w + w_idx
                        idx001 = d_idx * in_h * in_w + h_idx * in_w + w1_idx
                        idx010 = d_idx * in_h * in_w + h1_idx * in_w + w_idx
                        idx011 = d_idx * in_h * in_w + h1_idx * in_w + w1_idx
                        idx100 = d1_idx * in_h * in_w + h_idx * in_w + w_idx
                        idx101 = d1_idx * in_h * in_w + h_idx * in_w + w1_idx
                        idx110 = d1_idx * in_h * in_w + h1_idx * in_w + w_idx
                        idx111 = d1_idx * in_h * in_w + h1_idx * in_w + w1_idx
                        
                        # Fetch 8 corner values
                        v000 = channel_data_ptr[idx000]
                        v001 = channel_data_ptr[idx001]
                        v010 = channel_data_ptr[idx010]
                        v011 = channel_data_ptr[idx011]
                        v100 = channel_data_ptr[idx100]
                        v101 = channel_data_ptr[idx101]
                        v110 = channel_data_ptr[idx110]
                        v111 = channel_data_ptr[idx111]
                        
                        # Trilinear interpolation
                        interp_w00 = v000 * fw1 + v001 * fw
                        interp_w01 = v010 * fw1 + v011 * fw
                        interp_w10 = v100 * fw1 + v101 * fw
                        interp_w11 = v110 * fw1 + v111 * fw
                        
                        interp_h0 = interp_w00 * fh1 + interp_w01 * fh
                        interp_h1 = interp_w10 * fh1 + interp_w11 * fh
                        
                        idx_out = channel_out_offset + od * out_h * out_w + oh * out_w + ow
                        output_ptr[idx_out] = interp_h0 * fd1 + interp_h1 * fd
    
    elif mode == "area":
        # Area interpolation per channel
        if scale_d >= 1.0 and scale_h >= 1.0 and scale_w >= 1.0:
            # Downsampling: use OPTIMIZED parallel area averaging
            _resample_area_4d(data_ptr, output_ptr, n_channels,
                            in_d, in_h, in_w, out_d, out_h, out_w,
                            scale_d, scale_h, scale_w)
        else:
            # Upsampling: fall back to nearest (PyTorch's behavior)
            for c in prange(n_channels, nogil=True, schedule='static'):
                for od in range(out_d):
                    src_d = (od + 0.5) * scale_d - 0.5
                    for oh in range(out_h):
                        src_h = (oh + 0.5) * scale_h - 0.5
                        for ow in range(out_w):
                            src_w = (ow + 0.5) * scale_w - 0.5
                            
                            idx_out = c * channel_size_out + od * out_h * out_w + oh * out_w + ow
                            
                            nearest_sample(
                                data_ptr + c * channel_size_in,
                                in_d, in_h, in_w,
                                src_d, src_h, src_w,
                                &output_ptr[idx_out]
                            )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    return output


cdef void _resample_area_4d(
    float* data_ptr,
    float* output_ptr,
    int n_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    float scale_d, float scale_h, float scale_w
) noexcept nogil:
    """Area interpolation matching PyTorch's implementation.
    
    Uses integer-bounded regions with uniform weights (no fractional weighting).
    Regions can overlap at boundaries, matching PyTorch F.interpolate mode='area'.
    
    Algorithm:
        For each output voxel at index (od, oh, ow):
            d_start = floor(od * scale_d)
            d_end = ceil((od + 1) * scale_d), clamped to in_d
            Average inputs[d_start:d_end] with uniform weights
    """
    cdef int channel_size_in = in_d * in_h * in_w
    cdef int channel_size_out = out_d * out_h * out_w
    cdef int c, od, oh, ow, id, ih, iw
    cdef int d_start_i, d_end_i, h_start_i, h_end_i, w_start_i, w_end_i
    cdef int count_d, count_h, count_w, total_count
    cdef float uniform_weight, value
    cdef int idx_in_base, idx_out_base
    cdef float* sum_vals
    
    # Process each output voxel (spatial dimensions in outer loops with parallelization)
    for od in prange(out_d, nogil=True, schedule='static'):
        # Allocate temporary array per-thread (inside prange)
        sum_vals = <float*>malloc(n_channels * sizeof(float))
        
        # Compute integer bounds for D dimension (PyTorch-style)
        d_start_i = <int>(od * scale_d)  # floor
        d_end_i = <int>ceil((od + 1) * scale_d)  # ceil
        if d_end_i > in_d:
            d_end_i = in_d
        count_d = d_end_i - d_start_i
        
        for oh in range(out_h):
            # Compute integer bounds for H dimension
            h_start_i = <int>(oh * scale_h)
            h_end_i = <int>ceil((oh + 1) * scale_h)
            if h_end_i > in_h:
                h_end_i = in_h
            count_h = h_end_i - h_start_i
            
            for ow in range(out_w):
                # Compute integer bounds for W dimension
                w_start_i = <int>(ow * scale_w)
                w_end_i = <int>ceil((ow + 1) * scale_w)
                if w_end_i > in_w:
                    w_end_i = in_w
                count_w = w_end_i - w_start_i
                
                # Total number of input voxels contributing to this output
                total_count = count_d * count_h * count_w
                uniform_weight = 1.0 / <float>total_count
                
                # Initialize accumulators for all channels
                for c in range(n_channels):
                    sum_vals[c] = 0.0
                
                # Sum over all contributing input voxels with uniform weights
                for id in range(d_start_i, d_end_i):
                    for ih in range(h_start_i, h_end_i):
                        for iw in range(w_start_i, w_end_i):
                            # Compute base spatial index (reused for all channels)
                            idx_in_base = id * in_h * in_w + ih * in_w + iw
                            
                            # Process all channels with same spatial coordinates
                            for c in range(n_channels):
                                value = data_ptr[c * channel_size_in + idx_in_base]
                                sum_vals[c] = sum_vals[c] + value * uniform_weight
                
                # Write results for all channels
                for c in range(n_channels):
                    idx_out_base = c * channel_size_out + od * out_h * out_w + oh * out_w + ow
                    output_ptr[idx_out_base] = sum_vals[c]
        
        # Free per-thread allocation
        free(sum_vals)


def _resample_area_3d(
    cnp.ndarray[cnp.float32_t, ndim=3] data,
    tuple size,
    float scale_d,
    float scale_h,
    float scale_w
):
    """Area interpolation matching PyTorch's implementation.
    
    Uses integer-bounded regions with uniform weights (no fractional weighting).
    Regions can overlap at boundaries, matching PyTorch F.interpolate mode='area'.
    
    Args:
        data: Input array of shape (D, H, W), float32
        size: Target size (new_D, new_H, new_W)
        scale_d, scale_h, scale_w: Scale factors
        
    Returns:
        Resampled array of shape (new_D, new_H, new_W), float32
    """
    cdef int in_d = data.shape[0]
    cdef int in_h = data.shape[1]
    cdef int in_w = data.shape[2]
    
    cdef int out_d = size[0]
    cdef int out_h = size[1]
    cdef int out_w = size[2]
    
    cdef cnp.ndarray[cnp.float32_t, ndim=3] output = np.zeros(
        (out_d, out_h, out_w), dtype=np.float32
    )
    
    cdef float* data_ptr = <float*>cnp.PyArray_DATA(data)
    cdef float* output_ptr = <float*>cnp.PyArray_DATA(output)
    
    cdef int od, oh, ow, id, ih, iw
    cdef int d_start_i, d_end_i, h_start_i, h_end_i, w_start_i, w_end_i
    cdef int count_d, count_h, count_w, total_count
    cdef float uniform_weight, value
    cdef int idx_in, idx_out
    cdef float sum_val
    
    # Process each output voxel
    for od in prange(out_d, nogil=True, schedule='static'):
        # Compute integer bounds for D dimension (PyTorch-style)
        d_start_i = <int>(od * scale_d)  # floor
        d_end_i = <int>ceil((od + 1) * scale_d)  # ceil
        if d_end_i > in_d:
            d_end_i = in_d
        count_d = d_end_i - d_start_i
        
        for oh in range(out_h):
            # Compute integer bounds for H dimension
            h_start_i = <int>(oh * scale_h)
            h_end_i = <int>ceil((oh + 1) * scale_h)
            if h_end_i > in_h:
                h_end_i = in_h
            count_h = h_end_i - h_start_i
            
            for ow in range(out_w):
                # Compute integer bounds for W dimension
                w_start_i = <int>(ow * scale_w)
                w_end_i = <int>ceil((ow + 1) * scale_w)
                if w_end_i > in_w:
                    w_end_i = in_w
                count_w = w_end_i - w_start_i
                
                # Total number of input voxels contributing to this output
                total_count = count_d * count_h * count_w
                uniform_weight = 1.0 / <float>total_count
                
                sum_val = 0.0
                
                # Sum over all contributing input voxels with uniform weights
                for id in range(d_start_i, d_end_i):
                    for ih in range(h_start_i, h_end_i):
                        for iw in range(w_start_i, w_end_i):
                            idx_in = id * in_h * in_w + ih * in_w + iw
                            value = data_ptr[idx_in]
                            sum_val = sum_val + value * uniform_weight
                
                idx_out = od * out_h * out_w + oh * out_w + ow
                output_ptr[idx_out] = sum_val
    
    return output