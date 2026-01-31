# Area resampling implementation (included via main file)

cdef void _resample_area(
    float* data_ptr,
    float* output_ptr,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    float scale_d, float scale_h, float scale_w
) noexcept nogil:
    """Area interpolation for 3D using raw pointers, matching PyTorch.
    
    PyTorch's area mode behavior:
    - For downsampling (scale >= 1.0): average the input pixels that map to each output pixel
    - For upsampling (scale < 1.0): replicate input pixels (nearest-neighbor style)
    
    This is applied per-dimension independently, allowing mixed scaling.
    """
    cdef int od, oh, ow, id_loop, ih, iw
    cdef int d_start_i, d_end_i, h_start_i, h_end_i, w_start_i, w_end_i
    cdef int total_count
    cdef float local_sum
    cdef int idx_out
    cdef int in_hw = in_h * in_w
    cdef int out_hw = out_h * out_w
    
    # Determine which dimensions are upsampling vs downsampling
    cdef bint d_upsample = scale_d < 1.0
    cdef bint h_upsample = scale_h < 1.0
    cdef bint w_upsample = scale_w < 1.0
    
    # Pre-allocate arrays for start/end indices
    cdef int* d_start_arr = <int*>malloc(out_d * sizeof(int))
    cdef int* d_end_arr = <int*>malloc(out_d * sizeof(int))
    cdef int* h_start_arr = <int*>malloc(out_h * sizeof(int))
    cdef int* h_end_arr = <int*>malloc(out_h * sizeof(int))
    cdef int* w_start_arr = <int*>malloc(out_w * sizeof(int))
    cdef int* w_end_arr = <int*>malloc(out_w * sizeof(int))
    
    cdef int i
    cdef int src_idx
    
    # Pre-compute depth ranges
    # For upsampling: use nearest neighbor (single source pixel)
    # For downsampling: compute range to average
    for i in range(out_d):
        if d_upsample:
            # Upsampling: nearest neighbor - pick the source pixel
            src_idx = <int>floor((i + 0.5) * scale_d)
            if src_idx >= in_d:
                src_idx = in_d - 1
            d_start_arr[i] = src_idx
            d_end_arr[i] = src_idx + 1
        else:
            # Downsampling: compute range to average
            d_start_arr[i] = <int>floor(i * scale_d)
            d_end_arr[i] = <int>ceil((i + 1) * scale_d)
            if d_end_arr[i] > in_d:
                d_end_arr[i] = in_d
    
    # Pre-compute height ranges
    for i in range(out_h):
        if h_upsample:
            src_idx = <int>floor((i + 0.5) * scale_h)
            if src_idx >= in_h:
                src_idx = in_h - 1
            h_start_arr[i] = src_idx
            h_end_arr[i] = src_idx + 1
        else:
            h_start_arr[i] = <int>floor(i * scale_h)
            h_end_arr[i] = <int>ceil((i + 1) * scale_h)
            if h_end_arr[i] > in_h:
                h_end_arr[i] = in_h
    
    # Pre-compute width ranges
    for i in range(out_w):
        if w_upsample:
            src_idx = <int>floor((i + 0.5) * scale_w)
            if src_idx >= in_w:
                src_idx = in_w - 1
            w_start_arr[i] = src_idx
            w_end_arr[i] = src_idx + 1
        else:
            w_start_arr[i] = <int>floor(i * scale_w)
            w_end_arr[i] = <int>ceil((i + 1) * scale_w)
            if w_end_arr[i] > in_w:
                w_end_arr[i] = in_w
    
    # Main loop - now using pre-computed indices
    for od in prange(out_d, schedule='static'):
        d_start_i = d_start_arr[od]
        d_end_i = d_end_arr[od]

        for oh in range(out_h):
            h_start_i = h_start_arr[oh]
            h_end_i = h_end_arr[oh]

            for ow in range(out_w):
                w_start_i = w_start_arr[ow]
                w_end_i = w_end_arr[ow]

                local_sum = 0.0

                for id_loop in range(d_start_i, d_end_i):
                    for ih in range(h_start_i, h_end_i):
                        for iw in range(w_start_i, w_end_i):
                            local_sum = local_sum + data_ptr[id_loop * in_hw + ih * in_w + iw]
                
                total_count = (d_end_i - d_start_i) * (h_end_i - h_start_i) * (w_end_i - w_start_i)
                idx_out = od * out_hw + oh * out_w + ow
                
                output_ptr[idx_out] = local_sum / <float>total_count
    
    # Free allocated memory
    free(d_start_arr)
    free(d_end_arr)
    free(h_start_arr)
    free(h_end_arr)
    free(w_start_arr)
    free(w_end_arr)
