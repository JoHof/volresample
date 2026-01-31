# Area resampling implementation (included via main file)

cdef void _resample_area(
    float* data_ptr,
    float* output_ptr,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    float scale_d, float scale_h, float scale_w
) noexcept nogil:
    """Area interpolation for 3D using raw pointers, matching PyTorch."""
    cdef int od, oh, ow, id, ih, iw
    cdef int d_start_i, d_end_i, h_start_i, h_end_i, w_start_i, w_end_i
    cdef int total_count
    cdef float local_sum
    cdef int idx_out

    for od in prange(out_d, schedule='static'):
        d_start_i = <int>floor(od * scale_d)
        d_end_i = <int>ceil((od + 1) * scale_d)
        if d_end_i > in_d: d_end_i = in_d

        for oh in range(out_h):
            h_start_i = <int>floor(oh * scale_h)
            h_end_i = <int>ceil((oh + 1) * scale_h)
            if h_end_i > in_h: h_end_i = in_h

            for ow in range(out_w):
                w_start_i = <int>floor(ow * scale_w)
                w_end_i = <int>ceil((ow + 1) * scale_w)
                if w_end_i > in_w: w_end_i = in_w

                local_sum = 0.0

                for id in range(d_start_i, d_end_i):
                    for ih in range(h_start_i, h_end_i):
                        for iw in range(w_start_i, w_end_i):
                            local_sum = local_sum + data_ptr[id * in_h * in_w + ih * in_w + iw]
                
                total_count = (d_end_i - d_start_i) * (h_end_i - h_start_i) * (w_end_i - w_start_i)
                idx_out = od * out_h * out_w + oh * out_w + ow
                
                output_ptr[idx_out] = local_sum / <float>total_count
