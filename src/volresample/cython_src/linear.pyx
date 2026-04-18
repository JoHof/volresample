# Linear (trilinear) resampling implementation
# Included via main file, uses definitions from utils.pyx

# =============================================================================
# Helper: precompute 1D indices and weights into caller-provided buffers
# =============================================================================
cdef inline void _precompute_linear_coords(
    int out_n, int in_n, float scale, bint align_corners,
    int* idx0_arr, int* idx1_arr, float* f_arr, float* f1_arr
) noexcept nogil:
    cdef int i, idx0, idx1, n_max
    cdef float src, src_floor
    n_max = in_n - 1
    for i in range(out_n):
        if align_corners:
            if out_n > 1:
                src = <float>(<double>i * <double>n_max / <double>(out_n - 1))
            else:
                src = 0.0
        else:
            src = (i + 0.5) * scale - 0.5
        src_floor = floor(src)
        idx0 = <int>src_floor
        idx1 = idx0 + 1
        if idx0 < 0: idx0 = 0
        elif idx0 > n_max: idx0 = n_max
        if idx1 < 0: idx1 = 0
        elif idx1 > n_max: idx1 = n_max
        idx0_arr[i] = idx0
        idx1_arr[i] = idx1
        f_arr[i] = src - src_floor
        f1_arr[i] = 1.0 - (src - src_floor)


# =============================================================================
# Single-channel trilinear resampling (original hot path preserved)
# =============================================================================
cdef void _resample_linear(
    float* data_ptr,
    float* output_ptr,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    float scale_d, float scale_h, float scale_w,
    bint align_corners
) noexcept nogil:
    cdef int in_hw = in_h * in_w
    cdef int out_hw = out_h * out_w

    # Single consolidated allocation for indices + weights
    cdef int n_ints = 2 * (out_d + out_h + out_w)
    cdef int n_floats = 2 * (out_d + out_h + out_w)
    cdef int* int_buf = <int*>malloc(n_ints * sizeof(int))
    cdef float* float_buf = <float*>malloc(n_floats * sizeof(float))
    if int_buf == NULL or float_buf == NULL:
        if int_buf != NULL: free(int_buf)
        if float_buf != NULL: free(float_buf)
        return

    cdef int* d0_arr = int_buf
    cdef int* d1_arr = d0_arr + out_d
    cdef int* h0_arr = d1_arr + out_d
    cdef int* h1_arr = h0_arr + out_h
    cdef int* w0_arr = h1_arr + out_h
    cdef int* w1_arr = w0_arr + out_w
    cdef float* fd_arr = float_buf
    cdef float* fd1_arr = fd_arr + out_d
    cdef float* fh_arr = fd1_arr + out_d
    cdef float* fh1_arr = fh_arr + out_h
    cdef float* fw_arr = fh1_arr + out_h
    cdef float* fw1_arr = fw_arr + out_w

    _precompute_linear_coords(out_d, in_d, scale_d, align_corners,
                              d0_arr, d1_arr, fd_arr, fd1_arr)
    _precompute_linear_coords(out_h, in_h, scale_h, align_corners,
                              h0_arr, h1_arr, fh_arr, fh1_arr)
    _precompute_linear_coords(out_w, in_w, scale_w, align_corners,
                              w0_arr, w1_arr, fw_arr, fw1_arr)

    cdef int od, oh, ow
    cdef int d0, d1, h0, h1, w0, w1
    cdef float fd, fd1, fh, fh1, fw, fw1
    cdef float v000, v001, v010, v011, v100, v101, v110, v111
    cdef int base_d0, base_d1, row_h0, row_h1
    cdef int out_offset

    for od in prange(out_d, schedule='static'):
        d0 = d0_arr[od]
        d1 = d1_arr[od]
        fd = fd_arr[od]
        fd1 = fd1_arr[od]

        base_d0 = d0 * in_hw
        base_d1 = d1 * in_hw
        out_offset = od * out_hw

        for oh in range(out_h):
            h0 = h0_arr[oh]
            h1 = h1_arr[oh]
            fh = fh_arr[oh]
            fh1 = fh1_arr[oh]

            row_h0 = h0 * in_w
            row_h1 = h1 * in_w

            for ow in range(out_w):
                w0 = w0_arr[ow]
                w1 = w1_arr[ow]
                fw = fw_arr[ow]
                fw1 = fw1_arr[ow]

                v000 = data_ptr[base_d0 + row_h0 + w0]
                v001 = data_ptr[base_d0 + row_h0 + w1]
                v010 = data_ptr[base_d0 + row_h1 + w0]
                v011 = data_ptr[base_d0 + row_h1 + w1]
                v100 = data_ptr[base_d1 + row_h0 + w0]
                v101 = data_ptr[base_d1 + row_h0 + w1]
                v110 = data_ptr[base_d1 + row_h1 + w0]
                v111 = data_ptr[base_d1 + row_h1 + w1]

                output_ptr[out_offset + oh * out_w + ow] = (
                    fd1 * (fh1 * (fw1 * v000 + fw * v001) +
                           fh  * (fw1 * v010 + fw * v011)) +
                    fd  * (fh1 * (fw1 * v100 + fw * v101) +
                           fh  * (fw1 * v110 + fw * v111))
                )

    free(int_buf)
    free(float_buf)


# =============================================================================
# Multi-channel trilinear resampling with shared index precomputation
# Parallelizes over n_channels * out_d work items so that indices/weights
# are computed once and shared across all channels.
# =============================================================================
cdef void _resample_linear_multi(
    float* data_ptr,
    float* output_ptr,
    int n_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    float scale_d, float scale_h, float scale_w,
    bint align_corners
) noexcept nogil:
    cdef int in_hw = in_h * in_w
    cdef int in_vol = in_d * in_hw
    cdef int out_hw = out_h * out_w
    cdef int out_vol = out_d * out_hw

    # Single allocation for indices + weights
    cdef int n_ints = 2 * (out_d + out_h + out_w)
    cdef int n_floats = 2 * (out_d + out_h + out_w)
    cdef int* int_buf = <int*>malloc(n_ints * sizeof(int))
    cdef float* float_buf = <float*>malloc(n_floats * sizeof(float))
    if int_buf == NULL or float_buf == NULL:
        if int_buf != NULL: free(int_buf)
        if float_buf != NULL: free(float_buf)
        return

    cdef int* d0_arr = int_buf
    cdef int* d1_arr = d0_arr + out_d
    cdef int* h0_arr = d1_arr + out_d
    cdef int* h1_arr = h0_arr + out_h
    cdef int* w0_arr = h1_arr + out_h
    cdef int* w1_arr = w0_arr + out_w
    cdef float* fd_arr = float_buf
    cdef float* fd1_arr = fd_arr + out_d
    cdef float* fh_arr = fd1_arr + out_d
    cdef float* fh1_arr = fh_arr + out_h
    cdef float* fw_arr = fh1_arr + out_h
    cdef float* fw1_arr = fw_arr + out_w

    _precompute_linear_coords(out_d, in_d, scale_d, align_corners,
                              d0_arr, d1_arr, fd_arr, fd1_arr)
    _precompute_linear_coords(out_h, in_h, scale_h, align_corners,
                              h0_arr, h1_arr, fh_arr, fh1_arr)
    _precompute_linear_coords(out_w, in_w, scale_w, align_corners,
                              w0_arr, w1_arr, fw_arr, fw1_arr)

    cdef int task_id, od, ch, oh, ow
    cdef int total_tasks = n_channels * out_d
    cdef int d0, d1, h0, h1, w0, w1
    cdef float fd, fd1, fh, fh1, fw, fw1
    cdef float v000, v001, v010, v011, v100, v101, v110, v111
    cdef int base_d0, base_d1, row_h0, row_h1
    cdef int out_offset
    cdef float* ch_data
    cdef float* ch_out

    for task_id in prange(total_tasks, schedule='static'):
        ch = task_id / out_d
        od = task_id - ch * out_d

        ch_data = data_ptr + ch * in_vol
        ch_out = output_ptr + ch * out_vol

        d0 = d0_arr[od]
        d1 = d1_arr[od]
        fd = fd_arr[od]
        fd1 = fd1_arr[od]

        base_d0 = d0 * in_hw
        base_d1 = d1 * in_hw
        out_offset = od * out_hw

        for oh in range(out_h):
            h0 = h0_arr[oh]
            h1 = h1_arr[oh]
            fh = fh_arr[oh]
            fh1 = fh1_arr[oh]

            row_h0 = h0 * in_w
            row_h1 = h1 * in_w

            for ow in range(out_w):
                w0 = w0_arr[ow]
                w1 = w1_arr[ow]
                fw = fw_arr[ow]
                fw1 = fw1_arr[ow]

                v000 = ch_data[base_d0 + row_h0 + w0]
                v001 = ch_data[base_d0 + row_h0 + w1]
                v010 = ch_data[base_d0 + row_h1 + w0]
                v011 = ch_data[base_d0 + row_h1 + w1]
                v100 = ch_data[base_d1 + row_h0 + w0]
                v101 = ch_data[base_d1 + row_h0 + w1]
                v110 = ch_data[base_d1 + row_h1 + w0]
                v111 = ch_data[base_d1 + row_h1 + w1]

                ch_out[out_offset + oh * out_w + ow] = (
                    fd1 * (fh1 * (fw1 * v000 + fw * v001) +
                           fh  * (fw1 * v010 + fw * v011)) +
                    fd  * (fh1 * (fw1 * v100 + fw * v101) +
                           fh  * (fw1 * v110 + fw * v111))
                    )

    free(int_buf)
    free(float_buf)
