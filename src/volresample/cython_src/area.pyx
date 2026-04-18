# Area resampling implementation (included via main file)

from libc.string cimport memset

cdef void _resample_area(
    float* data_ptr,
    float* output_ptr,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    float scale_d, float scale_h, float scale_w
) noexcept nogil:
    """Area interpolation for 3D using raw pointers, matching PyTorch.

    Loop order is d -> oh -> h -> ow -> k (streaming through input) with
    a per-thread accumulator array (out_h x out_w).  The outer loops
    iterate in input-sequential order (depth plane by depth plane, row by
    row) so the hardware prefetcher sees a streaming read pattern.
    """
    cdef int od, oh, ow, id_loop, ih, k
    cdef int d_start_i, d_end_i, h_start_i, h_end_i
    cdef int d_count, h_count, wc, total_count
    cdef int in_hw = in_h * in_w
    cdef int out_hw = out_h * out_w
    cdef float* sums
    cdef float* sums_row
    cdef float* row_base
    cdef float* seg_ptr
    cdef float acc
    cdef int sums_bytes = out_hw * <int>sizeof(float)

    # Pre-allocate arrays for start/end indices
    cdef int* d_start_arr = <int*>malloc(out_d * sizeof(int))
    cdef int* d_end_arr   = <int*>malloc(out_d * sizeof(int))
    cdef int* h_start_arr = <int*>malloc(out_h * sizeof(int))
    cdef int* h_end_arr   = <int*>malloc(out_h * sizeof(int))
    cdef int* w_start_arr = <int*>malloc(out_w * sizeof(int))
    cdef int* w_count_arr = <int*>malloc(out_w * sizeof(int))

    cdef int i

    for i in range(out_d):
        d_start_arr[i] = <int>floor(i * scale_d)
        d_end_arr[i]   = <int>ceil((i + 1) * scale_d)
        if d_end_arr[i] > in_d:
            d_end_arr[i] = in_d

    for i in range(out_h):
        h_start_arr[i] = <int>floor(i * scale_h)
        h_end_arr[i]   = <int>ceil((i + 1) * scale_h)
        if h_end_arr[i] > in_h:
            h_end_arr[i] = in_h

    for i in range(out_w):
        w_start_arr[i] = <int>floor(i * scale_w)
        w_count_arr[i] = <int>ceil((i + 1) * scale_w)
        if w_count_arr[i] > in_w:
            w_count_arr[i] = in_w
        w_count_arr[i] = w_count_arr[i] - w_start_arr[i]

    for od in prange(out_d, schedule='static'):
        sums = <float*>malloc(sums_bytes)
        memset(sums, 0, sums_bytes)

        d_start_i = d_start_arr[od]
        d_end_i   = d_end_arr[od]
        d_count   = d_end_i - d_start_i

        for id_loop in range(d_start_i, d_end_i):
            for oh in range(out_h):
                h_start_i = h_start_arr[oh]
                h_end_i   = h_end_arr[oh]
                sums_row  = sums + oh * out_w

                for ih in range(h_start_i, h_end_i):
                    row_base = data_ptr + id_loop * in_hw + ih * in_w

                    for ow in range(out_w):
                        seg_ptr = row_base + w_start_arr[ow]
                        wc = w_count_arr[ow]
                        acc = sums_row[ow]
                        for k in range(wc):
                            acc = acc + seg_ptr[k]
                        sums_row[ow] = acc

        for oh in range(out_h):
            h_count  = h_end_arr[oh] - h_start_arr[oh]
            sums_row = sums + oh * out_w
            for ow in range(out_w):
                total_count = d_count * h_count * w_count_arr[ow]
                output_ptr[od * out_hw + oh * out_w + ow] = sums_row[ow] / <float>total_count

        free(sums)

    free(d_start_arr)
    free(d_end_arr)
    free(h_start_arr)
    free(h_end_arr)
    free(w_start_arr)
    free(w_count_arr)
