# Cubic B-spline resampling — high-performance implementation
# Matches scipy.ndimage.zoom(order=3, mode='reflect'):
# - grid_mode=True when align_corners=False
# - grid_mode=False when align_corners=True

from libc.string cimport memcpy

# Prefetch hint
cdef extern from * nogil:
    """
    #if defined(__clang__) || defined(__GNUC__)
    static inline void volresample_prefetch(const void* addr, int rw, int locality) {
        __builtin_prefetch(addr, rw, locality);
    }
    #else
    static inline void volresample_prefetch(const void* addr, int rw, int locality) {
        (void)addr;
        (void)rw;
        (void)locality;
    }
    #endif
    """
    void volresample_prefetch(const void* addr, int rw, int locality)

# ---------------------------------------------------------------------------
# Boundary helper: reflect100 (scipy 'reflect')
# ---------------------------------------------------------------------------
cdef inline int _reflect100(int i, int size) noexcept nogil:
    cdef int period
    if size == 1:
        return 0
    period = 2 * size
    i = i % period
    if i < 0:
        i += period
    if i > size - 1:
        i = 2 * size - 1 - i
    return i

# ---------------------------------------------------------------------------
# Cubic B-spline basis weights
# ---------------------------------------------------------------------------
cdef inline void _cubic_weights(double t, double* w) noexcept nogil:
    cdef double t2 = t * t
    cdef double t3 = t2 * t
    cdef double u = 1.0 - t
    cdef double u2 = u * u
    cdef double u3 = u2 * u
    w[0] = u3 / 6.0
    w[1] = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0
    w[2] = (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0
    w[3] = t3 / 6.0

# ---------------------------------------------------------------------------
# IIR B-spline prefilter — double precision
# Uses contiguous temp buffers for strided axes to ensure IIR runs in L1.
# Gain fused into strided copy-in to avoid a separate pass.
# Software prefetch for large-stride axis 0.
# ---------------------------------------------------------------------------

cdef double POLE = -0.2679491924311228  # -2 + sqrt(3)
cdef double GAIN = 6.0


cdef inline void _prefilter_1d_inplace(double* data, int size,
                                       double z_n) noexcept nogil:
    """IIR B-spline prefilter on contiguous stride-1 data, in-place.
    Applies gain internally.  z_n = z^size (precomputed).
    """
    cdef double z = POLE
    cdef double g = GAIN
    cdef int i
    cdef double z_i, c0

    if size <= 1:
        return

    # Apply gain
    for i in range(size):
        data[i] = data[i] * g

    # Causal init (reflect boundary, matching scipy v1.17.1)
    z_i = z
    c0 = data[0]
    data[0] = data[0] + z_n * data[size - 1]
    for i in range(1, size):
        data[0] = data[0] + z_i * (data[i] + z_n * data[size - 1 - i])
        z_i = z_i * z
    data[0] = data[0] * z / (1.0 - z_n * z_n)
    data[0] = data[0] + c0

    # Causal recursion
    for i in range(1, size):
        data[i] = data[i] + z * data[i - 1]

    # Anticausal init
    data[size - 1] = data[size - 1] * z / (z - 1.0)

    # Anticausal recursion
    for i in range(size - 2, -1, -1):
        data[i] = z * (data[i + 1] - data[i])


cdef void _prefilter_3d(double* data, int d, int h, int w,
                       int nt) noexcept nogil:
    """Separable cubic B-spline prefilter along all 3 axes.

    - Axis 2 (width,  stride=1):   in-place
    - Axis 1 (height, stride=w):   per-thread contiguous temp
    - Axis 0 (depth,  stride=h*w): per-thread contiguous temp + prefetch
    """
    cdef int i, j, k, hw = h * w
    cdef int max_line = d if d > h else h
    cdef double* temps = <double*>malloc(nt * max_line * sizeof(double))
    if temps == NULL:
        return
    cdef int tid
    cdef double* temp
    cdef double gain_h, gain_d
    if h > 1:
        gain_h = GAIN
    else:
        gain_h = 1.0
    if d > 1:
        gain_d = GAIN
    else:
        gain_d = 1.0

    # Precompute z^N for each axis (avoids O(N-1) serial multiplies per line)
    cdef double z = POLE
    cdef double zz
    cdef double z_n_w, z_n_h, z_n_d
    zz = z
    for i in range(1, w):
        zz = zz * z
    z_n_w = zz
    zz = z
    for i in range(1, h):
        zz = zz * z
    z_n_h = zz
    zz = z
    for i in range(1, d):
        zz = zz * z
    z_n_d = zz

    # --- Axis 2 (width): stride=1, in-place ---
    for i in prange(d * h, schedule='static', num_threads=nt):
        _prefilter_1d_inplace(&data[i * w], w, z_n_w)

    # --- Axis 1 (height): stride=w, per-thread temp ---
    # Process per depth slice to keep working set in L2
    for i in prange(d, schedule='static', num_threads=nt):
        tid = omp_get_thread_num()
        temp = &temps[tid * max_line]
        for j in range(w):
            # Copy strided -> contiguous with gain fused
            for k in range(h):
                temp[k] = data[i * hw + j + k * w] * gain_h
            # Filter contiguous data (all in L1)
            _prefilter_1d_core_gained(temp, h, z_n_h)
            # Copy back
            for k in range(h):
                data[i * hw + j + k * w] = temp[k]

    # --- Axis 0 (depth): stride=h*w, per-thread temp + prefetch ---
    for i in prange(hw, schedule='static', num_threads=nt):
        tid = omp_get_thread_num()
        temp = &temps[tid * max_line]
        # Copy strided -> contiguous with gain fused + prefetch
        for k in range(d):
            if k + 8 < d:
                volresample_prefetch(<const void*>&data[(k + 8) * hw + i], 0, 0)
            temp[k] = data[k * hw + i] * gain_d
        # Filter in L1
        _prefilter_1d_core_gained(temp, d, z_n_d)
        # Copy back with prefetch
        for k in range(d):
            if k + 8 < d:
                volresample_prefetch(<const void*>&data[(k + 8) * hw + i], 1, 0)
            data[k * hw + i] = temp[k]

    free(temps)


cdef inline void _prefilter_1d_core_gained(double* data, int size,
                                            double z_n) noexcept nogil:
    """IIR B-spline prefilter on contiguous data that ALREADY has gain applied.
    z_n = z^size (precomputed).
    """
    cdef double z = POLE
    cdef int i
    cdef double z_i, c0

    if size <= 1:
        return

    # Causal init (reflect boundary, matching scipy v1.17.1)
    z_i = z
    c0 = data[0]
    data[0] = data[0] + z_n * data[size - 1]
    for i in range(1, size):
        data[0] = data[0] + z_i * (data[i] + z_n * data[size - 1 - i])
        z_i = z_i * z
    data[0] = data[0] * z / (1.0 - z_n * z_n)
    data[0] = data[0] + c0

    # Causal recursion
    for i in range(1, size):
        data[i] = data[i] + z * data[i - 1]

    # Anticausal init
    data[size - 1] = data[size - 1] * z / (z - 1.0)

    # Anticausal recursion
    for i in range(size - 2, -1, -1):
        data[i] = z * (data[i + 1] - data[i])


# ---------------------------------------------------------------------------
# Build reflected index LUT for one axis
# ---------------------------------------------------------------------------
cdef void _build_index_lut(int* idx_lut, double* weights, int out_size,
                           int in_size, float scale,
                           bint align_corners) noexcept nogil:
    cdef int i, tap, base_idx
    cdef double src, src_floor, t
    cdef double wt[4]

    for i in range(out_size):
        if align_corners:
            if out_size > 1:
                src = <double>i * <double>(in_size - 1) / <double>(out_size - 1)
            else:
                src = 0.0
        else:
            src = (<double>i + 0.5) * <double>scale - 0.5
        src_floor = floor(src)
        base_idx = <int>src_floor
        t = src - src_floor
        _cubic_weights(t, wt)
        weights[i * 4 + 0] = wt[0]
        weights[i * 4 + 1] = wt[1]
        weights[i * 4 + 2] = wt[2]
        weights[i * 4 + 3] = wt[3]
        for tap in range(4):
            idx_lut[i * 4 + tap] = _reflect100(base_idx + tap - 1, in_size)


# ---------------------------------------------------------------------------
# Main entry: tricubic B-spline resampling
# ---------------------------------------------------------------------------
cdef void _resample_cubic(
    float* data_ptr,
    float* output_ptr,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    float scale_d, float scale_h, float scale_w,
    int num_threads,
    bint align_corners
) noexcept nogil:
    cdef int total_in = in_d * in_h * in_w
    cdef int in_hw = in_h * in_w
    cdef int out_hw = out_h * out_w

    # --- Stage 1: Copy to double and prefilter ---
    cdef double* coeffs_d = <double*>malloc(total_in * sizeof(double))
    if coeffs_d == NULL:
        return

    cdef int i
    for i in prange(total_in, schedule='static', num_threads=num_threads):
        coeffs_d[i] = <double>data_ptr[i]

    _prefilter_3d(coeffs_d, in_d, in_h, in_w, num_threads)

    # --- Stage 2: Pre-compute index LUTs and weights ---
    cdef int* d_idx = <int*>malloc(out_d * 4 * sizeof(int))
    cdef double* d_w  = <double*>malloc(out_d * 4 * sizeof(double))
    cdef int* h_idx = <int*>malloc(out_h * 4 * sizeof(int))
    cdef double* h_w  = <double*>malloc(out_h * 4 * sizeof(double))
    cdef int* w_idx = <int*>malloc(out_w * 4 * sizeof(int))
    cdef double* w_w  = <double*>malloc(out_w * 4 * sizeof(double))

    _build_index_lut(d_idx, d_w, out_d, in_d, scale_d, align_corners)
    _build_index_lut(h_idx, h_w, out_h, in_h, scale_h, align_corners)
    _build_index_lut(w_idx, w_w, out_w, in_w, scale_w, align_corners)

    # --- Stage 3: Pre-compute row offsets ---
    cdef int* h_row_off = <int*>malloc(out_h * 4 * sizeof(int))
    cdef int oh, tap
    for oh in range(out_h):
        for tap in range(4):
            h_row_off[oh * 4 + tap] = h_idx[oh * 4 + tap] * in_w

    # --- Stage 4: Evaluate tricubic stencil (double precision) ---
    cdef int od, ow
    cdef int dd, hh
    cdef int slice_off, row_off
    cdef double wd, wdh
    cdef double val
    cdef int od4, oh4, ow4

    for od in prange(out_d, schedule='static', num_threads=num_threads):
        od4 = od * 4
        for oh in range(out_h):
            oh4 = oh * 4
            for ow in range(out_w):
                ow4 = ow * 4
                val = 0.0

                for dd in range(4):
                    slice_off = d_idx[od4 + dd] * in_hw
                    wd = d_w[od4 + dd]

                    for hh in range(4):
                        row_off = slice_off + h_row_off[oh4 + hh]
                        wdh = wd * h_w[oh4 + hh]

                        val = val + wdh * (
                            w_w[ow4    ] * coeffs_d[row_off + w_idx[ow4    ]] +
                            w_w[ow4 + 1] * coeffs_d[row_off + w_idx[ow4 + 1]] +
                            w_w[ow4 + 2] * coeffs_d[row_off + w_idx[ow4 + 2]] +
                            w_w[ow4 + 3] * coeffs_d[row_off + w_idx[ow4 + 3]])

                output_ptr[od * out_hw + oh * out_w + ow] = <float>val

    # --- Cleanup ---
    free(coeffs_d)
    free(d_idx)
    free(d_w)
    free(h_idx)
    free(h_w)
    free(w_idx)
    free(w_w)
    free(h_row_off)
