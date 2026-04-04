# Cubic B-spline resampling — high-performance implementation
# Matches scipy.ndimage.zoom(order=3, mode='reflect', grid_mode=True)

from libc.string cimport memcpy

# Prefetch hint
cdef extern from * nogil:
    void __builtin_prefetch(const void* addr, int rw, int locality)

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
cdef inline void _cubic_weights_f(float t, float* w) noexcept nogil:
    cdef float t2 = t * t
    cdef float t3 = t2 * t
    cdef float u = 1.0 - t
    cdef float u2 = u * u
    cdef float u3 = u2 * u
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
    cdef double z_i, s

    if size <= 1:
        return

    # Apply gain
    for i in range(size):
        data[i] = data[i] * g

    # Causal init (full reflect boundary — correct for half-sample symmetric)
    z_i = z
    s = data[0] + z_n * data[size - 1]
    for i in range(1, size):
        s = s + z_i * (data[i] + z_n * data[size - 1 - i])
        z_i = z_i * z
    data[0] = data[0] + s * z / (1.0 - z_n * z_n)

    # Causal recursion
    for i in range(1, size):
        data[i] = data[i] + z * data[i - 1]

    # Anticausal init
    data[size - 1] = data[size - 1] * z / (z - 1.0)

    # Anticausal recursion
    for i in range(size - 2, -1, -1):
        data[i] = z * (data[i + 1] - data[i])


cdef void _prefilter_3d(double* data, int d, int h, int w) noexcept nogil:
    """Separable cubic B-spline prefilter along all 3 axes.

    - Axis 2 (width,  stride=1):   in-place
    - Axis 1 (height, stride=w):   per-thread contiguous temp
    - Axis 0 (depth,  stride=h*w): per-thread contiguous temp + prefetch
    """
    cdef int i, j, k, hw = h * w
    cdef int max_line = d if d > h else h
    cdef int nt = omp_get_max_threads()
    cdef double* temps = <double*>malloc(nt * max_line * sizeof(double))
    if temps == NULL:
        return
    cdef int tid
    cdef double* temp

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
    for i in prange(d * h, schedule='static'):
        _prefilter_1d_inplace(&data[i * w], w, z_n_w)

    # --- Axis 1 (height): stride=w, per-thread temp ---
    # Process per depth slice to keep working set in L2
    for i in prange(d, schedule='static'):
        tid = omp_get_thread_num()
        temp = &temps[tid * max_line]
        for j in range(w):
            # Copy strided -> contiguous with gain fused
            for k in range(h):
                temp[k] = data[i * hw + j + k * w] * GAIN
            # Filter contiguous data (all in L1)
            _prefilter_1d_core_gained(temp, h, z_n_h)
            # Copy back
            for k in range(h):
                data[i * hw + j + k * w] = temp[k]

    # --- Axis 0 (depth): stride=h*w, per-thread temp + prefetch ---
    for i in prange(hw, schedule='static'):
        tid = omp_get_thread_num()
        temp = &temps[tid * max_line]
        # Copy strided -> contiguous with gain fused + prefetch
        for k in range(d):
            if k + 8 < d:
                __builtin_prefetch(&data[(k + 8) * hw + i], 0, 0)
            temp[k] = data[k * hw + i] * GAIN
        # Filter in L1
        _prefilter_1d_core_gained(temp, d, z_n_d)
        # Copy back with prefetch
        for k in range(d):
            if k + 8 < d:
                __builtin_prefetch(&data[(k + 8) * hw + i], 1, 0)
            data[k * hw + i] = temp[k]

    free(temps)


cdef inline void _prefilter_1d_core_gained(double* data, int size,
                                            double z_n) noexcept nogil:
    """IIR B-spline prefilter on contiguous data that ALREADY has gain applied.
    z_n = z^size (precomputed).
    """
    cdef double z = POLE
    cdef int i
    cdef double z_i, s

    if size <= 1:
        return

    # Causal init (full reflect boundary)
    z_i = z
    s = data[0] + z_n * data[size - 1]
    for i in range(1, size):
        s = s + z_i * (data[i] + z_n * data[size - 1 - i])
        z_i = z_i * z
    data[0] = data[0] + s * z / (1.0 - z_n * z_n)

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
cdef void _build_index_lut(int* idx_lut, float* weights, int out_size,
                           int in_size, float scale) noexcept nogil:
    cdef int i, tap, base_idx
    cdef double src, src_floor, t
    cdef float wt[4]

    for i in range(out_size):
        src = (<double>i + 0.5) * <double>scale - 0.5
        src_floor = floor(src)
        base_idx = <int>src_floor
        t = src - src_floor
        _cubic_weights_f(<float>t, wt)
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
    float scale_d, float scale_h, float scale_w
) noexcept nogil:
    cdef int total_in = in_d * in_h * in_w
    cdef int in_hw = in_h * in_w
    cdef int out_hw = out_h * out_w

    # --- Stage 1: Copy to double and prefilter ---
    cdef double* coeffs_d = <double*>malloc(total_in * sizeof(double))
    if coeffs_d == NULL:
        return

    cdef int i
    for i in prange(total_in, schedule='static'):
        coeffs_d[i] = <double>data_ptr[i]

    _prefilter_3d(coeffs_d, in_d, in_h, in_w)

    # --- Stage 2: Convert to float32 for evaluation ---
    cdef float* coeffs = <float*>malloc(total_in * sizeof(float))
    if coeffs == NULL:
        free(coeffs_d)
        return

    for i in prange(total_in, schedule='static'):
        coeffs[i] = <float>coeffs_d[i]

    free(coeffs_d)

    # --- Stage 3: Pre-compute index LUTs and weights ---
    cdef int* d_idx = <int*>malloc(out_d * 4 * sizeof(int))
    cdef float* d_w  = <float*>malloc(out_d * 4 * sizeof(float))
    cdef int* h_idx = <int*>malloc(out_h * 4 * sizeof(int))
    cdef float* h_w  = <float*>malloc(out_h * 4 * sizeof(float))
    cdef int* w_idx = <int*>malloc(out_w * 4 * sizeof(int))
    cdef float* w_w  = <float*>malloc(out_w * 4 * sizeof(float))

    _build_index_lut(d_idx, d_w, out_d, in_d, scale_d)
    _build_index_lut(h_idx, h_w, out_h, in_h, scale_h)
    _build_index_lut(w_idx, w_w, out_w, in_w, scale_w)

    # --- Stage 4: Pre-compute row offsets ---
    cdef int* h_row_off = <int*>malloc(out_h * 4 * sizeof(int))
    cdef int oh, tap
    for oh in range(out_h):
        for tap in range(4):
            h_row_off[oh * 4 + tap] = h_idx[oh * 4 + tap] * in_w

    # --- Stage 5: Evaluate tricubic stencil ---
    cdef int od, ow
    cdef int dd, hh
    cdef int slice_off, row_off
    cdef float wd, wdh
    cdef float val
    cdef int od4, oh4, ow4

    for od in prange(out_d, schedule='static'):
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
                            w_w[ow4    ] * coeffs[row_off + w_idx[ow4    ]] +
                            w_w[ow4 + 1] * coeffs[row_off + w_idx[ow4 + 1]] +
                            w_w[ow4 + 2] * coeffs[row_off + w_idx[ow4 + 2]] +
                            w_w[ow4 + 3] * coeffs[row_off + w_idx[ow4 + 3]])

                output_ptr[od * out_hw + oh * out_w + ow] = val

    # --- Cleanup ---
    free(coeffs)
    free(d_idx)
    free(d_w)
    free(h_idx)
    free(h_w)
    free(w_idx)
    free(w_w)
    free(h_row_off)
