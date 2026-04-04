# Cubic B-spline resampling implementation (included via main file)
# Matches scipy.ndimage.zoom(order=3, mode='reflect', grid_mode=True)
#
# Algorithm:
#   1. Separable IIR prefilter converts samples -> B-spline coefficients
#   2. Evaluate cubic B-spline basis at target positions
# Boundary: reflect100 (scipy 'reflect' = half-sample symmetric)
# Coordinate: src = (i + 0.5) * (in_size / out_size) - 0.5

# ---------------------------------------------------------------------------
# Boundary helper: reflect100 (scipy 'reflect')
# Pattern: ... d c b | a b c d | d c b a ...
# ---------------------------------------------------------------------------
cdef inline int _reflect100(int i, int size) noexcept nogil:
    """Map index i into [0, size-1] using half-sample symmetric reflection."""
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
# Cubic B-spline basis weights for fractional offset t in [0, 1)
# Taps at offsets -1, 0, +1, +2 relative to floor(src)
# ---------------------------------------------------------------------------
cdef inline void _cubic_weights(double t, double* w) noexcept nogil:
    """Compute 4 cubic B-spline weights for fractional offset t.

    w[0] = (1-t)^3 / 6          tap at floor(src) - 1
    w[1] = (4 - 6t^2 + 3t^3)/6  tap at floor(src)
    w[2] = (4 - 6u^2 + 3u^3)/6  tap at floor(src) + 1  (u = 1-t)
    w[3] = t^3 / 6              tap at floor(src) + 2
    """
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
# IIR B-spline prefilter (causal + anticausal passes)
# Cubic B-spline pole: z1 = -2 + sqrt(3) ≈ -0.26794919243112
# ---------------------------------------------------------------------------

cdef double POLE = -0.2679491924311228  # -2 + sqrt(3)
cdef double GAIN = 6.0  # overall gain = 6 for cubic B-spline


cdef void _prefilter_1d_inplace(double* data, int size, int stride) noexcept nogil:
    """Apply cubic B-spline IIR prefilter to a 1D signal in-place.

    The signal is accessed as data[i * stride].
    Boundary: reflect (half-sample symmetric), matching scipy NI_EXTEND_REFLECT.
    Algorithm from scipy's ni_splines.c (_init_causal_reflect / _init_anticausal_reflect).
    Uses double precision to match scipy's internal accuracy.
    """
    cdef double z = POLE
    cdef double lambda_val = GAIN
    cdef int i
    cdef double z_n, z_i, s

    if size == 1:
        return

    # --- Apply gain ---
    for i in range(size):
        data[i * stride] = data[i * stride] * lambda_val

    # --- Causal init (scipy _init_causal_reflect) ---
    # For half-sample symmetric boundary:
    #   sum = c[0] + z^n * c[n-1]
    #   for i = 1..n-1:  sum += z^i * (c[i] + z^n * c[n-1-i])
    #   c[0] += sum * z / (1 - z^(2n))
    z_n = z
    for i in range(1, size):
        z_n = z_n * z  # z_n = z^size after loop
    # z_n is now z^size
    z_i = z
    s = data[0] + z_n * data[(size - 1) * stride]
    for i in range(1, size):
        s = s + z_i * (data[i * stride] + z_n * data[(size - 1 - i) * stride])
        z_i = z_i * z
    data[0] = data[0] + s * z / (1.0 - z_n * z_n)

    # --- Causal recursion ---
    for i in range(1, size):
        data[i * stride] = data[i * stride] + z * data[(i - 1) * stride]

    # --- Anticausal init (scipy _init_anticausal_reflect) ---
    # For half-sample symmetric: c[n-1] *= z / (z - 1)
    data[(size - 1) * stride] = data[(size - 1) * stride] * z / (z - 1.0)

    # --- Anticausal recursion ---
    for i in range(size - 2, -1, -1):
        data[i * stride] = z * (data[(i + 1) * stride] - data[i * stride])


cdef void _prefilter_3d(double* data, int d, int h, int w) noexcept nogil:
    """Apply separable cubic B-spline prefilter along all 3 axes.

    Modifies data in-place. Data is stored in row-major (D, H, W) order.
    Stride for axis 0 (depth) = h*w, axis 1 (height) = w, axis 2 (width) = 1.
    """
    cdef int i, j

    # Along width (axis 2): stride=1, length=w
    for i in range(d):
        for j in range(h):
            _prefilter_1d_inplace(&data[i * h * w + j * w], w, 1)

    # Along height (axis 1): stride=w, length=h
    for i in range(d):
        for j in range(w):
            _prefilter_1d_inplace(&data[i * h * w + j], h, w)

    # Along depth (axis 0): stride=h*w, length=d
    for i in range(h):
        for j in range(w):
            _prefilter_1d_inplace(&data[i * w + j], d, h * w)


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
    """Tricubic B-spline resampling matching scipy.ndimage.zoom(order=3).

    1. Copy input into a working buffer
    2. Apply separable IIR prefilter (in-place) -> B-spline coefficients
    3. For each output voxel, evaluate the separable cubic B-spline kernel

    The approach is NOT separable in the evaluation (we do full 4x4x4 stencil)
    to avoid allocating large intermediate buffers. For production-quality
    performance with very large volumes, a separable approach would be faster.
    """
    cdef int total_in = in_d * in_h * in_w
    cdef int in_hw = in_h * in_w
    cdef int out_hw = out_h * out_w

    # Allocate double-precision working copy for prefilter.
    # Using double precision matches scipy's internal accuracy and avoids
    # float32 rounding errors accumulating through the IIR filter recursion.
    cdef double* coeffs = <double*>malloc(total_in * sizeof(double))
    if coeffs == NULL:
        return

    # Copy input (float32) to double-precision coeffs buffer
    cdef int i
    for i in range(total_in):
        coeffs[i] = <double>data_ptr[i]

    # Apply separable prefilter -> B-spline coefficients (in double precision)
    _prefilter_3d(coeffs, in_d, in_h, in_w)

    # Pre-compute per-axis weights and indices
    # For each output index along each axis, store: base index + 4 weights
    cdef int* d_base = <int*>malloc(out_d * sizeof(int))
    cdef double* d_w = <double*>malloc(out_d * 4 * sizeof(double))
    cdef int* h_base = <int*>malloc(out_h * sizeof(int))
    cdef double* h_w = <double*>malloc(out_h * 4 * sizeof(double))
    cdef int* w_base = <int*>malloc(out_w * sizeof(int))
    cdef double* w_w = <double*>malloc(out_w * 4 * sizeof(double))

    cdef double src, src_floor, t
    cdef int base

    # Pre-compute depth
    for i in range(out_d):
        src = (i + 0.5) * scale_d - 0.5
        src_floor = floor(src)
        base = <int>src_floor
        t = src - src_floor
        d_base[i] = base
        _cubic_weights(t, &d_w[i * 4])

    # Pre-compute height
    for i in range(out_h):
        src = (i + 0.5) * scale_h - 0.5
        src_floor = floor(src)
        base = <int>src_floor
        t = src - src_floor
        h_base[i] = base
        _cubic_weights(t, &h_w[i * 4])

    # Pre-compute width
    for i in range(out_w):
        src = (i + 0.5) * scale_w - 0.5
        src_floor = floor(src)
        base = <int>src_floor
        t = src - src_floor
        w_base[i] = base
        _cubic_weights(t, &w_w[i * 4])

    # Pre-compute reflected indices for each axis
    # d_base ranges from approx -1 to in_d, so offset -1..+2 covers -2..in_d+2
    # We'll compute them on-the-fly to keep memory bounded.

    # Main interpolation loop: 4x4x4 stencil per output voxel
    cdef int od, oh, ow
    cdef int dd, hh, ww
    cdef int di, hi, wi
    cdef double wd, wh, ww_val
    cdef double val
    cdef int out_idx

    for od in prange(out_d, schedule='static'):
        for oh in range(out_h):
            for ow in range(out_w):
                val = 0.0

                for dd in range(4):
                    di = _reflect100(d_base[od] + dd - 1, in_d)
                    wd = d_w[od * 4 + dd]

                    for hh in range(4):
                        hi = _reflect100(h_base[oh] + hh - 1, in_h)
                        wh = h_w[oh * 4 + hh]

                        for ww in range(4):
                            wi = _reflect100(w_base[ow] + ww - 1, in_w)
                            ww_val = w_w[ow * 4 + ww]

                            val = val + wd * wh * ww_val * coeffs[di * in_hw + hi * in_w + wi]

                output_ptr[od * out_hw + oh * out_w + ow] = <float>val

    # Free all allocations
    free(coeffs)
    free(d_base)
    free(d_w)
    free(h_base)
    free(h_w)
    free(w_base)
    free(w_w)
