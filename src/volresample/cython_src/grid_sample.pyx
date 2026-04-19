# Grid sample implementation (included via main file)
# Matches PyTorch's grid_sample with align_corners=False (default)
#
# For align_corners=False:
#   unnorm_coord = ((grid_coord + 1) / 2) * size - 0.5
#
# Supports:
#   - 4D input (N, C, H_in, W_in) with 4D grid (N, H_out, W_out, 2) -> 4D output (N, C, H_out, W_out)
#   - 5D input (N, C, D_in, H_in, W_in) with 5D grid (N, D_out, H_out, W_out, 3) -> 5D output (N, C, D_out, H_out, W_out)
#
# Modes: bilinear (trilinear for 5D), nearest
# Padding modes: zeros, border, reflection

# Import rint for nearest-neighbor rounding (banker's rounding)
from libc.math cimport rint


# =============================================================================
# Helper functions for coordinate transformation and padding
# =============================================================================

cdef inline int round_to_nearest(float x) noexcept nogil:
    """Round to nearest integer using banker's rounding (ties to even).
    
    This matches PyTorch's std::nearbyint behavior.
    """
    return <int>rint(x)


cdef inline float unnormalize_coord(float coord, int size) noexcept nogil:
    """Convert normalized coordinate [-1, 1] to pixel coordinate.
    
    For align_corners=False:
        pixel = ((coord + 1) / 2) * size - 0.5
    """
    return ((coord + 1.0) * size) / 2.0 - 0.5


cdef inline float reflect_coord(float coord, float min_val, float max_val) noexcept nogil:
    """Reflect coordinate to stay within [min_val, max_val].
    
    Uses the same reflection logic as PyTorch.
    """
    cdef float range_size = max_val - min_val
    cdef float double_range = 2.0 * range_size
    
    # First, shift to [0, 2*range]
    coord = coord - min_val
    
    # Handle negative values
    if coord < 0:
        coord = -coord
    
    # Reduce to [0, 2*range]
    if double_range > 0:
        coord = coord - (<int>(coord / double_range)) * double_range
    
    # Reflect values > range back
    if coord > range_size:
        coord = double_range - coord
    
    return coord + min_val


cdef inline int apply_padding_zeros(float* coord_ptr, int size) noexcept nogil:
    """Apply zeros padding - return 0 if out of bounds (and sets coord to safe value)."""
    cdef float coord = coord_ptr[0]
    if coord < -0.5 or coord > <float>size - 0.5:
        coord_ptr[0] = 0.0  # Set to safe value
        return 0  # Mark as out of bounds
    return 1  # In bounds


cdef inline void apply_padding_border(float* coord_ptr, int size) noexcept nogil:
    """Apply border padding - clamp to valid range."""
    cdef float coord = coord_ptr[0]
    if coord < -0.5:
        coord_ptr[0] = -0.5
    elif coord > <float>size - 0.5:
        coord_ptr[0] = <float>size - 0.5


cdef inline void apply_padding_reflection(float* coord_ptr, int size) noexcept nogil:
    """Apply reflection padding."""
    cdef float coord = coord_ptr[0]
    # Reflect into valid range [-0.5, size - 0.5]
    coord_ptr[0] = reflect_coord(coord, -0.5, <float>size - 0.5)


# =============================================================================
# 2D Grid Sample (for 4D input: N, C, H, W)
# =============================================================================

cdef void _grid_sample_2d_bilinear_zeros(
    float* input_ptr,
    float* grid_ptr,
    float* output_ptr,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out
) noexcept nogil:
    """2D bilinear grid sample with zeros padding."""
    cdef int n, c, h, w
    cdef int h0, h1, w0, w1
    cdef float ix, iy
    cdef float h_frac, w_frac
    cdef float h_frac_1, w_frac_1
    cdef float v00, v01, v10, v11
    cdef int in_stride_n = C * H_in * W_in
    cdef int in_stride_c = H_in * W_in
    cdef int out_stride_n = C * H_out * W_out
    cdef int out_stride_c = H_out * W_out
    cdef int grid_stride_n = H_out * W_out * 2
    cdef int grid_idx, out_idx, in_base
    cdef int in_bounds_h0_w0, in_bounds_h0_w1, in_bounds_h1_w0, in_bounds_h1_w1
    cdef float val
    
    for n in range(N):
        for h in prange(H_out, schedule='static'):
            for w in range(W_out):
                # Get grid coordinates (x, y) where x is for W, y is for H
                grid_idx = n * grid_stride_n + h * W_out * 2 + w * 2
                ix = unnormalize_coord(grid_ptr[grid_idx], W_in)  # x -> W
                iy = unnormalize_coord(grid_ptr[grid_idx + 1], H_in)  # y -> H
                
                # Get interpolation corners
                w0 = <int>floor(ix)
                h0 = <int>floor(iy)
                w1 = w0 + 1
                h1 = h0 + 1
                
                # Compute interpolation weights
                w_frac = ix - <float>w0
                h_frac = iy - <float>h0
                w_frac_1 = 1.0 - w_frac
                h_frac_1 = 1.0 - h_frac
                
                # Check bounds for each corner
                in_bounds_h0_w0 = (h0 >= 0 and h0 < H_in and w0 >= 0 and w0 < W_in)
                in_bounds_h0_w1 = (h0 >= 0 and h0 < H_in and w1 >= 0 and w1 < W_in)
                in_bounds_h1_w0 = (h1 >= 0 and h1 < H_in and w0 >= 0 and w0 < W_in)
                in_bounds_h1_w1 = (h1 >= 0 and h1 < H_in and w1 >= 0 and w1 < W_in)
                
                for c in range(C):
                    in_base = n * in_stride_n + c * in_stride_c
                    
                    # Get values (zero if out of bounds)
                    if in_bounds_h0_w0:
                        v00 = input_ptr[in_base + h0 * W_in + w0]
                    else:
                        v00 = 0.0
                    
                    if in_bounds_h0_w1:
                        v01 = input_ptr[in_base + h0 * W_in + w1]
                    else:
                        v01 = 0.0
                    
                    if in_bounds_h1_w0:
                        v10 = input_ptr[in_base + h1 * W_in + w0]
                    else:
                        v10 = 0.0
                    
                    if in_bounds_h1_w1:
                        v11 = input_ptr[in_base + h1 * W_in + w1]
                    else:
                        v11 = 0.0
                    
                    # Bilinear interpolation
                    val = (h_frac_1 * (w_frac_1 * v00 + w_frac * v01) +
                           h_frac * (w_frac_1 * v10 + w_frac * v11))
                    
                    out_idx = n * out_stride_n + c * out_stride_c + h * W_out + w
                    output_ptr[out_idx] = val


cdef void _grid_sample_2d_bilinear_border(
    float* input_ptr,
    float* grid_ptr,
    float* output_ptr,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out
) noexcept nogil:
    """2D bilinear grid sample with border padding."""
    cdef int n, c, h, w
    cdef int h0, h1, w0, w1
    cdef float ix, iy
    cdef float h_frac, w_frac
    cdef float h_frac_1, w_frac_1
    cdef float v00, v01, v10, v11
    cdef int in_stride_n = C * H_in * W_in
    cdef int in_stride_c = H_in * W_in
    cdef int out_stride_n = C * H_out * W_out
    cdef int out_stride_c = H_out * W_out
    cdef int grid_stride_n = H_out * W_out * 2
    cdef int grid_idx, out_idx, in_base
    cdef float val
    cdef int H_max = H_in - 1
    cdef int W_max = W_in - 1
    
    for n in range(N):
        for h in prange(H_out, schedule='static'):
            for w in range(W_out):
                grid_idx = n * grid_stride_n + h * W_out * 2 + w * 2
                ix = unnormalize_coord(grid_ptr[grid_idx], W_in)
                iy = unnormalize_coord(grid_ptr[grid_idx + 1], H_in)
                
                # Get interpolation corners
                w0 = <int>floor(ix)
                h0 = <int>floor(iy)
                w1 = w0 + 1
                h1 = h0 + 1
                
                w_frac = ix - <float>w0
                h_frac = iy - <float>h0
                w_frac_1 = 1.0 - w_frac
                h_frac_1 = 1.0 - h_frac
                
                # Clamp indices to valid range
                if w0 < 0: w0 = 0
                elif w0 > W_max: w0 = W_max
                if w1 < 0: w1 = 0
                elif w1 > W_max: w1 = W_max
                if h0 < 0: h0 = 0
                elif h0 > H_max: h0 = H_max
                if h1 < 0: h1 = 0
                elif h1 > H_max: h1 = H_max
                
                for c in range(C):
                    in_base = n * in_stride_n + c * in_stride_c
                    
                    v00 = input_ptr[in_base + h0 * W_in + w0]
                    v01 = input_ptr[in_base + h0 * W_in + w1]
                    v10 = input_ptr[in_base + h1 * W_in + w0]
                    v11 = input_ptr[in_base + h1 * W_in + w1]
                    
                    val = (h_frac_1 * (w_frac_1 * v00 + w_frac * v01) +
                           h_frac * (w_frac_1 * v10 + w_frac * v11))
                    
                    out_idx = n * out_stride_n + c * out_stride_c + h * W_out + w
                    output_ptr[out_idx] = val


cdef inline int reflect_bound(int idx, int size) noexcept nogil:
    """Reflect index to stay within [0, size-1]."""
    if size == 1:
        return 0
    if idx < 0:
        idx = -idx - 1
        idx = idx % (2 * size)
        if idx >= size:
            idx = 2 * size - 1 - idx
    elif idx >= size:
        idx = idx % (2 * size)
        if idx >= size:
            idx = 2 * size - 1 - idx
    return idx


cdef void _grid_sample_2d_bilinear_reflection(
    float* input_ptr,
    float* grid_ptr,
    float* output_ptr,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out
) noexcept nogil:
    """2D bilinear grid sample with reflection padding."""
    cdef int n, c, h, w
    cdef int h0, h1, w0, w1
    cdef int h0_r, h1_r, w0_r, w1_r
    cdef float ix, iy
    cdef float h_frac, w_frac
    cdef float h_frac_1, w_frac_1
    cdef float v00, v01, v10, v11
    cdef int in_stride_n = C * H_in * W_in
    cdef int in_stride_c = H_in * W_in
    cdef int out_stride_n = C * H_out * W_out
    cdef int out_stride_c = H_out * W_out
    cdef int grid_stride_n = H_out * W_out * 2
    cdef int grid_idx, out_idx, in_base
    cdef float val
    
    for n in range(N):
        for h in prange(H_out, schedule='static'):
            for w in range(W_out):
                grid_idx = n * grid_stride_n + h * W_out * 2 + w * 2
                ix = unnormalize_coord(grid_ptr[grid_idx], W_in)
                iy = unnormalize_coord(grid_ptr[grid_idx + 1], H_in)
                
                w0 = <int>floor(ix)
                h0 = <int>floor(iy)
                w1 = w0 + 1
                h1 = h0 + 1
                
                w_frac = ix - <float>w0
                h_frac = iy - <float>h0
                w_frac_1 = 1.0 - w_frac
                h_frac_1 = 1.0 - h_frac
                
                # Reflect indices
                w0_r = reflect_bound(w0, W_in)
                w1_r = reflect_bound(w1, W_in)
                h0_r = reflect_bound(h0, H_in)
                h1_r = reflect_bound(h1, H_in)
                
                for c in range(C):
                    in_base = n * in_stride_n + c * in_stride_c
                    
                    v00 = input_ptr[in_base + h0_r * W_in + w0_r]
                    v01 = input_ptr[in_base + h0_r * W_in + w1_r]
                    v10 = input_ptr[in_base + h1_r * W_in + w0_r]
                    v11 = input_ptr[in_base + h1_r * W_in + w1_r]
                    
                    val = (h_frac_1 * (w_frac_1 * v00 + w_frac * v01) +
                           h_frac * (w_frac_1 * v10 + w_frac * v11))
                    
                    out_idx = n * out_stride_n + c * out_stride_c + h * W_out + w
                    output_ptr[out_idx] = val


cdef void _grid_sample_2d_nearest_zeros(
    float* input_ptr,
    float* grid_ptr,
    float* output_ptr,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out
) noexcept nogil:
    """2D nearest neighbor grid sample with zeros padding."""
    cdef int n, c, h, w
    cdef int h_idx, w_idx
    cdef float ix, iy
    cdef int in_stride_n = C * H_in * W_in
    cdef int in_stride_c = H_in * W_in
    cdef int out_stride_n = C * H_out * W_out
    cdef int out_stride_c = H_out * W_out
    cdef int grid_stride_n = H_out * W_out * 2
    cdef int grid_idx, out_idx, in_base
    cdef int in_bounds
    cdef float val
    
    for n in range(N):
        for h in prange(H_out, schedule='static'):
            for w in range(W_out):
                grid_idx = n * grid_stride_n + h * W_out * 2 + w * 2
                ix = unnormalize_coord(grid_ptr[grid_idx], W_in)
                iy = unnormalize_coord(grid_ptr[grid_idx + 1], H_in)
                
                # Round to nearest using banker's rounding (matches PyTorch)
                w_idx = round_to_nearest(ix)
                h_idx = round_to_nearest(iy)
                
                in_bounds = (h_idx >= 0 and h_idx < H_in and w_idx >= 0 and w_idx < W_in)
                
                for c in range(C):
                    out_idx = n * out_stride_n + c * out_stride_c + h * W_out + w
                    if in_bounds:
                        in_base = n * in_stride_n + c * in_stride_c
                        output_ptr[out_idx] = input_ptr[in_base + h_idx * W_in + w_idx]
                    else:
                        output_ptr[out_idx] = 0.0


cdef void _grid_sample_2d_nearest_border(
    float* input_ptr,
    float* grid_ptr,
    float* output_ptr,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out
) noexcept nogil:
    """2D nearest neighbor grid sample with border padding."""
    cdef int n, c, h, w
    cdef int h_idx, w_idx
    cdef float ix, iy
    cdef int in_stride_n = C * H_in * W_in
    cdef int in_stride_c = H_in * W_in
    cdef int out_stride_n = C * H_out * W_out
    cdef int out_stride_c = H_out * W_out
    cdef int grid_stride_n = H_out * W_out * 2
    cdef int grid_idx, out_idx, in_base
    cdef int H_max = H_in - 1
    cdef int W_max = W_in - 1
    
    for n in range(N):
        for h in prange(H_out, schedule='static'):
            for w in range(W_out):
                grid_idx = n * grid_stride_n + h * W_out * 2 + w * 2
                ix = unnormalize_coord(grid_ptr[grid_idx], W_in)
                iy = unnormalize_coord(grid_ptr[grid_idx + 1], H_in)
                
                # Round to nearest using banker's rounding (matches PyTorch)
                w_idx = round_to_nearest(ix)
                h_idx = round_to_nearest(iy)
                
                # Clamp to valid range
                if w_idx < 0: w_idx = 0
                elif w_idx > W_max: w_idx = W_max
                if h_idx < 0: h_idx = 0
                elif h_idx > H_max: h_idx = H_max
                
                for c in range(C):
                    in_base = n * in_stride_n + c * in_stride_c
                    out_idx = n * out_stride_n + c * out_stride_c + h * W_out + w
                    output_ptr[out_idx] = input_ptr[in_base + h_idx * W_in + w_idx]


cdef void _grid_sample_2d_nearest_reflection(
    float* input_ptr,
    float* grid_ptr,
    float* output_ptr,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out
) noexcept nogil:
    """2D nearest neighbor grid sample with reflection padding."""
    cdef int n, c, h, w
    cdef int h_idx, w_idx
    cdef float ix, iy
    cdef int in_stride_n = C * H_in * W_in
    cdef int in_stride_c = H_in * W_in
    cdef int out_stride_n = C * H_out * W_out
    cdef int out_stride_c = H_out * W_out
    cdef int grid_stride_n = H_out * W_out * 2
    cdef int grid_idx, out_idx, in_base
    
    for n in range(N):
        for h in prange(H_out, schedule='static'):
            for w in range(W_out):
                grid_idx = n * grid_stride_n + h * W_out * 2 + w * 2
                ix = unnormalize_coord(grid_ptr[grid_idx], W_in)
                iy = unnormalize_coord(grid_ptr[grid_idx + 1], H_in)
                
                # Round to nearest using banker's rounding (matches PyTorch)
                w_idx = round_to_nearest(ix)
                h_idx = round_to_nearest(iy)
                
                # Reflect
                w_idx = reflect_bound(w_idx, W_in)
                h_idx = reflect_bound(h_idx, H_in)
                
                for c in range(C):
                    in_base = n * in_stride_n + c * in_stride_c
                    out_idx = n * out_stride_n + c * out_stride_c + h * W_out + w
                    output_ptr[out_idx] = input_ptr[in_base + h_idx * W_in + w_idx]


# =============================================================================
# 3D Grid Sample (for 5D input: N, C, D, H, W)
# =============================================================================

cdef void _grid_sample_bilinear_zeros(
    float* input_ptr,
    float* grid_ptr,
    float* output_ptr,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    float fill_value=0.0
) noexcept nogil:
    """3D trilinear grid sample with constant fill padding.
    
    Parallelizes over N * C * D_out work items so that batch and channel
    dimensions contribute to thread utilization.
    Out-of-bounds samples use fill_value (default 0.0 for zeros padding).
    """
    cdef int task_id, n, c, d, h, w, remainder
    cdef int d0, d1, h0, h1, w0, w1
    cdef float ix, iy, iz
    cdef float d_frac, h_frac, w_frac
    cdef float d_frac_1, h_frac_1, w_frac_1
    cdef float v000, v001, v010, v011, v100, v101, v110, v111
    cdef int in_stride_n = C * D_in * H_in * W_in
    cdef int in_stride_c = D_in * H_in * W_in
    cdef int in_stride_d = H_in * W_in
    cdef int out_stride_n = C * D_out * H_out * W_out
    cdef int out_stride_c = D_out * H_out * W_out
    cdef int out_stride_d = H_out * W_out
    cdef int grid_stride_n = D_out * H_out * W_out * 3
    cdef int grid_stride_d = H_out * W_out * 3
    cdef int grid_idx, out_idx, in_base
    cdef int ib000, ib001, ib010, ib011, ib100, ib101, ib110, ib111
    cdef float val
    cdef int total_tasks = N * C * D_out
    cdef int CD = C * D_out

    for task_id in prange(total_tasks, schedule='static'):
        n = task_id / CD
        remainder = task_id - n * CD
        c = remainder / D_out
        d = remainder - c * D_out

        in_base = n * in_stride_n + c * in_stride_c

        for h in range(H_out):
            for w in range(W_out):
                # Grid coordinates: x (W), y (H), z (D)
                grid_idx = n * grid_stride_n + d * grid_stride_d + h * W_out * 3 + w * 3
                ix = unnormalize_coord(grid_ptr[grid_idx], W_in)      # x -> W
                iy = unnormalize_coord(grid_ptr[grid_idx + 1], H_in)  # y -> H
                iz = unnormalize_coord(grid_ptr[grid_idx + 2], D_in)  # z -> D

                w0 = <int>floor(ix)
                h0 = <int>floor(iy)
                d0 = <int>floor(iz)
                w1 = w0 + 1
                h1 = h0 + 1
                d1 = d0 + 1

                w_frac = ix - <float>w0
                h_frac = iy - <float>h0
                d_frac = iz - <float>d0
                w_frac_1 = 1.0 - w_frac
                h_frac_1 = 1.0 - h_frac
                d_frac_1 = 1.0 - d_frac

                # Check bounds for each corner
                ib000 = (d0 >= 0 and d0 < D_in and h0 >= 0 and h0 < H_in and w0 >= 0 and w0 < W_in)
                ib001 = (d0 >= 0 and d0 < D_in and h0 >= 0 and h0 < H_in and w1 >= 0 and w1 < W_in)
                ib010 = (d0 >= 0 and d0 < D_in and h1 >= 0 and h1 < H_in and w0 >= 0 and w0 < W_in)
                ib011 = (d0 >= 0 and d0 < D_in and h1 >= 0 and h1 < H_in and w1 >= 0 and w1 < W_in)
                ib100 = (d1 >= 0 and d1 < D_in and h0 >= 0 and h0 < H_in and w0 >= 0 and w0 < W_in)
                ib101 = (d1 >= 0 and d1 < D_in and h0 >= 0 and h0 < H_in and w1 >= 0 and w1 < W_in)
                ib110 = (d1 >= 0 and d1 < D_in and h1 >= 0 and h1 < H_in and w0 >= 0 and w0 < W_in)
                ib111 = (d1 >= 0 and d1 < D_in and h1 >= 0 and h1 < H_in and w1 >= 0 and w1 < W_in)

                # Get values (fill_value if out of bounds)
                v000 = input_ptr[in_base + d0 * in_stride_d + h0 * W_in + w0] if ib000 else fill_value
                v001 = input_ptr[in_base + d0 * in_stride_d + h0 * W_in + w1] if ib001 else fill_value
                v010 = input_ptr[in_base + d0 * in_stride_d + h1 * W_in + w0] if ib010 else fill_value
                v011 = input_ptr[in_base + d0 * in_stride_d + h1 * W_in + w1] if ib011 else fill_value
                v100 = input_ptr[in_base + d1 * in_stride_d + h0 * W_in + w0] if ib100 else fill_value
                v101 = input_ptr[in_base + d1 * in_stride_d + h0 * W_in + w1] if ib101 else fill_value
                v110 = input_ptr[in_base + d1 * in_stride_d + h1 * W_in + w0] if ib110 else fill_value
                v111 = input_ptr[in_base + d1 * in_stride_d + h1 * W_in + w1] if ib111 else fill_value

                # Trilinear interpolation
                val = (d_frac_1 * (h_frac_1 * (w_frac_1 * v000 + w_frac * v001) +
                                   h_frac * (w_frac_1 * v010 + w_frac * v011)) +
                       d_frac * (h_frac_1 * (w_frac_1 * v100 + w_frac * v101) +
                                 h_frac * (w_frac_1 * v110 + w_frac * v111)))

                out_idx = n * out_stride_n + c * out_stride_c + d * out_stride_d + h * W_out + w
                output_ptr[out_idx] = val


cdef void _grid_sample_bilinear_border(
    float* input_ptr,
    float* grid_ptr,
    float* output_ptr,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) noexcept nogil:
    """3D trilinear grid sample with border padding.
    
    Parallelizes over N * C * D_out work items.
    """
    cdef int task_id, n, c, d, h, w, remainder
    cdef int d0, d1, h0, h1, w0, w1
    cdef float ix, iy, iz
    cdef float d_frac, h_frac, w_frac
    cdef float d_frac_1, h_frac_1, w_frac_1
    cdef float v000, v001, v010, v011, v100, v101, v110, v111
    cdef int in_stride_n = C * D_in * H_in * W_in
    cdef int in_stride_c = D_in * H_in * W_in
    cdef int in_stride_d = H_in * W_in
    cdef int out_stride_n = C * D_out * H_out * W_out
    cdef int out_stride_c = D_out * H_out * W_out
    cdef int out_stride_d = H_out * W_out
    cdef int grid_stride_n = D_out * H_out * W_out * 3
    cdef int grid_stride_d = H_out * W_out * 3
    cdef int grid_idx, out_idx, in_base
    cdef float val
    cdef int D_max = D_in - 1
    cdef int H_max = H_in - 1
    cdef int W_max = W_in - 1
    cdef int total_tasks = N * C * D_out
    cdef int CD = C * D_out

    for task_id in prange(total_tasks, schedule='static'):
        n = task_id / CD
        remainder = task_id - n * CD
        c = remainder / D_out
        d = remainder - c * D_out

        in_base = n * in_stride_n + c * in_stride_c

        for h in range(H_out):
            for w in range(W_out):
                grid_idx = n * grid_stride_n + d * grid_stride_d + h * W_out * 3 + w * 3
                ix = unnormalize_coord(grid_ptr[grid_idx], W_in)
                iy = unnormalize_coord(grid_ptr[grid_idx + 1], H_in)
                iz = unnormalize_coord(grid_ptr[grid_idx + 2], D_in)

                w0 = <int>floor(ix)
                h0 = <int>floor(iy)
                d0 = <int>floor(iz)
                w1 = w0 + 1
                h1 = h0 + 1
                d1 = d0 + 1

                w_frac = ix - <float>w0
                h_frac = iy - <float>h0
                d_frac = iz - <float>d0
                w_frac_1 = 1.0 - w_frac
                h_frac_1 = 1.0 - h_frac
                d_frac_1 = 1.0 - d_frac

                # Clamp indices
                if w0 < 0: w0 = 0
                elif w0 > W_max: w0 = W_max
                if w1 < 0: w1 = 0
                elif w1 > W_max: w1 = W_max
                if h0 < 0: h0 = 0
                elif h0 > H_max: h0 = H_max
                if h1 < 0: h1 = 0
                elif h1 > H_max: h1 = H_max
                if d0 < 0: d0 = 0
                elif d0 > D_max: d0 = D_max
                if d1 < 0: d1 = 0
                elif d1 > D_max: d1 = D_max

                v000 = input_ptr[in_base + d0 * in_stride_d + h0 * W_in + w0]
                v001 = input_ptr[in_base + d0 * in_stride_d + h0 * W_in + w1]
                v010 = input_ptr[in_base + d0 * in_stride_d + h1 * W_in + w0]
                v011 = input_ptr[in_base + d0 * in_stride_d + h1 * W_in + w1]
                v100 = input_ptr[in_base + d1 * in_stride_d + h0 * W_in + w0]
                v101 = input_ptr[in_base + d1 * in_stride_d + h0 * W_in + w1]
                v110 = input_ptr[in_base + d1 * in_stride_d + h1 * W_in + w0]
                v111 = input_ptr[in_base + d1 * in_stride_d + h1 * W_in + w1]

                val = (d_frac_1 * (h_frac_1 * (w_frac_1 * v000 + w_frac * v001) +
                                   h_frac * (w_frac_1 * v010 + w_frac * v011)) +
                       d_frac * (h_frac_1 * (w_frac_1 * v100 + w_frac * v101) +
                                 h_frac * (w_frac_1 * v110 + w_frac * v111)))

                out_idx = n * out_stride_n + c * out_stride_c + d * out_stride_d + h * W_out + w
                output_ptr[out_idx] = val


cdef void _grid_sample_bilinear_reflection(
    float* input_ptr,
    float* grid_ptr,
    float* output_ptr,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) noexcept nogil:
    """3D trilinear grid sample with reflection padding.
    
    Parallelizes over N * C * D_out work items.
    """
    cdef int task_id, n, c, d, h, w, remainder
    cdef int d0, d1, h0, h1, w0, w1
    cdef int d0_r, d1_r, h0_r, h1_r, w0_r, w1_r
    cdef float ix, iy, iz
    cdef float d_frac, h_frac, w_frac
    cdef float d_frac_1, h_frac_1, w_frac_1
    cdef float v000, v001, v010, v011, v100, v101, v110, v111
    cdef int in_stride_n = C * D_in * H_in * W_in
    cdef int in_stride_c = D_in * H_in * W_in
    cdef int in_stride_d = H_in * W_in
    cdef int out_stride_n = C * D_out * H_out * W_out
    cdef int out_stride_c = D_out * H_out * W_out
    cdef int out_stride_d = H_out * W_out
    cdef int grid_stride_n = D_out * H_out * W_out * 3
    cdef int grid_stride_d = H_out * W_out * 3
    cdef int grid_idx, out_idx, in_base
    cdef float val
    cdef int total_tasks = N * C * D_out
    cdef int CD = C * D_out

    for task_id in prange(total_tasks, schedule='static'):
        n = task_id / CD
        remainder = task_id - n * CD
        c = remainder / D_out
        d = remainder - c * D_out

        in_base = n * in_stride_n + c * in_stride_c

        for h in range(H_out):
            for w in range(W_out):
                grid_idx = n * grid_stride_n + d * grid_stride_d + h * W_out * 3 + w * 3
                ix = unnormalize_coord(grid_ptr[grid_idx], W_in)
                iy = unnormalize_coord(grid_ptr[grid_idx + 1], H_in)
                iz = unnormalize_coord(grid_ptr[grid_idx + 2], D_in)

                w0 = <int>floor(ix)
                h0 = <int>floor(iy)
                d0 = <int>floor(iz)
                w1 = w0 + 1
                h1 = h0 + 1
                d1 = d0 + 1

                w_frac = ix - <float>w0
                h_frac = iy - <float>h0
                d_frac = iz - <float>d0
                w_frac_1 = 1.0 - w_frac
                h_frac_1 = 1.0 - h_frac
                d_frac_1 = 1.0 - d_frac

                # Reflect indices
                w0_r = reflect_bound(w0, W_in)
                w1_r = reflect_bound(w1, W_in)
                h0_r = reflect_bound(h0, H_in)
                h1_r = reflect_bound(h1, H_in)
                d0_r = reflect_bound(d0, D_in)
                d1_r = reflect_bound(d1, D_in)

                v000 = input_ptr[in_base + d0_r * in_stride_d + h0_r * W_in + w0_r]
                v001 = input_ptr[in_base + d0_r * in_stride_d + h0_r * W_in + w1_r]
                v010 = input_ptr[in_base + d0_r * in_stride_d + h1_r * W_in + w0_r]
                v011 = input_ptr[in_base + d0_r * in_stride_d + h1_r * W_in + w1_r]
                v100 = input_ptr[in_base + d1_r * in_stride_d + h0_r * W_in + w0_r]
                v101 = input_ptr[in_base + d1_r * in_stride_d + h0_r * W_in + w1_r]
                v110 = input_ptr[in_base + d1_r * in_stride_d + h1_r * W_in + w0_r]
                v111 = input_ptr[in_base + d1_r * in_stride_d + h1_r * W_in + w1_r]

                val = (d_frac_1 * (h_frac_1 * (w_frac_1 * v000 + w_frac * v001) +
                                   h_frac * (w_frac_1 * v010 + w_frac * v011)) +
                       d_frac * (h_frac_1 * (w_frac_1 * v100 + w_frac * v101) +
                                 h_frac * (w_frac_1 * v110 + w_frac * v111)))

                out_idx = n * out_stride_n + c * out_stride_c + d * out_stride_d + h * W_out + w
                output_ptr[out_idx] = val


cdef void _grid_sample_nearest_zeros(
    numeric_type* input_ptr,
    float* grid_ptr,
    numeric_type* output_ptr,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    numeric_type fill_value=0
) noexcept nogil:
    """3D nearest neighbor grid sample with constant fill padding.
    
    Parallelizes over N * C * D_out work items.
    Out-of-bounds samples use fill_value (default 0 for zeros padding).
    """
    cdef int task_id, n, c, d, h, w, remainder
    cdef int d_idx, h_idx, w_idx
    cdef float ix, iy, iz
    cdef int in_stride_n = C * D_in * H_in * W_in
    cdef int in_stride_c = D_in * H_in * W_in
    cdef int in_stride_d = H_in * W_in
    cdef int out_stride_n = C * D_out * H_out * W_out
    cdef int out_stride_c = D_out * H_out * W_out
    cdef int out_stride_d = H_out * W_out
    cdef int grid_stride_n = D_out * H_out * W_out * 3
    cdef int grid_stride_d = H_out * W_out * 3
    cdef int grid_idx, out_idx, in_base
    cdef int in_bounds
    cdef int total_tasks = N * C * D_out
    cdef int CD = C * D_out

    for task_id in prange(total_tasks, schedule='static'):
        n = task_id / CD
        remainder = task_id - n * CD
        c = remainder / D_out
        d = remainder - c * D_out

        in_base = n * in_stride_n + c * in_stride_c

        for h in range(H_out):
            for w in range(W_out):
                grid_idx = n * grid_stride_n + d * grid_stride_d + h * W_out * 3 + w * 3
                ix = unnormalize_coord(grid_ptr[grid_idx], W_in)
                iy = unnormalize_coord(grid_ptr[grid_idx + 1], H_in)
                iz = unnormalize_coord(grid_ptr[grid_idx + 2], D_in)

                # Round to nearest using banker's rounding (matches PyTorch)
                w_idx = round_to_nearest(ix)
                h_idx = round_to_nearest(iy)
                d_idx = round_to_nearest(iz)

                out_idx = n * out_stride_n + c * out_stride_c + d * out_stride_d + h * W_out + w
                in_bounds = (d_idx >= 0 and d_idx < D_in and
                             h_idx >= 0 and h_idx < H_in and
                             w_idx >= 0 and w_idx < W_in)

                if in_bounds:
                    output_ptr[out_idx] = input_ptr[in_base + d_idx * in_stride_d + h_idx * W_in + w_idx]
                else:
                    output_ptr[out_idx] = fill_value


cdef void _grid_sample_nearest_border(
    numeric_type* input_ptr,
    float* grid_ptr,
    numeric_type* output_ptr,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) noexcept nogil:
    """3D nearest neighbor grid sample with border padding.
    
    Parallelizes over N * C * D_out work items.
    """
    cdef int task_id, n, c, d, h, w, remainder
    cdef int d_idx, h_idx, w_idx
    cdef float ix, iy, iz
    cdef int in_stride_n = C * D_in * H_in * W_in
    cdef int in_stride_c = D_in * H_in * W_in
    cdef int in_stride_d = H_in * W_in
    cdef int out_stride_n = C * D_out * H_out * W_out
    cdef int out_stride_c = D_out * H_out * W_out
    cdef int out_stride_d = H_out * W_out
    cdef int grid_stride_n = D_out * H_out * W_out * 3
    cdef int grid_stride_d = H_out * W_out * 3
    cdef int grid_idx, out_idx, in_base
    cdef int D_max = D_in - 1
    cdef int H_max = H_in - 1
    cdef int W_max = W_in - 1
    cdef int total_tasks = N * C * D_out
    cdef int CD = C * D_out

    for task_id in prange(total_tasks, schedule='static'):
        n = task_id / CD
        remainder = task_id - n * CD
        c = remainder / D_out
        d = remainder - c * D_out

        in_base = n * in_stride_n + c * in_stride_c

        for h in range(H_out):
            for w in range(W_out):
                grid_idx = n * grid_stride_n + d * grid_stride_d + h * W_out * 3 + w * 3
                ix = unnormalize_coord(grid_ptr[grid_idx], W_in)
                iy = unnormalize_coord(grid_ptr[grid_idx + 1], H_in)
                iz = unnormalize_coord(grid_ptr[grid_idx + 2], D_in)

                # Round to nearest using banker's rounding (matches PyTorch)
                w_idx = round_to_nearest(ix)
                h_idx = round_to_nearest(iy)
                d_idx = round_to_nearest(iz)

                # Clamp to valid range
                if w_idx < 0: w_idx = 0
                elif w_idx > W_max: w_idx = W_max
                if h_idx < 0: h_idx = 0
                elif h_idx > H_max: h_idx = H_max
                if d_idx < 0: d_idx = 0
                elif d_idx > D_max: d_idx = D_max

                out_idx = n * out_stride_n + c * out_stride_c + d * out_stride_d + h * W_out + w
                output_ptr[out_idx] = input_ptr[in_base + d_idx * in_stride_d + h_idx * W_in + w_idx]


cdef void _grid_sample_nearest_reflection(
    numeric_type* input_ptr,
    float* grid_ptr,
    numeric_type* output_ptr,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) noexcept nogil:
    """3D nearest neighbor grid sample with reflection padding.
    
    Parallelizes over N * C * D_out work items.
    """
    cdef int task_id, n, c, d, h, w, remainder
    cdef int d_idx, h_idx, w_idx
    cdef float ix, iy, iz
    cdef int in_stride_n = C * D_in * H_in * W_in
    cdef int in_stride_c = D_in * H_in * W_in
    cdef int in_stride_d = H_in * W_in
    cdef int out_stride_n = C * D_out * H_out * W_out
    cdef int out_stride_c = D_out * H_out * W_out
    cdef int out_stride_d = H_out * W_out
    cdef int grid_stride_n = D_out * H_out * W_out * 3
    cdef int grid_stride_d = H_out * W_out * 3
    cdef int grid_idx, out_idx, in_base
    cdef int total_tasks = N * C * D_out
    cdef int CD = C * D_out

    for task_id in prange(total_tasks, schedule='static'):
        n = task_id / CD
        remainder = task_id - n * CD
        c = remainder / D_out
        d = remainder - c * D_out

        in_base = n * in_stride_n + c * in_stride_c

        for h in range(H_out):
            for w in range(W_out):
                grid_idx = n * grid_stride_n + d * grid_stride_d + h * W_out * 3 + w * 3
                ix = unnormalize_coord(grid_ptr[grid_idx], W_in)
                iy = unnormalize_coord(grid_ptr[grid_idx + 1], H_in)
                iz = unnormalize_coord(grid_ptr[grid_idx + 2], D_in)

                # Round to nearest using banker's rounding (matches PyTorch)
                w_idx = round_to_nearest(ix)
                h_idx = round_to_nearest(iy)
                d_idx = round_to_nearest(iz)

                # Reflect
                w_idx = reflect_bound(w_idx, W_in)
                h_idx = reflect_bound(h_idx, H_in)
                d_idx = reflect_bound(d_idx, D_in)

                out_idx = n * out_stride_n + c * out_stride_c + d * out_stride_d + h * W_out + w
                output_ptr[out_idx] = input_ptr[in_base + d_idx * in_stride_d + h_idx * W_in + w_idx]
