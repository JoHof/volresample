from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from numpy cimport uint8_t, int16_t

# Portable software prefetch (read, low temporal locality)
cdef extern from *:
    """
    #if defined(__GNUC__) || defined(__clang__)
    #define _vol_prefetch_r(addr) __builtin_prefetch((const void*)(addr), 0, 0)
    #elif defined(_MSC_VER)
    #include <xmmintrin.h>
    #define _vol_prefetch_r(addr) _mm_prefetch((const char*)(addr), _MM_HINT_NTA)
    #else
    #define _vol_prefetch_r(addr) ((void)0)
    #endif
    """
    void _vol_prefetch_r(const void* addr) nogil

# Define fused type for supported dtypes in nearest neighbor
ctypedef fused numeric_type:
    uint8_t
    int16_t
    float

cdef inline void _nearest_gather_row(
    numeric_type* src_row,
    numeric_type* dst_row,
    int* w_indices,
    int out_w
) noexcept nogil:
    cdef int ow = 0
    cdef int limit = out_w - 8

    while ow <= limit:
        dst_row[ow] = src_row[w_indices[ow]]
        dst_row[ow + 1] = src_row[w_indices[ow + 1]]
        dst_row[ow + 2] = src_row[w_indices[ow + 2]]
        dst_row[ow + 3] = src_row[w_indices[ow + 3]]
        dst_row[ow + 4] = src_row[w_indices[ow + 4]]
        dst_row[ow + 5] = src_row[w_indices[ow + 5]]
        dst_row[ow + 6] = src_row[w_indices[ow + 6]]
        dst_row[ow + 7] = src_row[w_indices[ow + 7]]
        ow += 8

    while ow < out_w:
        dst_row[ow] = src_row[w_indices[ow]]
        ow += 1


cdef inline void _nearest_expand_row(
    numeric_type* src_row,
    numeric_type* dst_row,
    int source_count,
    int repeat_count
) noexcept nogil:
    cdef int src_x, repeat_idx, out_x = 0
    cdef numeric_type value

    for src_x in range(source_count):
        value = src_row[src_x]
        for repeat_idx in range(repeat_count):
            dst_row[out_x] = value
            out_x += 1


cdef inline void _nearest_stride_row(
    numeric_type* src_row,
    numeric_type* dst_row,
    int out_w,
    int step
) noexcept nogil:
    cdef int ow = 0
    cdef int src_x = 0
    cdef int step8 = step * 8
    cdef int limit = out_w - 8

    while ow <= limit:
        dst_row[ow] = src_row[src_x]
        dst_row[ow + 1] = src_row[src_x + step]
        dst_row[ow + 2] = src_row[src_x + 2 * step]
        dst_row[ow + 3] = src_row[src_x + 3 * step]
        dst_row[ow + 4] = src_row[src_x + 4 * step]
        dst_row[ow + 5] = src_row[src_x + 5 * step]
        dst_row[ow + 6] = src_row[src_x + 6 * step]
        dst_row[ow + 7] = src_row[src_x + 7 * step]
        src_x += step8
        ow += 8

    while ow < out_w:
        dst_row[ow] = src_row[src_x]
        src_x += step
        ow += 1


cdef inline void _nearest_repeat_rows(
    numeric_type* dst_row,
    int out_w,
    int repeat_count,
    size_t row_bytes
) noexcept nogil:
    cdef int repeat_idx
    cdef numeric_type* dst_repeat = dst_row + out_w

    for repeat_idx in range(1, repeat_count):
        memcpy(dst_repeat, dst_row, row_bytes)
        dst_repeat += out_w


cdef inline void _nearest_repeat_planes(
    numeric_type* dst_plane,
    int out_h_w,
    int repeat_count,
    size_t plane_bytes
) noexcept nogil:
    cdef int repeat_idx
    cdef numeric_type* dst_repeat = dst_plane + out_h_w

    for repeat_idx in range(1, repeat_count):
        memcpy(dst_repeat, dst_plane, plane_bytes)
        dst_repeat += out_h_w


cdef void _resample_nearest(
    numeric_type* data_ptr,
    numeric_type* output_ptr,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    float scale_d, float scale_h, float scale_w,
    bint align_corners
) noexcept nogil:
    """Perform nearest neighbor resampling on a 3D volume."""
    cdef int od, oh, ow, task_id
    cdef int next_od, next_oh
    cdef int depth_run, row_run
    cdef int depth_run_count = 0
    cdef int row_run_count = 0
    cdef int total_tasks
    cdef int d_step = 0
    cdef int h_step = 0
    cdef int w_step = 0
    cdef int base_d_offset
    cdef int row_offset
    cdef float src
    cdef int idx
    cdef int in_h_w = in_h * in_w
    cdef int out_h_w = out_h * out_w
    cdef size_t total_bytes
    cdef size_t row_bytes
    cdef size_t plane_bytes
    cdef int* index_buf
    cdef int* d_offsets
    cdef int* depth_run_starts
    cdef int* depth_run_lengths
    cdef int* depth_run_offsets
    cdef int* h_offsets
    cdef int* row_run_starts
    cdef int* row_run_lengths
    cdef int* row_run_offsets
    cdef int* w_indices

    cdef numeric_type* src_d
    cdef numeric_type* src_row
    cdef numeric_type* dst_row
    cdef numeric_type* dst_plane
    cdef int depth_out_offset

    cdef int w_base = 0
    cdef int w_repeat_count = 0
    cdef int w_source_count = 0
    cdef bint w_contiguous = True
    cdef bint w_replicate = False
    cdef bint d_regular_stride = True
    cdef bint h_regular_stride = True
    cdef bint w_regular_stride = True

    if in_d == out_d and in_h == out_h and in_w == out_w:
        total_bytes = <size_t>in_d * <size_t>in_h_w * sizeof(numeric_type)
        memcpy(output_ptr, data_ptr, total_bytes)
        return

    index_buf = <int*>malloc((4 * out_d + 4 * out_h + out_w) * sizeof(int))
    if index_buf == NULL:
        return

    d_offsets = index_buf
    depth_run_starts = d_offsets + out_d
    depth_run_lengths = depth_run_starts + out_d
    depth_run_offsets = depth_run_lengths + out_d
    h_offsets = depth_run_offsets + out_d
    row_run_starts = h_offsets + out_h
    row_run_lengths = row_run_starts + out_h
    row_run_offsets = row_run_lengths + out_h
    w_indices = row_run_offsets + out_h

    # Precompute depth offsets (d_idx * in_h_w baked in)
    for od in range(out_d):
        if align_corners:
            src = <float>od * <float>(in_d - 1) / <float>(out_d - 1) if out_d > 1 else 0.0
        else:
            src = (od + 0.5) * scale_d - 0.5
        idx = <int>(src + 0.5)
        if idx < 0:
            idx = 0
        elif idx >= in_d:
            idx = in_d - 1
        d_offsets[od] = idx * in_h_w

    # Precompute height row offsets (h_idx * in_w baked in)
    for oh in range(out_h):
        if align_corners:
            src = <float>oh * <float>(in_h - 1) / <float>(out_h - 1) if out_h > 1 else 0.0
        else:
            src = (oh + 0.5) * scale_h - 0.5
        idx = <int>(src + 0.5)
        if idx < 0:
            idx = 0
        elif idx >= in_h:
            idx = in_h - 1
        h_offsets[oh] = idx * in_w

    # Precompute width indices
    for ow in range(out_w):
        if align_corners:
            src = <float>ow * <float>(in_w - 1) / <float>(out_w - 1) if out_w > 1 else 0.0
        else:
            src = (ow + 0.5) * scale_w - 0.5
        idx = <int>(src + 0.5)
        if idx < 0:
            idx = 0
        elif idx >= in_w:
            idx = in_w - 1
        w_indices[ow] = idx

    w_base = w_indices[0]
    for ow in range(1, out_w):
        if w_indices[ow] != w_base + ow:
            w_contiguous = False
            break

    if not w_contiguous:
        w_repeat_count = 1
        while w_repeat_count < out_w and w_indices[w_repeat_count] == w_base:
            w_repeat_count += 1
        if w_repeat_count > 1 and out_w % w_repeat_count == 0:
            w_replicate = True
            w_source_count = out_w / w_repeat_count
            for ow in range(out_w):
                if w_indices[ow] != w_base + ow / w_repeat_count:
                    w_replicate = False
                    break

    if out_d > 1:
        d_step = d_offsets[1] - d_offsets[0]
        for od in range(2, out_d):
            if d_offsets[od] - d_offsets[od - 1] != d_step:
                d_regular_stride = False
                break

    if out_h > 1:
        h_step = h_offsets[1] - h_offsets[0]
        for oh in range(2, out_h):
            if h_offsets[oh] - h_offsets[oh - 1] != h_step:
                h_regular_stride = False
                break

    if out_w > 1:
        w_step = w_indices[1] - w_indices[0]
        if w_step <= 0:
            w_regular_stride = False
        else:
            for ow in range(2, out_w):
                if w_indices[ow] - w_indices[ow - 1] != w_step:
                    w_regular_stride = False
                    break

    row_bytes = <size_t>out_w * sizeof(numeric_type)

    # Constant-step mappings are common for integer downsample and crop cases.
    # Use a simpler depth-parallel kernel with direct strided loads to avoid
    # index-array traffic and row-task overhead in the hot path.
    if d_regular_stride and h_regular_stride and w_regular_stride and h_step > 0 and w_step > 0:
        for od in prange(out_d, schedule='static'):
            if out_d > 1:
                base_d_offset = d_offsets[0] + od * d_step
            else:
                base_d_offset = d_offsets[0]
            dst_plane = output_ptr + od * out_h_w

            for oh in range(out_h):
                if out_h > 1:
                    row_offset = h_offsets[0] + oh * h_step
                else:
                    row_offset = h_offsets[0]

                src_row = data_ptr + base_d_offset + row_offset
                dst_row = dst_plane + oh * out_w
                if w_contiguous:
                    memcpy(dst_row, src_row + w_base, row_bytes)
                else:
                    _nearest_stride_row(src_row + w_base, dst_row, out_w, w_step)

        free(index_buf)
        return

    od = 0
    while od < out_d:
        depth_run_starts[depth_run_count] = od
        depth_run_offsets[depth_run_count] = d_offsets[od]
        next_od = od + 1
        while next_od < out_d and d_offsets[next_od] == d_offsets[od]:
            next_od += 1
        depth_run_lengths[depth_run_count] = next_od - od
        depth_run_count += 1
        od = next_od

    oh = 0
    while oh < out_h:
        row_run_starts[row_run_count] = oh
        row_run_offsets[row_run_count] = h_offsets[oh]
        next_oh = oh + 1
        while next_oh < out_h and h_offsets[next_oh] == h_offsets[oh]:
            next_oh += 1
        row_run_lengths[row_run_count] = next_oh - oh
        row_run_count += 1
        oh = next_oh

    plane_bytes = <size_t>out_h_w * sizeof(numeric_type)
    total_tasks = depth_run_count * row_run_count

    # Generate the first row of each repeated run, then copy rows and planes.
    for task_id in prange(total_tasks, schedule='static'):
        depth_run = task_id / row_run_count
        row_run = task_id - depth_run * row_run_count

        src_d = data_ptr + depth_run_offsets[depth_run]
        src_row = src_d + row_run_offsets[row_run]
        depth_out_offset = depth_run_starts[depth_run] * out_h_w
        dst_row = output_ptr + depth_out_offset + row_run_starts[row_run] * out_w

        if w_contiguous:
            memcpy(dst_row, src_row + w_base, row_bytes)
        elif w_replicate:
            _nearest_expand_row(src_row + w_base, dst_row, w_source_count, w_repeat_count)
        else:
            _nearest_gather_row(src_row, dst_row, w_indices, out_w)

        if row_run_lengths[row_run] > 1:
            _nearest_repeat_rows(dst_row, out_w, row_run_lengths[row_run], row_bytes)

    for depth_run in prange(depth_run_count, schedule='static'):
        if depth_run_lengths[depth_run] > 1:
            dst_plane = output_ptr + depth_run_starts[depth_run] * out_h_w
            _nearest_repeat_planes(dst_plane, out_h_w, depth_run_lengths[depth_run], plane_bytes)

    free(index_buf)
