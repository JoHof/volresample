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

cdef void _resample_nearest(
    numeric_type* data_ptr,
    numeric_type* output_ptr,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    float scale_d, float scale_h, float scale_w
) noexcept nogil:
    """Perform nearest neighbor resampling on a 3D volume."""
    cdef int od, oh, ow
    cdef float src
    cdef int idx
    cdef int in_h_w = in_h * in_w
    cdef int out_h_w = out_h * out_w

    # Single allocation: d_offsets[out_d] | h_offsets[out_h] | w_indices[out_w]
    # d_offsets stores d_idx * in_h_w  (multiply precomputed, not repeated in prange)
    # h_offsets stores h_idx * in_w    (multiply precomputed, not repeated per row)
    cdef int* d_offsets = <int*>malloc((out_d + out_h + out_w) * sizeof(int))
    if d_offsets == NULL:
        return
    cdef int* h_offsets = d_offsets + out_d
    cdef int* w_indices = h_offsets + out_h

    cdef numeric_type* src_d
    cdef numeric_type* src_row
    cdef numeric_type* dst_row
    cdef int out_offset

    cdef int w_base
    cdef bint w_contiguous
    cdef size_t row_bytes

    # Precompute depth offsets (d_idx * in_h_w baked in)
    for od in range(out_d):
        src = (od + 0.5) * scale_d - 0.5
        idx = <int>(src + 0.5)
        if idx < 0:
            idx = 0
        elif idx >= in_d:
            idx = in_d - 1
        d_offsets[od] = idx * in_h_w

    # Precompute height row offsets (h_idx * in_w baked in)
    for oh in range(out_h):
        src = (oh + 0.5) * scale_h - 0.5
        idx = <int>(src + 0.5)
        if idx < 0:
            idx = 0
        elif idx >= in_h:
            idx = in_h - 1
        h_offsets[oh] = idx * in_w

    # Precompute width indices
    for ow in range(out_w):
        src = (ow + 0.5) * scale_w - 0.5
        idx = <int>(src + 0.5)
        if idx < 0:
            idx = 0
        elif idx >= in_w:
            idx = in_w - 1
        w_indices[ow] = idx

    # ── Check for contiguous width pattern (stride-1) → memcpy path ────
    w_base = w_indices[0]
    w_contiguous = (out_w <= 1)
    if out_w > 1:
        w_contiguous = True
        for ow in range(1, out_w):
            if w_indices[ow] != w_base + ow:
                w_contiguous = False
                break

    # ── Main resampling loop ────────────────────────────────────────────
    if w_contiguous:
        # Contiguous source → memcpy per row (identity / upscale in width)
        row_bytes = <size_t>out_w * sizeof(numeric_type)
        for od in prange(out_d, schedule='static'):
            out_offset = od * out_h_w
            src_d = data_ptr + d_offsets[od]
            for oh in range(out_h):
                memcpy(
                    output_ptr + out_offset + oh * out_w,
                    src_d + h_offsets[oh] + w_base,
                    row_bytes,
                )
    else:
        # General gather with software prefetching.
        # Prefetch the next source row while processing the current one
        # to hide DRAM latency on large volumes that exceed LLC.
        for od in prange(out_d, schedule='static'):
            out_offset = od * out_h_w
            src_d = data_ptr + d_offsets[od]
            for oh in range(out_h):
                if oh + 1 < out_h:
                    _vol_prefetch_r(src_d + h_offsets[oh + 1])
                src_row = src_d + h_offsets[oh]
                dst_row = output_ptr + out_offset + oh * out_w
                for ow in range(out_w):
                    dst_row[ow] = src_row[w_indices[ow]]

    free(d_offsets)
