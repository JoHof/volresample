from libc.stdlib cimport malloc, free
from libc.math cimport floor, ceil
from numpy cimport uint8_t, int16_t

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
    cdef float src_d, src_h, src_w
    cdef int d_idx, h_idx, w_idx, idx_in, idx_out
    cdef int in_h_w = in_h * in_w
    
    # Precompute height indices
    cdef int* h_indices = <int*>malloc(out_h * sizeof(int))
    cdef int* w_indices = <int*>malloc(out_w * sizeof(int))
    
    try:
        # Precompute height indices
        for oh in range(out_h):
            src_h = (oh + 0.5) * scale_h - 0.5
            h_idx = <int>(src_h + 0.5)
            if h_idx < 0:
                h_idx = 0
            elif h_idx >= in_h:
                h_idx = in_h - 1
            h_indices[oh] = h_idx
        
        # Precompute width indices
        for ow in range(out_w):
            src_w = (ow + 0.5) * scale_w - 0.5
            w_idx = <int>(src_w + 0.5)
            if w_idx < 0:
                w_idx = 0
            elif w_idx >= in_w:
                w_idx = in_w - 1
            w_indices[ow] = w_idx
        
        # Main resampling loop
        for od in prange(out_d, schedule='static'):
            src_d = (od + 0.5) * scale_d - 0.5
            d_idx = <int>(src_d + 0.5)
            if d_idx < 0:
                d_idx = 0
            elif d_idx >= in_d:
                d_idx = in_d - 1
            
            for oh in range(out_h):
                for ow in range(out_w):
                    idx_in = d_idx * in_h_w + h_indices[oh] * in_w + w_indices[ow]
                    idx_out = od * out_h * out_w + oh * out_w + ow
                    output_ptr[idx_out] = data_ptr[idx_in]
    finally:
        free(h_indices)
        free(w_indices)
