# Linear (trilinear) resampling implementation
# Included via main file, uses definitions from utils.pyx

# =============================================================================


# =============================================================================
# Trilinear resampling with pre-computed weights and indices
# =============================================================================
cdef void _resample_linear(
    float* data_ptr,
    float* output_ptr,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    float scale_d, float scale_h, float scale_w
) noexcept nogil:
    """
    Trilinear resampling for arbitrary scale factors.
    Pre-computes all weights and indices before the main loop.
    """
    cdef int in_hw = in_h * in_w
    cdef int out_hw = out_h * out_w
    
    cdef int d_max = in_d - 1
    cdef int h_max = in_h - 1
    cdef int w_max = in_w - 1
    
    # Pre-allocate arrays for weights and indices
    cdef int* d0_arr = <int*>malloc(out_d * sizeof(int))
    cdef int* d1_arr = <int*>malloc(out_d * sizeof(int))
    cdef float* fd_arr = <float*>malloc(out_d * sizeof(float))
    cdef float* fd1_arr = <float*>malloc(out_d * sizeof(float))
    
    cdef int* h0_arr = <int*>malloc(out_h * sizeof(int))
    cdef int* h1_arr = <int*>malloc(out_h * sizeof(int))
    cdef float* fh_arr = <float*>malloc(out_h * sizeof(float))
    cdef float* fh1_arr = <float*>malloc(out_h * sizeof(float))
    
    cdef int* w0_arr = <int*>malloc(out_w * sizeof(int))
    cdef int* w1_arr = <int*>malloc(out_w * sizeof(int))
    cdef float* fw_arr = <float*>malloc(out_w * sizeof(float))
    cdef float* fw1_arr = <float*>malloc(out_w * sizeof(float))
    
    cdef int i
    cdef float src, src_floor
    cdef int idx0, idx1
    
    # Pre-compute depth weights and indices
    for i in range(out_d):
        src = (i + 0.5) * scale_d - 0.5
        src_floor = floor(src)
        idx0 = <int>src_floor
        idx1 = idx0 + 1
        
        if idx0 < 0: idx0 = 0
        elif idx0 > d_max: idx0 = d_max
        if idx1 < 0: idx1 = 0
        elif idx1 > d_max: idx1 = d_max
        
        d0_arr[i] = idx0
        d1_arr[i] = idx1
        fd_arr[i] = src - src_floor
        fd1_arr[i] = 1.0 - fd_arr[i]
    
    # Pre-compute height weights and indices
    for i in range(out_h):
        src = (i + 0.5) * scale_h - 0.5
        src_floor = floor(src)
        idx0 = <int>src_floor
        idx1 = idx0 + 1
        
        if idx0 < 0: idx0 = 0
        elif idx0 > h_max: idx0 = h_max
        if idx1 < 0: idx1 = 0
        elif idx1 > h_max: idx1 = h_max
        
        h0_arr[i] = idx0
        h1_arr[i] = idx1
        fh_arr[i] = src - src_floor
        fh1_arr[i] = 1.0 - fh_arr[i]
    
    # Pre-compute width weights and indices
    for i in range(out_w):
        src = (i + 0.5) * scale_w - 0.5
        src_floor = floor(src)
        idx0 = <int>src_floor
        idx1 = idx0 + 1
        
        if idx0 < 0: idx0 = 0
        elif idx0 > w_max: idx0 = w_max
        if idx1 < 0: idx1 = 0
        elif idx1 > w_max: idx1 = w_max
        
        w0_arr[i] = idx0
        w1_arr[i] = idx1
        fw_arr[i] = src - src_floor
        fw1_arr[i] = 1.0 - fw_arr[i]
    
    # Main interpolation loop using pre-computed values
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
    
    # Free allocated memory
    free(d0_arr)
    free(d1_arr)
    free(fd_arr)
    free(fd1_arr)
    free(h0_arr)
    free(h1_arr)
    free(fh_arr)
    free(fh1_arr)
    free(w0_arr)
    free(w1_arr)
    free(fw_arr)
    free(fw1_arr)
