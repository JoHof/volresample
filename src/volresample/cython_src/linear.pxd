# cython: language_level=3
"""PXD header for cython_src/linear.pyx"""

cdef inline void _precompute_linear_coords(
    int out_n, int in_n, float scale, bint align_corners,
    int* idx0_arr, int* idx1_arr, float* f_arr, float* f1_arr
) noexcept nogil

cdef void _resample_linear(
    float* data_ptr,
    float* output_ptr,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    float scale_d, float scale_h, float scale_w,
    bint align_corners
) noexcept nogil

cdef void _resample_linear_multi(
    float* data_ptr,
    float* output_ptr,
    int n_channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    float scale_d, float scale_h, float scale_w,
    bint align_corners
) noexcept nogil
