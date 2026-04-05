# cython: language_level=3
"""PXD header for cython_src/cubic.pyx — cubic B-spline resampling."""

cdef void _resample_cubic(
    float* data_ptr,
    float* output_ptr,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    float scale_d, float scale_h, float scale_w,
    int num_threads,
    bint align_corners
) noexcept nogil
