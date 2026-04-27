# cython: language_level=3
"""PXD header for cython_src/nearest.pyx"""


cpdef void _resample_nearest(
    float* data_ptr,
    float* output_ptr,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    float scale_d, float scale_h, float scale_w,
    bint align_corners
) noexcept nogil
