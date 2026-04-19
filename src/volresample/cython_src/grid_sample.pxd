# cython: language_level=3
"""PXD header for cython_src/grid_sample.pyx"""

from numpy cimport uint8_t, int16_t

ctypedef fused grid_numeric_type:
    uint8_t
    int16_t
    float

cdef inline float unnormalize_coord(float coord, int size) noexcept nogil
cdef inline float reflect_coord(float coord, float min_val, float max_val) noexcept nogil
cdef inline int reflect_bound(int idx, int size) noexcept nogil

# 2D functions (4D input)
cdef void _grid_sample_2d_bilinear_zeros(
    float* input_ptr, float* grid_ptr, float* output_ptr,
    int N, int C, int H_in, int W_in, int H_out, int W_out
) noexcept nogil

cdef void _grid_sample_2d_bilinear_border(
    float* input_ptr, float* grid_ptr, float* output_ptr,
    int N, int C, int H_in, int W_in, int H_out, int W_out
) noexcept nogil

cdef void _grid_sample_2d_bilinear_reflection(
    float* input_ptr, float* grid_ptr, float* output_ptr,
    int N, int C, int H_in, int W_in, int H_out, int W_out
) noexcept nogil

cdef void _grid_sample_2d_nearest_zeros(
    float* input_ptr, float* grid_ptr, float* output_ptr,
    int N, int C, int H_in, int W_in, int H_out, int W_out
) noexcept nogil

cdef void _grid_sample_2d_nearest_border(
    float* input_ptr, float* grid_ptr, float* output_ptr,
    int N, int C, int H_in, int W_in, int H_out, int W_out
) noexcept nogil

cdef void _grid_sample_2d_nearest_reflection(
    float* input_ptr, float* grid_ptr, float* output_ptr,
    int N, int C, int H_in, int W_in, int H_out, int W_out
) noexcept nogil

# 3D functions (5D input)
cdef void _grid_sample_bilinear_zeros(
    float* input_ptr, float* grid_ptr, float* output_ptr,
    int N, int C, int D_in, int H_in, int W_in, int D_out, int H_out, int W_out
) noexcept nogil

cdef void _grid_sample_bilinear_border(
    float* input_ptr, float* grid_ptr, float* output_ptr,
    int N, int C, int D_in, int H_in, int W_in, int D_out, int H_out, int W_out
) noexcept nogil

cdef void _grid_sample_bilinear_reflection(
    float* input_ptr, float* grid_ptr, float* output_ptr,
    int N, int C, int D_in, int H_in, int W_in, int D_out, int H_out, int W_out
) noexcept nogil

cdef void _grid_sample_nearest_zeros(
    grid_numeric_type* input_ptr, float* grid_ptr, grid_numeric_type* output_ptr,
    int N, int C, int D_in, int H_in, int W_in, int D_out, int H_out, int W_out
) noexcept nogil

cdef void _grid_sample_nearest_border(
    grid_numeric_type* input_ptr, float* grid_ptr, grid_numeric_type* output_ptr,
    int N, int C, int D_in, int H_in, int W_in, int D_out, int H_out, int W_out
) noexcept nogil

cdef void _grid_sample_nearest_reflection(
    grid_numeric_type* input_ptr, float* grid_ptr, grid_numeric_type* output_ptr,
    int N, int C, int D_in, int H_in, int W_in, int D_out, int H_out, int W_out
) noexcept nogil
