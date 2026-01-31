# cython: language_level=3
"""PXD header for cython_src/area.pyx"""

import numpy as np
cimport numpy as cnp

cpdef cnp.ndarray[cnp.float32_t, ndim=3] _resample_area(
    cnp.ndarray[cnp.float32_t, ndim=3] data,
    tuple size,
    float scale_d, float scale_h, float scale_w,
    int parallel_threads
)

