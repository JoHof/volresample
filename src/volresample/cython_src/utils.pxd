# cython: language_level=3
"""PXD header for cython_src/utils.pyx"""

cdef extern from "omp.h":
    int omp_set_num_threads(int)

cdef inline float clip(float val, float min_val, float max_val) nogil
