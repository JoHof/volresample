"""Setup script for building Cython extension."""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "resampling_cython",
        ["resampling_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[
            "-O3",           # Maximum optimization
            "-march=native", # Use all CPU features available
            "-mavx",         # Enable AVX
            "-mavx2",        # Enable AVX2
            "-mfma",         # Enable FMA (fused multiply-add)
            "-fopenmp",      # Enable OpenMP for parallel
        ],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="resampling_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
        }
    ),
)
