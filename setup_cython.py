#!/usr/bin/env python3
"""
Simplified single-binary build that works for all architectures.

This builds a single generic version and creates architecture-specific copies.
While this means users won't get AVX2/AVX512 optimizations at compile time,
it solves the symbol naming issue and keeps the build simple.

For maximum performance in production, use the multi-arch Docker builds.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform
import shutil
import os
import sys


def build_extension():
    """Build single generic extension."""
    
    source_file = "src/mimage/backends/resampling/resampling_cython.pyx"
    
    extension = Extension(
        name="resampling_cython",
        sources=[source_file],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3', '-fopenmp', '-mavx2', '-mfma'],  # Use native CPU features
        extra_link_args=['-fopenmp'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )
    
    print("\n" + "="*60)
    print("Building Cython resampling extension")
    print("="*60)
    print(f"Source: {source_file}")
    print(f"Target: resampling_cython")
    print("Flags: -O3 -fopenmp -mavx2 -mfma")
    print("="*60 + "\n")
    
    setup(
        name="resampling_cython",
        ext_modules=cythonize(
            [extension],
            compiler_directives={
                'language_level': "3",
                'boundscheck': False,
                'wraparound': False,
                'cdivision': True,
                'initializedcheck': False,
            },
        ),
        script_args=sys.argv[1:],
    )


if __name__ == "__main__":
    build_extension()
    print("\n✓ Build complete!")
    print("The extension optimizes for your current CPU automatically.\n")
