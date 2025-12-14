#!/usr/bin/env python3
"""
Multi-architecture Cython build script.

Builds multiple versions of the extension optimized for different CPU architectures:
- Generic (no special instructions, ARM compatible)
- AVX2 (Intel/AMD with AVX2)
- AVX512 (Intel with AVX-512)

Usage:
    python setup_cython_multiarch.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform
import os
import sys

def get_extensions():
    """Create extension modules for different architectures."""
    
    extensions = []
    base_name = "resampling_cython"
    source_file = "src/mimage/backends/resampling/resampling_cython.pyx"
    
    # Common settings
    include_dirs = [np.get_include()]
    
    # Check environment variable for cross-compilation
    build_arch = os.environ.get('BUILD_ARCH', platform.machine().lower())
    
    # Detect target architecture
    is_arm = build_arch in ['arm64', 'aarch64', 'armv7l', 'armv8']
    is_x86 = build_arch in ['x86_64', 'amd64', 'i386', 'i686']
    
    # 1. Generic version (works everywhere, including ARM)
    print("Building generic version (portable)...")
    extensions.append(Extension(
        name=f"{base_name}_generic",
        sources=[source_file],
        include_dirs=include_dirs,
        extra_compile_args=[
            '-O3',
            '-fopenmp',
        ],
        extra_link_args=['-fopenmp'],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    ))
    
    if is_x86:
        # 2. AVX2 version (Intel/AMD from ~2013+)
        print("Building AVX2 version...")
        extensions.append(Extension(
            name=f"{base_name}_avx2",
            sources=[source_file],
            include_dirs=include_dirs,
            extra_compile_args=[
                '-O3',
                '-mavx2',
                '-mfma',
                '-fopenmp',
            ],
            extra_link_args=['-fopenmp'],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        ))
        
        # 3. AVX512 version (Intel Skylake-X and newer)
        print("Building AVX512 version...")
        extensions.append(Extension(
            name=f"{base_name}_avx512",
            sources=[source_file],
            include_dirs=include_dirs,
            extra_compile_args=[
                '-O3',
                '-mavx512f',
                '-mavx512cd',
                '-mavx512vl',
                '-mavx512dq',
                '-mavx512bw',
                '-mfma',
                '-fopenmp',
            ],
            extra_link_args=['-fopenmp'],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        ))
    
    if is_arm:
        # ARM-specific optimizations
        print("Building ARM optimized version...")
        extensions.append(Extension(
            name=f"{base_name}_arm",
            sources=[source_file],
            include_dirs=include_dirs,
            extra_compile_args=[
                '-O3',
                '-mcpu=native',  # Use native CPU features
                '-fopenmp',
            ],
            extra_link_args=['-fopenmp'],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        ))
    
    return extensions


if __name__ == "__main__":
    extensions = get_extensions()
    
    # Print build configuration
    print("\n" + "="*60)
    print("Building Cython extensions for multi-architecture support")
    print("="*60)
    print(f"Target architecture: {os.environ.get('BUILD_ARCH', platform.machine())}")
    print(f"Building {len(extensions)} extension(s):")
    for ext in extensions:
        print(f"  - {ext.name}")
    print("="*60 + "\n")
    
    # Build each extension separately to avoid Cython sorting issue
    for ext in extensions:
        print(f"Building {ext.name}...")
        setup(
            name="resampling_multiarch",
            ext_modules=cythonize(
                [ext],  # Single extension
                compiler_directives={
                    'language_level': "3",
                    'boundscheck': False,
                    'wraparound': False,
                    'cdivision': True,
                    'initializedcheck': False,
                },
            ),
            script_args=sys.argv[1:],  # Pass through command-line args
        )
        print(f"✓ {ext.name} built successfully\n")
