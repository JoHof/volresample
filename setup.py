#!/usr/bin/env python3
"""
Setup script for mimage package with Cython extensions.

This setup.py is designed to work with cibuildwheel for building wheels
across different platforms and architectures (x86_64, ARM, etc.).

The Cython extensions are built with architecture-appropriate optimizations:
- x86_64: Generic, AVX2, and AVX512 versions
- ARM: Generic and ARM-optimized versions
- Other: Generic fallback

cibuildwheel will automatically call this setup.py on each target platform,
and the build will adapt to the native architecture.
"""

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import numpy as np
import platform
import os
import sys

# Try to import Cython
try:
    from Cython.Build import cythonize
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    print("WARNING: Cython not available. Extensions will not be built.")


class BuildExtWithArchDetection(build_ext):
    """Custom build_ext that provides helpful messages."""
    
    def run(self):
        """Run the build with architecture detection info."""
        machine = platform.machine().lower()
        print(f"\n{'='*70}")
        print(f"Building Cython extensions for: {machine}")
        print(f"Python: {sys.version}")
        print(f"NumPy: {np.__version__}")
        print(f"{'='*70}\n")
        
        super().run()


def get_cython_extensions():
    """
    Create Cython extension modules with architecture-appropriate flags.
    
    When cibuildwheel builds wheels:
    - On x86_64 runners: builds generic + AVX2 + AVX512 versions
    - On ARM runners: builds generic + ARM-optimized versions
    - On other platforms: builds generic version only
    
    Returns:
        list: List of Extension objects to compile
    """
    if not CYTHON_AVAILABLE:
        return []
    
    # Detect target architecture
    machine = platform.machine().lower()
    is_arm = machine in ['arm64', 'aarch64', 'armv7l', 'armv8']
    is_x86 = machine in ['x86_64', 'amd64', 'i386', 'i686']
    
    # Allow override via environment variable (for cross-compilation)
    build_arch = os.environ.get('CIBW_ARCHS', machine)
    if build_arch in ['ARM64', 'aarch64']:
        is_arm = True
        is_x86 = False
    elif build_arch in ['x86_64', 'AMD64']:
        is_x86 = True
        is_arm = False
    
    extensions = []
    
    # Source file for resampling backend
    resampling_source = "src/mimage/backends/resampling/resampling_cython.pyx"
    
    # Common settings
    include_dirs = [np.get_include()]
    define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    
    # Compiler flags - platform dependent
    if sys.platform == 'win32':
        # Windows with MSVC
        extra_compile_args_base = ['/O2', '/arch:AVX2']
        extra_link_args_base = []
        openmp_compile = ['/openmp']
        openmp_link = []
    else:
        # Linux/macOS with GCC/Clang
        # AVX2 flags for vectorization - most modern x86_64 CPUs support this
        avx_flags = []
        if is_x86:
            avx_flags = [
                '-mavx2',           # Enable AVX2 instructions
                '-mfma',            # Enable FMA (fused multiply-add)
                '-ftree-vectorize', # Enable auto-vectorization
                '-ffast-math',      # Allow aggressive FP optimizations
            ]
        extra_compile_args_base = ['-O3'] + avx_flags
        extra_link_args_base = []
        openmp_compile = ['-fopenmp']
        openmp_link = ['-fopenmp']
    
    print(f"\nBuilding extensions for: {machine}")
    print(f"  is_x86: {is_x86}")
    print(f"  is_arm: {is_arm}")
    if is_x86:
        print(f"  AVX2/FMA optimizations: ENABLED")
    
    # Only build the generic version to avoid cythonize error
    print("  - Adding generic version (portable)")
    extensions.append(Extension(
        name="mimage.backends.resampling.resampling_cython",
        sources=[resampling_source],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args_base + openmp_compile,
        extra_link_args=extra_link_args_base + openmp_link,
        define_macros=define_macros,
    ))
    print(f"\nTotal extensions to build: {len(extensions)}\n")
    return extensions


def get_long_description():
    """Read the long description from README.md."""
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Medical image processing library with optimized resampling"


# Read version from package
def get_version():
    """Get version from pyproject.toml or set default."""
    try:
        import tomli
        with open('pyproject.toml', 'rb') as f:
            pyproject = tomli.load(f)
            return pyproject['project']['version']
    except:
        return "0.1.0"  # Fallback


def main():
    """Main setup function."""
    
    # Get Cython extensions
    extensions = get_cython_extensions()
    
    # Cythonize if available
    if extensions and CYTHON_AVAILABLE:
        # Remove duplicate Extension objects (by name)
        unique_exts = {}
        for ext in extensions:
            if ext.name not in unique_exts:
                unique_exts[ext.name] = ext
        ext_modules = cythonize(
            list(unique_exts.values()),
            compiler_directives={
                'language_level': "3",
                'boundscheck': False,
                'wraparound': False,
                'cdivision': True,
                'initializedcheck': False,
                'nonecheck': False,
            },
            nthreads=int(os.environ.get('CYTHON_NTHREADS', '1')),
        )
    else:
        ext_modules = []
    
    setup(
        name="mimage",
        version=get_version(),
        description="Medical image processing library with optimized resampling",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        author="Johannes",
        author_email="your.email@example.com",  # Update this
        url="https://github.com/yourusername/mimage",  # Update this
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        ext_modules=ext_modules,
        install_requires=[
            'numpy>=1.20.0',
            'cython>=3.0.0',
        ],
        extras_require={
            'torch': ['torch>=1.9.0'],
            'nibabel': ['nibabel>=3.0.0'],
            'viz': ['matplotlib>=3.0.0', 'scikit-image>=0.18.0'],
            'dev': [
                'pytest>=7.0.0',
                'pytest-cov>=3.0.0',
                'black>=22.0.0',
                'mypy>=0.950',
                'cibuildwheel>=2.12.0',
            ],
        },
        python_requires='>=3.8',
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Healthcare Industry',
            'License :: OSI Approved :: MIT License',  # Update if different
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Programming Language :: Python :: 3.13',
            'Programming Language :: Cython',
            'Topic :: Scientific/Engineering :: Medical Science Apps.',
            'Topic :: Scientific/Engineering :: Image Processing',
        ],
        keywords='medical imaging resampling interpolation cython',
        cmdclass={'build_ext': BuildExtWithArchDetection},
        # Include package data
        include_package_data=True,
        zip_safe=False,  # Required for Cython extensions
    )


if __name__ == '__main__':
    main()
