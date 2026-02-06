#!/usr/bin/env python3
"""
Setup script for volresample package with Cython extensions.

This setup.py builds optimized Cython extensions for 3D volume resampling
with OpenMP parallelization and architecture-specific optimizations.
"""

import os
import platform
import sys

import numpy as np
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

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
        print(f"Building volresample Cython extensions for: {machine}")
        print(f"Python: {sys.version}")
        print(f"NumPy: {np.__version__}")
        print(f"{'='*70}\n")

        super().run()


def get_cython_extensions():
    """
    Create Cython extension modules with architecture-appropriate flags.
    """
    if not CYTHON_AVAILABLE:
        return []

    # Detect target architecture
    machine = platform.machine().lower()
    is_arm = machine in ["arm64", "aarch64", "armv7l", "armv8"]
    is_x86 = machine in ["x86_64", "amd64", "i386", "i686"]

    # Allow override via environment variable (for cross-compilation)
    build_arch = os.environ.get("CIBW_ARCHS", machine)
    if build_arch in ["ARM64", "aarch64"]:
        is_arm = True
        is_x86 = False
    elif build_arch in ["x86_64", "AMD64"]:
        is_x86 = True
        is_arm = False

    extensions = []

    # Source file for resampling
    resampling_source = "src/volresample/_resample.pyx"

    # Common settings
    include_dirs = [np.get_include()]
    define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

    # Compiler flags - platform dependent
    if sys.platform == "win32":
        # Windows with MSVC
        extra_compile_args_base = ["/O2", "/arch:AVX2"]
        extra_link_args_base = []
        openmp_compile = ["/openmp"]
        openmp_link = []
    elif sys.platform == "darwin":
        # macOS with Apple Clang — needs special OpenMP handling
        avx_flags = []
        if is_x86:
            avx_flags = [
                "-mavx2",
                "-mfma",
                "-ftree-vectorize",
                "-ffast-math",
            ]
        extra_compile_args_base = ["-O3"] + avx_flags
        extra_link_args_base = []
        # Apple Clang does not support -fopenmp directly; use -Xclang -fopenmp
        openmp_compile = ["-Xclang", "-fopenmp"]
        openmp_link = ["-lomp"]
    else:
        # Linux with GCC/Clang
        avx_flags = []
        if is_x86:
            avx_flags = [
                "-mavx2",
                "-mfma",
                "-ftree-vectorize",
                "-ffast-math",
            ]
        extra_compile_args_base = ["-O3"] + avx_flags
        extra_link_args_base = []
        openmp_compile = ["-fopenmp"]
        openmp_link = ["-fopenmp"]

    print(f"\nBuilding extensions for: {machine}")
    print(f"  is_x86: {is_x86}")
    print(f"  is_arm: {is_arm}")
    if is_x86:
        print("  AVX2/FMA optimizations: ENABLED")

    # Build the volresample extension
    print("  - Building volresample._resample")
    extensions.append(
        Extension(
            name="volresample._resample",
            sources=[resampling_source],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args_base + openmp_compile,
            extra_link_args=extra_link_args_base + openmp_link,
            define_macros=define_macros,
        )
    )
    print(f"\nTotal extensions to build: {len(extensions)}\n")
    return extensions


def get_long_description():
    """Read the long description from README.md."""
    try:
        with open("README.md", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Fast 3D volume resampling with optimized Cython"


# Read version from package
def get_version():
    """Get version from pyproject.toml or set default."""
    try:
        import tomli

        with open("pyproject.toml", "rb") as f:
            pyproject = tomli.load(f)
            return pyproject["project"]["version"]
    except Exception:
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
                "language_level": "3",
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
                "initializedcheck": False,
                "nonecheck": False,
            },
            nthreads=int(os.environ.get("CYTHON_NTHREADS", "1")),
        )
    else:
        ext_modules = []

    setup(
        name="volresample",
        version=get_version(),
        description="Fast 3D volume resampling with optimized Cython",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        author="Johannes",
        author_email="j.hofmanninger@gmail.com",
        url="https://github.com/JoHof/volresample",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        ext_modules=ext_modules,
        # Wheel should only contain runtime artifacts.
        # Cython sources (.pyx/.pxd) and generated .c belong in sdist only.
        # (include-package-data = false in pyproject.toml enforces this)
        package_data={"volresample": ["*.pyi", "py.typed"]},
        install_requires=[
            "numpy>=1.20.0",
        ],
        python_requires=">=3.9",
        # Note: metadata (name, version, classifiers, license, etc.) is
        # authoritative in pyproject.toml [project].  Only build/extension
        # logic belongs here.
        keywords="medical imaging resampling interpolation cython",
        cmdclass={"build_ext": BuildExtWithArchDetection},
        zip_safe=False,  # Required for Cython extensions
    )


if __name__ == "__main__":
    main()
