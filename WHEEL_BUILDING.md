# Building and Distributing Wheels with cibuildwheel

This document explains how to build and distribute `mimage` wheels for multiple platforms and architectures.

## Overview

The project uses **cibuildwheel** to automatically build wheels for:

### Platforms
- **Linux**: manylinux2014 (compatible with most Linux distros)
- **macOS**: 10.9+ (Intel x86_64 and Apple Silicon ARM64)
- **Windows**: 64-bit (AMD64)

### Architectures
- **x86_64** (Intel/AMD): Generic, AVX2, and AVX512 optimized versions
- **ARM64** (aarch64): Generic and ARM-optimized versions

### Python Versions
- Python 3.8, 3.9, 3.10, 3.11, 3.12, 3.13

## Quick Start

### 1. Local Development Build

For local development on your current platform:

```bash
# Install build dependencies
pip install cython numpy setuptools wheel

# Build in-place for development
python setup.py build_ext --inplace

# Or install in editable mode
pip install -e .
```

### 2. Build Wheels Locally

To build wheels for your current platform:

```bash
# Install cibuildwheel
pip install cibuildwheel

# Build wheels
python -m cibuildwheel --output-dir wheelhouse

# Wheels will be in ./wheelhouse/
ls wheelhouse/
```

### 3. Build Wheels with GitHub Actions (Recommended)

The easiest way is to use GitHub Actions (see `.github/workflows/build_wheels.yml`):

1. **Push to main/master branch**: Builds wheels for testing
2. **Push a git tag** (`v1.0.0`): Builds and uploads to PyPI

```bash
# Create and push a release
git tag v0.1.0
git push origin v0.1.0

# GitHub Actions will automatically:
# 1. Build wheels for all platforms
# 2. Run tests
# 3. Upload to PyPI (if PYPI_API_TOKEN is set)
```

## Configuration

### pyproject.toml

The `[tool.cibuildwheel]` section controls the build:

```toml
[tool.cibuildwheel]
# Which Python versions to build
build = ["cp38-*", "cp39-*", "cp310-*", "cp311-*", "cp312-*", "cp313-*"]

# Skip certain builds
skip = ["pp*", "*-musllinux*", "*-win32", "*-manylinux_i686"]

# Test the wheels
test-requires = ["pytest", "numpy"]
test-command = "pytest {project}/tests -v"
```

### setup.py

The `setup.py` automatically detects the target architecture and builds appropriate extensions:

- **On x86_64**: Builds generic, AVX2, and AVX512 versions
- **On ARM64**: Builds generic and ARM-optimized versions
- **On Windows**: Builds generic version (with MSVC)

## Architecture-Specific Builds

### How it Works

1. **cibuildwheel** runs on native runners for each platform
2. **setup.py** detects `platform.machine()` and `CIBW_ARCHS` env var
3. **Appropriate compiler flags** are added based on architecture
4. **Multiple `.so` files** are built and packaged in the wheel

### What Gets Built

#### Linux x86_64
```
mimage-0.1.0-cp311-cp311-linux_x86_64.whl
├── resampling_cython.cpython-311-x86_64-linux-gnu.so         (generic)
├── resampling_cython_avx2.cpython-311-x86_64-linux-gnu.so    (AVX2)
└── resampling_cython_avx512.cpython-311-x86_64-linux-gnu.so  (AVX512)
```

#### Linux ARM64
```
mimage-0.1.0-cp311-cp311-linux_aarch64.whl
├── resampling_cython.cpython-311-aarch64-linux-gnu.so        (generic)
└── resampling_cython_arm.cpython-311-aarch64-linux-gnu.so    (ARM opt)
```

#### macOS Universal2
```
mimage-0.1.0-cp311-cp311-macosx_10_9_universal2.whl
├── resampling_cython.cpython-311-darwin.so                   (universal2)
├── resampling_cython_avx2.cpython-311-darwin.so             (x86_64 only)
└── resampling_cython_arm.cpython-311-darwin.so              (arm64 only)
```

## Advanced Usage

### Cross-Compilation for ARM on x86

```bash
# On Linux x86_64, build ARM wheels using QEMU
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
python -m cibuildwheel --platform linux --archs aarch64
```

### Build Only Specific Python Versions

```bash
# Build only Python 3.11
CIBW_BUILD="cp311-*" python -m cibuildwheel
```

### Build for Specific Platform

```bash
# Linux only
python -m cibuildwheel --platform linux

# macOS only
python -m cibuildwheel --platform macos

# Windows only
python -m cibuildwheel --platform windows
```

### Skip Tests

```bash
# Skip tests during wheel building (faster)
CIBW_TEST_SKIP="*" python -m cibuildwheel
```

## Testing Wheels

### Test Installation

```bash
# Install the wheel
pip install wheelhouse/mimage-0.1.0-cp311-cp311-linux_x86_64.whl

# Test import
python -c "import mimage; print('✓ Success')"

# Test backend loading
python -c "from mimage.backends.resampling.cpu_detection import load_best_backend; backend = load_best_backend(); print(f'Loaded: {backend.__name__}')"
```

### Run Tests

```bash
# Install test dependencies
pip install pytest numpy torch

# Run full test suite
pytest tests/ -v
```

### Benchmark

```bash
# Run performance benchmark
OMP_NUM_THREADS=1 python benchmark_resampling.py
```

## Publishing to PyPI

### Setup

1. Create account on [PyPI](https://pypi.org/)
2. Generate an API token
3. Add token to GitHub repository secrets as `PYPI_API_TOKEN`

### Manual Upload

```bash
# Install twine
pip install twine

# Upload to TestPyPI first (recommended)
twine upload --repository testpypi wheelhouse/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ mimage

# Upload to production PyPI
twine upload wheelhouse/*
```

### Automatic Upload via GitHub Actions

Simply push a version tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

GitHub Actions will automatically build and upload to PyPI.

## Troubleshooting

### Build Fails on Windows

Windows uses MSVC which has different flags. The setup.py handles this, but OpenMP might need special treatment.

**Solution**: Skip OpenMP on Windows or install Intel OpenMP:
```bash
pip install intel-openmp
```

### ARM Build is Slow

ARM builds on x86 use QEMU emulation, which is slow.

**Solutions**:
- Use GitHub Actions (builds on native ARM runners)
- Use cloud ARM instances (AWS Graviton, Oracle ARM)
- Skip ARM builds locally: `CIBW_ARCHS_LINUX="x86_64"`

### Import Error After Installation

If you get "module not found" errors:

```bash
# Check wheel contents
unzip -l wheelhouse/*.whl

# Check Python can find the module
python -c "import sys; print(sys.path)"
python -c "import mimage; print(mimage.__file__)"
```

### Tests Fail with "No backend available"

The Cython extension didn't build properly.

```bash
# Check if extensions are present
python -c "from mimage.backends.resampling import resampling_cython; print('✓ Extension loaded')"

# Rebuild with verbose output
python setup.py build_ext --inplace --verbose
```

## Performance Verification

After building, verify the performance improvements:

```bash
# Should show 3-10x speedup depending on CPU
OMP_NUM_THREADS=1 python benchmark_resampling.py
```

Expected results:
- **NEAREST**: 4-5x speedup
- **LINEAR**: 3-4x speedup  
- **AREA**: 8-10x speedup

## File Structure

```
.
├── setup.py                    # Main build script (architecture detection)
├── pyproject.toml             # Package metadata + cibuildwheel config
├── .github/
│   └── workflows/
│       └── build_wheels.yml   # GitHub Actions for automated builds
├── src/
│   └── mimage/
│       └── backends/
│           └── resampling/
│               ├── resampling_cython.pyx          # Source
│               ├── cpu_detection.py               # Runtime selection
│               └── resampling_cython_wrapper.py   # Python interface
└── wheelhouse/                # Built wheels go here
```

## Further Reading

- [cibuildwheel documentation](https://cibuildwheel.readthedocs.io/)
- [Python Packaging Guide](https://packaging.python.org/)
- [manylinux compatibility](https://github.com/pypa/manylinux)
- [Cython documentation](https://cython.readthedocs.io/)
