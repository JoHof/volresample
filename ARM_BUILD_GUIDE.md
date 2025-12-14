# Cross-Compilation Guide for ARM

## Current Status

Your x86_64 machine has built:
- ✅ `resampling_cython_generic` (x86_64)
- ✅ `resampling_cython_avx2` (x86_64 with AVX2)
- ✅ `resampling_cython_avx512` (x86_64 with AVX512)

**NOT built:**
- ❌ ARM versions (requires ARM hardware or cross-compilation)

## Why?

The build script detects your current architecture (`platform.machine() == 'x86_64'`) and only builds binaries compatible with it. This is the correct behavior - you can't run ARM binaries on x86_64 without emulation.

## How to Build ARM Versions

### Method 1: On Real ARM Hardware (Recommended)

**On Raspberry Pi or ARM server:**
```bash
# Install dependencies
sudo apt-get install python3-dev gcc

# Clone your project
git clone <your-repo>
cd mimage

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install numpy cython setuptools

# Build
python setup_cython_multiarch.py build_ext --inplace
```

This will create:
- `resampling_cython_generic.cpython-3XX-aarch64-linux-gnu.so`
- `resampling_cython_arm.cpython-3XX-aarch64-linux-gnu.so`

### Method 2: Cross-Compile with Docker + QEMU (Advanced)

**On your x86_64 machine:**

1. **Set up QEMU for ARM emulation:**
```bash
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

2. **Build in ARM Docker container:**
```bash
docker run --rm --platform linux/arm64 \
  -v $(pwd):/work -w /work \
  python:3.13-slim bash -c "
    # Install build tools
    apt-get update
    apt-get install -y gcc g++ make
    
    # Install Python packages
    pip install numpy cython setuptools
    
    # Build ARM version
    BUILD_ARCH=aarch64 python setup_cython_multiarch.py build_ext --inplace
  "
```

This creates ARM binaries on your x86_64 machine (slowly, via emulation).

### Method 3: GitHub Actions CI/CD (Production)

**Set up GitHub Actions to build on real ARM runners:**

```yaml
# .github/workflows/build-wheels.yml
name: Build Wheels

on: [push, pull_request]

jobs:
  build-linux-x86:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Build x86_64 extensions
        run: |
          pip install numpy cython setuptools
          python setup_cython_multiarch.py build_ext --inplace
      
      - uses: actions/upload-artifact@v3
        with:
          name: linux-x86_64
          path: src/**/*.so
  
  build-linux-arm:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up QEMU
        uses: docker/setup-qemu-action@v2
        with:
          platforms: arm64
      
      - name: Build ARM64 extensions
        run: |
          docker run --rm --platform linux/arm64 \
            -v $PWD:/work -w /work \
            python:3.10 bash -c "
              pip install numpy cython setuptools
              BUILD_ARCH=aarch64 python setup_cython_multiarch.py build_ext --inplace
            "
      
      - uses: actions/upload-artifact@v3
        with:
          name: linux-aarch64
          path: src/**/*.so
```

## For Package Distribution

When building wheels for PyPI, use separate builds:

```bash
# Build x86_64 wheel
python setup.py bdist_wheel
# → mimage-1.0.0-cp310-cp310-linux_x86_64.whl

# Build ARM wheel (on ARM hardware or via Docker)
python setup.py bdist_wheel  
# → mimage-1.0.0-cp310-cp310-linux_aarch64.whl
```

Upload both to PyPI, and users will automatically get the right one:
```bash
# On x86_64
pip install mimage  # Gets x86_64 wheel

# On ARM (Raspberry Pi)
pip install mimage  # Gets aarch64 wheel
```

## Summary

✅ **Your current setup is correct** - you have optimized x86_64 binaries
❌ **ARM binaries require ARM hardware** or cross-compilation
🚀 **For production**, use CI/CD to build on multiple architectures
📦 **For distribution**, upload separate wheels per platform to PyPI

The system is designed so that:
1. Developers build for their own platform (simple)
2. CI/CD builds for all platforms (automated)
3. Users get the right binary (automatic via pip)
