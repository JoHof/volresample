# Architecture

This document describes the code organization and design decisions in volresample.

## Overview

volresample is a Cython-based 3D volume resampling library with OpenMP parallelization. It provides two main operations:

- `resample()` - Resize volumes to new dimensions (like PyTorch's `F.interpolate`). Supports 3D, 4D (multi-channel), and 5D (batched multi-channel) tensors.
- `grid_sample()` - Sample at arbitrary locations (like PyTorch's `F.grid_sample`). Supports 5D tensors for 3D volumes.

## Project Structure

```
src/volresample/
├── __init__.py          # Public API: resample, grid_sample, set_num_threads, get_num_threads
├── _config.py           # Global thread configuration (pure Python)
├── _resample.pyx        # Main Cython entry point
├── _resample.pyi        # Type stubs for IDE support
└── cython_src/          # Implementation modules (included at compile time)
    ├── utils.pyx        # Shared utilities (clip function)
    ├── nearest.pyx      # Nearest neighbor interpolation
    ├── linear.pyx       # Trilinear interpolation
    ├── area.pyx         # Area-based resampling
    ├── cubic.pyx        # Cubic B-spline interpolation with IIR prefilter
    └── grid_sample.pyx  # Grid sampling implementation
```

## Compilation Model

All `.pyx` files in `cython_src/` are **included** (not imported) into `_resample.pyx`:

```cython
include "cython_src/utils.pyx"
include "cython_src/nearest.pyx"
include "cython_src/linear.pyx"
include "cython_src/area.pyx"
include "cython_src/cubic.pyx"
include "cython_src/grid_sample.pyx"
```

This produces a **single compiled extension** (`_resample.cpython-*.so`). Benefits:

- No runtime import overhead
- Compiler can inline and optimize across modules
- Single file deployment

Trade-off: Any change requires full recompilation.

## Thread Configuration

Threads are configured globally via `_config.py`:

```python
volresample.set_num_threads(4)  # Set thread count
volresample.get_num_threads()   # Get current count (default: min(cpu_count, 4))
```

The Cython code reads this value before each operation:

```cython
cdef inline void _apply_thread_settings() noexcept:
    cdef int num_threads = get_num_threads()  # Python call
    omp_set_num_threads(num_threads)          # OpenMP call
```

## Key Design Decisions

### 1. Fused Types for Multi-dtype Support

Nearest neighbor supports `uint8`, `int16`, and `float32` using Cython's fused types:

```cython
ctypedef fused numeric_type:
    uint8_t
    int16_t
    float

cdef void _resample_nearest(numeric_type* data_ptr, ...) noexcept nogil:
    # Single implementation, compiled for each type
```

Linear, area, and cubic modes only support `float32` (interpolation requires floating point).

### 2. Pre-computed Index Tables

To avoid redundant coordinate calculations in inner loops, indices and weights are pre-computed:

```cython
# Pre-compute height indices once
cdef int* h_indices = <int*>malloc(out_h * sizeof(int))
for oh in range(out_h):
    h_indices[oh] = compute_source_index(oh, scale_h)

# Main loop uses pre-computed values
for od in prange(out_d):
    for oh in range(out_h):
        for ow in range(out_w):
            value = data_ptr[d_idx * H * W + h_indices[oh] * W + w_indices[ow]]
```

### 3. GIL Release for Parallelism

All core loops release the GIL with `nogil`:

```cython
with nogil:
    _resample_linear(data_ptr, output_ptr, ...)
```

This allows OpenMP threads to run in parallel without Python overhead.

### 4. C-Contiguous Memory Enforcement

Input arrays are converted to C-contiguous layout before processing:

```cython
data_np = np.ascontiguousarray(data)
```

This ensures predictable memory access patterns and enables the raw pointer arithmetic used throughout.

### 5. 4D/5D as Batch and Channel Iteration

The `resample()` function supports 3D `(D, H, W)`, 4D `(C, D, H, W)`, and 5D `(N, C, D, H, W)` tensors by iterating over batch and channel dimensions:

```cython
# 5D: iterate over batch and channels
if ndim == 5:
    for b in range(n_batch):
        for c in range(n_channels):
            channel_output = _resample_channel(data_np[b, c], size, mode)
            # ... stack results

# 4D: iterate over channels only
elif ndim == 4:
    for c in range(n_channels):
        channel_output = _resample_channel(data_np[c], size, mode)
        # ... stack results
```

This keeps the core resampling functions simple (3D only) while supporting multi-channel and batched data. Each 3D volume is processed independently, allowing for straightforward parallelization within each volume.

### 6. PyTorch Compatibility

The coordinate system matches PyTorch's `align_corners=False` (the default):

```cython
# Source coordinate for output index i
src = (i + 0.5) * scale - 0.5
```

For `grid_sample`, normalized coordinates in `[-1, 1]` map to pixel coordinates:

```cython
# align_corners=False formula
pixel = ((coord + 1) / 2) * size - 0.5
```

### 7. Cubic B-spline Interpolation

The cubic mode implements tricubic B-spline interpolation matching `scipy.ndimage.zoom(order=3, mode='reflect', grid_mode=True)`. Unlike the other modes which directly compute output values, cubic interpolation is a two-stage process:

1. **IIR prefilter**: A separable in-place infinite impulse response filter along each axis converts sample values into B-spline coefficients. This uses causal and anticausal passes with pole `z = sqrt(3) - 2` and reflect (half-sample symmetric) boundary conditions. The initialization formulas match scipy's `_init_causal_reflect` and `_init_anticausal_reflect` from `ni_splines.c`.

2. **Evaluation**: For each output voxel, compute source coordinates, determine the 4×4×4 neighborhood of B-spline coefficients, apply the cubic B-spline basis weights, and accumulate the result.

An identity fast-path skips both stages when the input and output sizes match, returning a copy of the input directly.

## Parallelization Strategy

OpenMP `prange` is used on the outermost spatial dimension:

```cython
for od in prange(out_d, schedule='static', nogil=True):
    for oh in range(out_h):
        for ow in range(out_w):
            # Independent computation for each output voxel
```

- **3D resampling**: Parallelizes over output depth
- **4D/5D resampling**: Each 3D volume (per batch/channel) is processed sequentially in Python, but parallelized over depth within each volume
- **grid_sample**: Parallelizes over batch × depth (flattened)

Each thread writes to independent output locations, so no synchronization is needed.

## Memory Access Patterns

### Row-major (C-order) Indexing

All arrays use C-contiguous layout `(D, H, W)`:

```cython
index = d * H * W + h * W + w
value = data_ptr[index]
```

### Cache-friendly Iteration

The inner loop iterates over the width dimension, which is contiguous in memory:

```cython
for od in prange(out_d):      # Parallel
    for oh in range(out_h):   # Sequential
        for ow in range(out_w):  # Sequential, contiguous access
```

## grid_sample Implementation

`grid_sample` supports 3D (5D tensor: N, C, D, H, W) and 2D (4D tensor: N, C, H, W).

### Coordinate Order

Grid coordinates are `(x, y, z)` order matching PyTorch:

- `grid[..., 0]` = x (width)
- `grid[..., 1]` = y (height)
- `grid[..., 2]` = z (depth)

### Padding Modes

Three padding modes handle out-of-bounds coordinates:

- **zeros**: Return 0 for out-of-bounds samples
- **border**: Clamp to edge values
- **reflection**: Reflect coordinates at boundaries

### Function Variants

Separate functions exist for each mode × padding combination to avoid branching in inner loops:

```
_grid_sample_bilinear_zeros()
_grid_sample_bilinear_border()
_grid_sample_bilinear_reflection()
_grid_sample_nearest_zeros()
_grid_sample_nearest_border()
_grid_sample_nearest_reflection()
```

## Build System

### setup.py

The build detects architecture (x86 vs ARM) and applies appropriate compiler flags:

- **x86**: `-O3 -mavx2 -mfma -fopenmp`
- **ARM**: `-O3 -fopenmp` (no AVX)
- **Windows**: `/O2 /openmp`

### Cython Directives

Performance-oriented directives disable runtime checks:

```cython
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
```

## Testing

Tests compare against PyTorch (nearest, linear, area) and SciPy (cubic) to verify numerical correctness:

```python
# tests/test_resampling.py, tests/test_grid_sample.py
torch_output = F.interpolate(input, size, mode='trilinear')
cython_output = volresample.resample(input, size, mode='linear')
assert np.allclose(torch_output, cython_output, atol=1e-5)

# Cubic tests compare against scipy.ndimage.zoom
scipy_output = scipy.ndimage.zoom(input, zoom, order=3, mode='reflect', grid_mode=True)
cython_output = volresample.resample(input, size, mode='cubic')
assert np.allclose(scipy_output, cython_output, atol=1e-6)
```

The `TorchReference` class in `tests/torch_reference.py` provides a consistent interface for PyTorch operations with mode name mapping (e.g., `linear` → `trilinear`).
