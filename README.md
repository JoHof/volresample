# volresample

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Fast 3D volume resampling with Cython and OpenMP parallelization.

Implemented against PyTorch's `F.interpolate` and `F.grid_sample` as a reference, producing identical results for nearest, linear, and area modes. The cubic mode matches `scipy.ndimage.zoom(order=3, mode='reflect')`, using `grid_mode=True` when `align_corners=False` and `grid_mode=False` when `align_corners=True`. Can be used as a drop-in replacement when PyTorch or SciPy is not available or when better performance is desired on CPU.

[Blogpost](https://johof.github.io/2026/02/volresample-3d-volume-resampling/)
## Features

- Cython-optimized with OpenMP parallelization
- Simple API: `resample()` and `grid_sample()`
- Interpolation modes: nearest, linear, area, and cubic
- Supports 3D, 4D (multi-channel), and 5D (batched multi-channel) volumes
- Supports `align_corners=True` for nearest, linear, and cubic resampling
- Supports uint8, int16 (nearest) and float32 dtypes (all other modes) for both `resample` and `grid_sample`

## Installation

```bash
pip install volresample
```

Or build from source:

```bash
git clone https://github.com/JoHof/volresample.git
cd volresample
uv sync
```

## Quick Start

### Basic Resampling

```python
import numpy as np
import volresample

# Create a 3D volume
volume = np.random.rand(128, 128, 128).astype(np.float32)

# Resample to a different size
resampled = volresample.resample(volume, (64, 64, 64), mode='linear')
print(resampled.shape)  # (64, 64, 64)
```

### Cubic Resampling (scipy-compatible)

```python
# Cubic B-spline resampling.
# align_corners=False -> scipy zoom(..., grid_mode=True)
# align_corners=True  -> scipy zoom(..., grid_mode=False)
resampled = volresample.resample(volume, (64, 64, 64), mode='cubic')
```

### Align Corners

```python
# For nearest, linear, and cubic modes, align_corners=True preserves the corner voxels.
aligned = volresample.resample(volume, (192, 192, 192), mode='linear', align_corners=True)

# Nearest with align_corners=True: corner input voxels map exactly to corner output voxels.
# This is not supported by PyTorch's interpolate, but follows the same geometric convention.
nearest_ac = volresample.resample(volume, (192, 192, 192), mode='nearest', align_corners=True)
```

### Multi-Channel Volumes

```python
# 4D volume with 4 channels
volume_4d = np.random.rand(4, 128, 128, 128).astype(np.float32)

# Resample all channels
resampled_4d = volresample.resample(volume_4d, (64, 64, 64), mode='linear')
print(resampled_4d.shape)  # (4, 64, 64, 64)
```

### Batched Multi-Channel Volumes

```python
# 5D volume with batch dimension (N, C, D, H, W)
volume_5d = np.random.rand(2, 4, 128, 128, 128).astype(np.float32)

# Resample all batches and channels
resampled_5d = volresample.resample(volume_5d, (64, 64, 64), mode='linear')
print(resampled_5d.shape)  # (2, 4, 64, 64, 64)
```

### Grid Sampling

```python
# Input volume: (N, C, D, H, W)
input = np.random.rand(2, 3, 32, 32, 32).astype(np.float32)

# Sampling grid with normalized coordinates in [-1, 1]
grid = np.random.uniform(-1, 1, (2, 24, 24, 24, 3)).astype(np.float32)

# Sample with linear interpolation
output = volresample.grid_sample(input, grid, mode='linear', padding_mode='zeros')
print(output.shape)  # (2, 3, 24, 24, 24)
```

### Parallelization

```python
import volresample

# Check default thread count (min of cpu_count and 4)
print(volresample.get_num_threads())  # e.g., 4

# Set custom thread count
volresample.set_num_threads(8)

# All subsequent operations use 8 threads
resampled = volresample.resample(volume, (64, 64, 64), mode='linear')
```

## API Reference

### `resample(data, size, mode='linear', align_corners=False)`

Resample a 3D, 4D, or 5D volume to a new size.

**Parameters:**
- `data` (ndarray): Input volume of shape `(D, H, W)`, `(C, D, H, W)`, or `(N, C, D, H, W)`
- `size` (tuple): Target size `(D_out, H_out, W_out)`
- `mode` (str): Interpolation mode:
  - `'nearest'`: Nearest neighbor (works with all dtypes)
  - `'linear'`: Trilinear interpolation (float32 only)
  - `'area'`: Area-based averaging (float32 only, suited for downsampling)
  - `'cubic'`: Tricubic B-spline interpolation with IIR prefilter (float32 only). Matches `scipy.ndimage.zoom(order=3, mode='reflect')`
- `align_corners` (bool): Supported for `mode='nearest'`, `mode='linear'`, and `mode='cubic'`
  - `False` (default): matches PyTorch `align_corners=False` for linear, and SciPy `grid_mode=True` for cubic
  - `True`: matches PyTorch `align_corners=True` for linear, SciPy `grid_mode=False` for cubic, and aligns corner voxels for nearest (not supported by PyTorch)
  - Passing `align_corners=True` with `area` raises `ValueError`

**PyTorch correspondence:**

| volresample | PyTorch `F.interpolate` |
|-------------|-------------------------|
| `mode='nearest'` | `mode='nearest-exact'` |
| `mode='nearest', align_corners=True` | *(no PyTorch equivalent)* |
| `mode='linear', align_corners=False` | `mode='trilinear', align_corners=False` |
| `mode='linear', align_corners=True` | `mode='trilinear', align_corners=True` |
| `mode='area'` | `mode='area'` |

**SciPy correspondence:**

| volresample | SciPy |
|-------------|-------|
| `mode='cubic', align_corners=False` | `scipy.ndimage.zoom(order=3, mode='reflect', grid_mode=True)` |
| `mode='cubic', align_corners=True` | `scipy.ndimage.zoom(order=3, mode='reflect', grid_mode=False)` |

**Returns:**
- Resampled array with same number of dimensions as input

**Supported Dtypes:**
- `uint8`, `int16`: Only with `mode='nearest'`
- `float32`: All modes (`nearest`, `linear`, `area`, `cubic`)

### `grid_sample(input, grid, mode='linear', padding_mode='zeros', fill_value=0)`

Sample input at arbitrary locations specified by a grid.

**Parameters:**
- `input` (ndarray): Input volume of shape `(N, C, D, H, W)`
- `grid` (ndarray): Sampling grid of shape `(N, D_out, H_out, W_out, 3)`
  - Values in range `[-1, 1]` where -1 maps to the first voxel, 1 to the last
- `mode` (str): `'nearest'` or `'linear'`
- `padding_mode` (str): `'zeros'`, `'border'`, `'reflection'`, or `'constant'`
- `fill_value` (float): Fill value for out-of-bounds samples when `padding_mode='constant'`. For integer dtypes in nearest mode, the value is clamped to the valid range. Default: `0`

**PyTorch correspondence:**

| volresample | PyTorch `F.grid_sample` |
|-------------|-------------------------|
| `mode='nearest'` | `mode='nearest'` |
| `mode='linear'` | `mode='bilinear'` |

The behavior matches PyTorch's `grid_sample` with `align_corners=False`.

**Returns:**
- Sampled array of shape `(N, C, D_out, H_out, W_out)`

**Supported Dtypes:**
- `uint8`, `int16`: Only with `mode='nearest'`
- `float32`: All modes (`nearest`, `linear`)

### `set_num_threads(num_threads)`

Set the number of threads used for parallel operations.

**Parameters:**
- `num_threads` (int): Number of threads to use (must be >= 1)

### `get_num_threads()`

Get the current number of threads used for parallel operations.

**Returns:**
- Current thread count (default: `min(cpu_count, 4)`)

## Performance

Measured on **2026-09-11** with the current Cython implementation on an
**Intel Core i5-14600K**, using four physical performance cores (CPU IDs
`0,2,4,6`). PyTorch and volresample were configured for **4 CPU threads**;
SciPy's thread count was not configured by the benchmark.

Versions: Python 3.13.7, NumPy 2.4.2, PyTorch
`2.14.0+rocm7.14` (CPU execution), and SciPy 1.17.1. The extension
was built with GCC 14.3.0, Cython 3.2.4, and the project's existing AVX2/FMA/OpenMP
compiler flags.

```bash
python setup.py build_ext --inplace
OMP_PROC_BIND=true OMP_PLACES=cores OMP_DYNAMIC=false taskset -c 0,2,4,6 \
  python tests/benchmark.py --threads 4 --profile default
```

Timings are **median / IQR in milliseconds**, measured over at least six rotating
blocks after three warmup calls. The default profile gives each callable at
least **1,800 ms of measured time**. Calls reuse the input and include NumPy
input/output conversion and allocation; input generation and output checks are
outside timing. Speedups use unrounded medians: reference time divided by
volresample time. Values above 1× favor volresample.

**volresample vs PyTorch**

| Case | Shape | PyTorch (ms) | volresample (ms) | Speedup | Max error |
|------|-------|--------------|------------------|---------|-----------|
| nearest | `128x128x128 -> 64x64x64` | 0.055 / 0.001 | 0.031 / 0.000 | 1.77× | 0 |
| nearest (`uint8`) | `128x128x128 -> 64x64x64` | 0.106 / 0.002 | 0.023 / 0.000 | 4.64× | 0 |
| nearest (`int16`) | `128x128x128 -> 64x64x64` | 0.136 / 0.002 | 0.027 / 0.000 | 4.95× | 0 |
| nearest, large volume | `512x512x512 -> 256x256x256` | 10.804 / 0.100 | 7.767 / 0.033 | 1.39× | 0 |
| linear | `128x128x128 -> 64x64x64` | 0.276 / 0.003 | 0.137 / 0.001 | 2.01× | 0 |
| linear, `align_corners=True` | `96x96x96 -> 144x144x144` | 2.683 / 0.034 | 1.420 / 0.031 | 1.89× | `4.50e-05` |
| linear, large volume | `512x512x512 -> 256x256x256` | 25.731 / 0.325 | 17.649 / 0.158 | 1.46× | 0 |
| area | `160x160x160 -> 80x80x80` | 4.179 / 0.114 | 0.733 / 0.038 | 5.70× | `1.19e-07` |
| area, large volume | `512x512x512 -> 64x64x64` | 52.356 / 0.506 | 12.433 / 0.221 | 4.21× | `2.24e-07` |
| 4D linear | `4x96x96x96 -> 64x64x64` | 1.004 / 0.009 | 0.517 / 0.007 | 1.94× | 0 |
| 5D linear | `2x4x80x80x80 -> 48x48x48` | 0.878 / 0.012 | 0.444 / 0.007 | 1.98× | 0 |

**volresample vs SciPy**

| Case | Shape | SciPy (ms) | volresample (ms) | Speedup | Max error |
|------|-------|--------------|------------------|---------|-----------|
| cubic, `align_corners=False` | `128x128x128 -> 64x64x64` | 77.631 / 1.138 | 10.995 / 1.059 | 7.06× | 0 |
| cubic, `align_corners=True` | `96x128x80 -> 64x160x48` | 80.826 / 0.555 | 4.465 / 1.062 | 18.10× | 0 |

**volresample vs PyTorch (`grid_sample`)**

| Case | Shape | PyTorch (ms) | volresample (ms) | Speedup | Max error |
|------|-------|--------------|------------------|---------|-----------|
| linear, zeros | `1x2x96x96x96 -> 80x80x80` | 11.930 / 0.287 | 4.376 / 0.215 | 2.73× | `4.40e-05` |
| nearest, zeros | `1x2x96x96x96 -> 80x80x80` | 3.697 / 0.100 | 0.777 / 0.025 | 4.76× | 0 |
| linear, reflection | `1x2x80x96x64 -> 72x88x56` | 16.535 / 0.377 | 3.462 / 0.061 | 4.78× | `5.42e-05` |

Arithmetic mean speedup across these 16 cases: **4.34×**.
This summary depends on the selected workloads.

**Notes:**

- Cubic timings include spline prefiltering for both implementations. The two rows cover both `align_corners` settings.
- For `int16` nearest, the PyTorch reference converts to `float32` and back; volresample operates directly on `int16`.
- Supplemental prepared-tensor PyTorch timings printed by the benchmark exclude input wrapping/casts and output conversion. They are not used in these tables or the speedup summary.
- Max error is the maximum absolute difference from the reference output. Floating-point interpolation can differ slightly because of rounding and operation order.
- These are CPU measurements on this machine. Core selection, memory bandwidth, thermal conditions, library versions, and workload shape affect performance.

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with PyTorch comparison tests
pip install torch
pytest tests/ -v

# Skip PyTorch tests
pytest tests/ --skip-torch
```


### Running Benchmarks

```bash
# Curated default run: all modes plus grid_sample
python tests/benchmark.py

# Faster smoke benchmark
python tests/benchmark.py --profile quick

# Or pin the thread count
python tests/benchmark.py --threads 4

# Output is printed live while the benchmark runs
python -u tests/benchmark.py
```

The main tables measure end-to-end CPU calls from NumPy input to NumPy output.
Times are median / IQR of per-call averages from at least six measurement blocks,
with three warmup calls and rotating backend order. Each callable receives the
profile's measured time budget (or `--target-ms`); total runtime also includes
warmup and output validation.

Supplemental PyTorch tables time `F.interpolate` / `F.grid_sample` with prepared
CPU tensors. They include dispatch and output allocation, but exclude input
wrapping, dtype conversion, and output conversion, and do not enter speedup
summaries. For int16 nearest, the end-to-end PyTorch path includes conversion to
float32 and back; its prepared timing uses float32 tensors. Cubic SciPy timings
include spline prefiltering and return float32 directly. `--threads` configures
PyTorch and volresample; it does not configure SciPy.

For Cython changes, `python -m tests.experiment` builds isolated source snapshots
and compares them in the same process with correctness gates, repeated rounds,
and JSON results. Run `python -m tests.experiment --help` for available commands.

### Building from Source

```bash
pip install -e ".[dev]"
python setup.py build_ext --inplace
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome. Please submit a Pull Request.
