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
- Supports `align_corners=True` for linear and cubic resampling
- Supports uint8, int16 (nearest) and float32 dtypes (all other modes)

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
# For linear and cubic modes, align_corners=True preserves the corner voxels.
aligned = volresample.resample(volume, (192, 192, 192), mode='linear', align_corners=True)
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
- `align_corners` (bool): Only supported for `mode='linear'` and `mode='cubic'`
  - `False` (default): matches PyTorch `align_corners=False` for linear, and SciPy `grid_mode=True` for cubic
  - `True`: matches PyTorch `align_corners=True` for linear, and SciPy `grid_mode=False` for cubic
  - Passing `align_corners=True` with `nearest` or `area` raises `ValueError`

**PyTorch correspondence:**

| volresample | PyTorch `F.interpolate` |
|-------------|-------------------------|
| `mode='nearest'` | `mode='nearest-exact'` |
| `mode='linear', align_corners=False` | `mode='trilinear', align_corners=False` |
| `mode='linear', align_corners=True` | `mode='trilinear', align_corners=True` |
| `mode='area'` | `mode='area'` |

`align_corners` is intentionally limited to the modes where the reference APIs support it: linear and cubic.

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

### `grid_sample(input, grid, mode='linear', padding_mode='zeros')`

Sample input at arbitrary locations specified by a grid.

**Parameters:**
- `input` (ndarray): Input volume of shape `(N, C, D, H, W)`
- `grid` (ndarray): Sampling grid of shape `(N, D_out, H_out, W_out, 3)`
  - Values in range `[-1, 1]` where -1 maps to the first voxel, 1 to the last
- `mode` (str): `'nearest'` or `'linear'`
- `padding_mode` (str): `'zeros'`, `'border'`, or `'reflection'`

**PyTorch correspondence:**

| volresample | PyTorch `F.grid_sample` |
|-------------|-------------------------|
| `mode='nearest'` | `mode='nearest'` |
| `mode='linear'` | `mode='bilinear'` |

The behavior matches PyTorch's `grid_sample` with `align_corners=False`.

**Returns:**
- Sampled array of shape `(N, C, D_out, H_out, W_out)`

### `set_num_threads(num_threads)`

Set the number of threads used for parallel operations.

**Parameters:**
- `num_threads` (int): Number of threads to use (must be >= 1)

### `get_num_threads()`

Get the current number of threads used for parallel operations.

**Returns:**
- Current thread count (default: `min(cpu_count, 4)`)

## Performance

Benchmarks below were produced by the default curated benchmark profile on an Intel i7-8565U using 4 CPU threads:

```bash
python tests/benchmark.py --threads 4
```

The default profile runs for about 30-60 seconds using adaptive repeat counts.

**volresample vs PyTorch**

| Case | Shape | PyTorch | volresample | Speedup | Max error |
|------|-------|-----------|-------------|---------|-----------|
| nearest | `128x128x128 -> 64x64x64` | 0.98 ms | 0.41 ms | 2.36× | 0 |
| nearest (`uint8`) | `128x128x128 -> 64x64x64` | 0.83 ms | 0.27 ms | 3.07× | 0 |
| nearest (`int16`) | `128x128x128 -> 64x64x64` | 4.20 ms | 0.40 ms | 10.54× | 0 |
| linear | `128x128x128 -> 64x64x64` | 3.45 ms | 2.33 ms | 1.48× | 0 |
| linear, `align_corners=True` | `96x96x96 -> 144x144x144` | 23.75 ms | 10.46 ms | 2.27× | `4.50e-05` |
| area | `160x160x160 -> 80x80x80` | 40.12 ms | 7.20 ms | 5.57× | 0 |
| 4D linear | `4x96x96x96 -> 64x64x64` | 12.01 ms | 7.50 ms | 1.60× | 0 |
| 5D linear | `2x4x80x80x80 -> 48x48x48` | 9.76 ms | 8.49 ms | 1.15× | 0 |

**volresample vs SciPy**

| Case | Shape | SciPy | volresample | Speedup | Max error |
|------|-------|-------|-------------|---------|-----------|
| cubic, `align_corners=False` | `128x128x128 -> 64x64x64` | 266.29 ms | 63.68 ms | 4.18× | `8.34e-07` |
| cubic, `align_corners=True` | `96x128x80 -> 64x160x48` | 403.68 ms | 46.23 ms | 8.73× | `1.43e-06` |

**volresample vs PyTorch (`grid_sample`)**

| Case | Shape | PyTorch | volresample | Speedup | Max error |
|------|-------|-----------|-------------|---------|-----------|
| linear, zeros | `1x2x96x96x96 -> 80x80x80` | 129.57 ms | 41.63 ms | 3.11× | `4.40e-05` |
| nearest, zeros | `1x2x96x96x96 -> 80x80x80` | 14.23 ms | 4.47 ms | 3.18× | 0 |
| linear, reflection | `1x2x80x96x64 -> 72x88x56` | 91.12 ms | 19.75 ms | 4.61× | `5.42e-05` |

Average speedup across the default benchmark suite: **3.99×**.

**Notes:**

- **Cubic mode** is validated against SciPy rather than PyTorch. The `align_corners` flag selects between SciPy's `grid_mode=True` and `grid_mode=False`, and both paths are benchmarked above.
- **`int16` nearest** shows the largest speedup because PyTorch must round-trip through `float32`, while volresample operates directly on `int16`.
- **Area mode** remains one of the strongest CPU wins because the implementation parallelizes efficiently over spatial work.
- **4D and 5D coverage** is included in the benchmark suite so multi-channel and batched paths are represented, even when the raw speedups are smaller than the single-volume cases.
- **These are machine-specific measurements.** CPU architecture, memory bandwidth, thermal throttling, and installed library versions can shift the absolute numbers substantially.

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
# Curated default run: all modes plus grid_sample, roughly 30-60 seconds
python tests/benchmark.py

# Faster smoke benchmark
python tests/benchmark.py --profile quick

# Or pin the thread count
python tests/benchmark.py --threads 4

# Output is printed live while the benchmark runs
python -u tests/benchmark.py
```

### Building from Source

```bash
pip install -e ".[dev]"
python setup.py build_ext --inplace
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome. Please submit a Pull Request.
