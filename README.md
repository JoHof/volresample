# volresample

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Fast 3D volume resampling with Cython and OpenMP parallelization.

Implemented against PyTorch's `F.interpolate` and `F.grid_sample` as a reference, producing identical results for nearest, linear, and area modes. The cubic mode matches `scipy.ndimage.zoom(order=3, mode='reflect', grid_mode=True)` instead. Can be used as a drop-in replacement when PyTorch or SciPy is not available or when better performance is desired on CPU.

[Blogpost](https://johof.github.io/2026/02/volresample-3d-volume-resampling/)
## Features

- Cython-optimized with OpenMP parallelization
- Simple API: `resample()` and `grid_sample()`
- Interpolation modes: nearest, linear, area, and cubic
- Supports 3D and 4D (multi-channel) volumes
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
# Cubic B-spline resampling — matches scipy.ndimage.zoom(order=3)
resampled = volresample.resample(volume, (64, 64, 64), mode='cubic')
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

### `resample(data, size, mode='linear')`

Resample a 3D, 4D, or 5D volume to a new size.

**Parameters:**
- `data` (ndarray): Input volume of shape `(D, H, W)`, `(C, D, H, W)`, or `(N, C, D, H, W)`
- `size` (tuple): Target size `(D_out, H_out, W_out)`
- `mode` (str): Interpolation mode:
  - `'nearest'`: Nearest neighbor (works with all dtypes)
  - `'linear'`: Trilinear interpolation (float32 only)
  - `'area'`: Area-based averaging (float32 only, suited for downsampling)
  - `'cubic'`: Tricubic B-spline interpolation with IIR prefilter (float32 only). Matches `scipy.ndimage.zoom(order=3, mode='reflect', grid_mode=True)`

**PyTorch correspondence:**

| volresample | PyTorch `F.interpolate` |
|-------------|-------------------------|
| `mode='nearest'` | `mode='nearest-exact'` |
| `mode='linear'` | `mode='trilinear'` |
| `mode='area'` | `mode='area'` |

volresample does not expose an `align_corners` parameter. The behavior matches PyTorch's `align_corners=False` (the default).

**SciPy correspondence:**

| volresample | SciPy |
|-------------|-------|
| `mode='cubic'` | `scipy.ndimage.zoom(order=3, mode='reflect', grid_mode=True)` |

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

Benchmarks on an Intel i7-8565U against PyTorch 2.8.0. Times are means over 10 iterations.

**`resample()`** — single large 3D volume:

| Operation   | Mode            | **Single-thread** |         |           | **Four-threads** |         |          |
| ----------- | --------------- | ----------------- | ------- | :-------: | ---------------- | ------- | :------: |
|             |                 | volresample       | PyTorch |  Speedup  | volresample      | PyTorch |  Speedup |
| 512³ → 256³ | nearest         | 23.6 ms           | 38.0 ms |    1.6×   | 12.6 ms          | 16.7 ms |   1.3×   |
| 512³ → 256³ | linear          | 99.9 ms           | 182 ms  |    1.8×   | 34.3 ms          | 54.6 ms |   1.6×   |
| 512³ → 256³ | area            | 230 ms            | 611 ms  |    2.7×   | 64.5 ms          | 613 ms  | **9.5×** |
| 512³ → 256³ | nearest (uint8) | 13.7 ms           | 33.8 ms |    2.5×   | 4.3 ms           | 10.4 ms |   2.4×   |
| 512³ → 256³ | nearest (int16) | 16.5 ms           | 217 ms  | **13.2×** | 8.4 ms           | 93.2 ms |   11.2×  |


**`grid_sample()`** — single large 3D volume (128³ input):

| Mode   | Padding    | **Single-thread** |         |         | **Four-threads** |         |         |
| ------ | ---------- | ----------------- | ------- | :-----: | ---------------- | ------- | :-----: |
|        |            | volresample       | PyTorch | Speedup | volresample      | PyTorch | Speedup |
| linear | zeros      | 118 ms            | 181 ms  |   1.5×  | 38.1 ms          | 169 ms  |   4.4×  |
| linear | reflection | 103 ms            | 211 ms  |   2.1×  | 33.2 ms          | 194 ms  |   5.9×  |


Average speedup across all benchmarks: **3.1× at 1 thread**, **6.0× at 4 threads**.

**Notes:**

- **Area mode**: At 1 thread the speedup is 2.7×; at 4 threads it reaches 9.5×. PyTorch's area interpolation does not appear to parallelize over spatial dimensions for single-image workloads — its runtime is essentially unchanged between 1 and 4 threads (611 ms vs. 613 ms). volresample parallelizes along the first spatial dimension, reducing runtime from 230 ms to 65 ms with 4 threads.
- **int16**: PyTorch does not support int16 interpolation natively and requires casting to float32, processing, then casting back. volresample operates directly on int16, eliminating two full-volume type conversions. The advantage is large even at 1 thread (13.2×) and persists at 4 threads because the conversion overhead scales with data volume, not thread count.
- **Thread scaling**: For large volumes, volresample typically halves wall time going from 1 to 4 threads on nearest and linear modes. Grid sample scales more strongly (1.5× → 4.4× for linear) because per-voxel work is higher. PyTorch scaling is more variable, and negligible for area mode.
- **These are estimates** on a single machine under light load. Actual results will vary with CPU architecture, memory bandwidth, and system conditions.

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
# Use default threads (min of cpu_count and 4)
python tests/benchmark_resampling.py --iterations 10

# Or specify thread count
python tests/benchmark_resampling.py --threads 4 --iterations 10
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
