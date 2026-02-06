# volresample

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Fast 3D volume resampling with Cython and OpenMP parallelization.

Implemented against PyTorch's `F.interpolate` and `F.grid_sample` as a reference, producing identical results. Can be used as a drop-in replacement when PyTorch is not available or when better performance is desired on CPU.

## Features

- Cython-optimized with OpenMP parallelization
- Simple API: `resample()` and `grid_sample()`
- Interpolation modes: nearest, linear and area
- Supports 3D and 4D (multi-channel) volumes
- Supports uint8, int16 (nearest) and float32 dtypes (all)

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

**PyTorch correspondence:**

| volresample | PyTorch `F.interpolate` |
|-------------|-------------------------|
| `mode='nearest'` | `mode='nearest-exact'` |
| `mode='linear'` | `mode='trilinear'` |
| `mode='area'` | `mode='area'` |

volresample does not expose an `align_corners` parameter. The behavior matches PyTorch's `align_corners=False` (the default).

**Returns:**
- Resampled array with same number of dimensions as input

**Supported Dtypes:**
- `uint8`, `int16`: Only with `mode='nearest'`
- `float32`: All modes

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

Benchmarks comparing volresample vs PyTorch 2.8.0 on Intel CPU with 4 threads:

| Operation | Size | Mode | volresample | PyTorch | Speedup |
|-----------|------|------|-------------|---------|---------|
| Resample | 64³ → 32³ | nearest | 0.01 ms | 0.10 ms | 8x |
| Resample | 512³ → 256³ | nearest | 15 ms | 18 ms | 1.3x |
| Resample | 512³ → 256³ | linear | 43 ms | 61 ms | 1.4x |
| Resample | 512³ → 256³ | area | 74 ms | 663 ms | 9x |
| Resample | 512³ → 256³ | nearest (uint8) | 5 ms | 14 ms | 2.6x |
| Resample | 512³ → 256³ | nearest (int16) | 9 ms | 161 ms | 18x |
| Grid sample | 128³ → 96³ | linear | 59 ms | 232 ms | 4x |

**Notes on speedups:**

- **Area mode (9x)**: volresample uses a direct area-weighted computation optimized for the resampling use case. Pytorch doesn't seem to optimally parallelize.
- **int16 (18x)**: PyTorch does not natively support int16 for interpolation and requires casting to float32 and back, adding memory bandwidth and conversion overhead. volresample operates directly on int16 data.
- **uint8 (2.6x)**: PyTorch supports uint8 natively for nearest mode, so the speedup is more modest.
- **Grid sample (4x)**: The Cython implementation avoids Python overhead and uses cache-friendly memory access patterns.

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
