# volresample Release Notes

Notable changes in volresample are summarized here by release.

The canonical published artifacts are attached to GitHub Releases and published on PyPI.

## 0.5.0

- Add `align_corners=True` support for nearest neighbor interpolation.
  Corner output voxels map exactly to corner input voxels using `i * (in_n - 1) / (out_n - 1)` coordinate mapping.
  Works with all dtypes (`uint8`, `int16`, `float32`) and all input shapes (3D, 4D, 5D).
  Note: this mode has no PyTorch equivalent.


## 0.4.1

- Small performance improvement for nearest interpolation, specifically on larger input sizes


## 0.4.0

- Add `padding_mode="constant"` to `grid_sample` with customizable `fill_value`.
- Support integer dtypes (`uint8`, `int16`) in `grid_sample`.
- Improve `grid_sample` performance by parallelizing over batch and channel dimensions.


## 0.3.0

- Optimize linear resampling with lower allocation overhead in the core 3D path.
- Add fused multi-channel and batched linear fast paths for 4D and 5D inputs.
- Improve area resampling runtime with a more streaming-friendly core accumulation order and precomputed index ranges.


## 0.2.0

- Support for cubic interpolation
- Support for align_corners=True
- Update benchmark script
- Updated packaging and release workflow metadata.

## 0.1.0

- Basic functionality for volume (3D) resampling
- Developed against PyTorchs interpolate and grid_sample functions
- Support for nearest neighbor, linear and area interpolation modes
- align_corners=False only
