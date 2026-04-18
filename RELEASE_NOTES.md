# volresample Release Notes

Notable changes in volresample are summarized here by release.

The canonical published artifacts are attached to GitHub Releases and published on PyPI.


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
