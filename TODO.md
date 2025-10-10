# Tasks

## Completed ✓
- [x] **3D Constraint Implementation**: Enforced exactly 3 spatial dimensions in Mimage
- [x] **Indexing Correctness**: Fixed `__getitem__` to match SimpleITK behavior for stride/reverse slicing
  - Handles stride slicing `[::step]` (scales spacing by abs(step))
  - Handles reverse slicing `[::-1]` (flips direction, adjusts origin)
  - Handles negative indices and combined slicing `[start:stop:step]`
- [x] **Comprehensive Indexing Tests**: Created test_mimage_indexing_vs_sitk.py with 14 tests
  - All 143 tests passing (113 original + 14 indexing + 16 clip)
- [x] **Unit Tests for Core Components**:
  - affine.py: 31 tests (construction, properties, transformations, vs SimpleITK)
  - mimage.py: 62 tests (construction, conversions, indexing, transformations, clip)
  - utils.py: 12 tests (orientation from direction cosines)
  - backends: 34 tests (numpy/torch consistency, including clip operations)
- [x] **Clip Functionality**: Implemented `clip()` method for value clamping
  - Numpy backend clip implementation
  - Torch backend clip implementation
  - Mimage.clip(min_val, max_val) method
  - Comprehensive tests for both backends
  - Preserves immutability pattern

## High Priority

### Documentation
- [ ] Add comprehensive README.md with examples and API reference
- [ ] Document the difference between stride slicing (subsampling) vs resampling (interpolation)
- [ ] Add usage examples for coordinate transformations

### Implement resampling with correct spacing/origin updates
- [ ] Design architecture: allow different backends (PyTorch, numba) for resampling
- [ ] Implement affine/mimage methods to update spacing and origin correctly
- [ ] Implement PyTorch backend functionality
- [ ] Implement numba backend functionality (see edt_3d_numba as reference)
- [ ] Implement mimage.resample() method (input: Mimage, output: Mimage with updated affine)
- [ ] Add unit tests comparing against SimpleITK.Resample()

**Note**: Resampling should support nearest, linear (trilinear), area, and cubic interpolation modes.
- `nearest` = PyTorch's 'nearest-exact' mode
- `linear` = PyTorch's 'trilinear' mode  
- `cubic` = Applied slice-by-slice (2D cubic on each slice)
- This creates a NEW grid covering physical space (different from stride slicing which subsamples)

## Medium Priority

### Additional Features
- [ ] Implement cropping/padding operations with correct affine updates
- [ ] Add support for non-uniform spacing along axes
<!-- - [ ] Implement image registration utilities -->
<!-- - [ ] Add visualization helpers (integrate with miviz) -->

### Performance Optimizations
<!-- - [ ] Profile coordinate transformation performance
- [ ] Optimize indexing operations for large arrays -->
- [ ] Consider caching affine matrix inverse

## Low Priority

### Quality of Life
- [ ] Add __array__ protocol for numpy compatibility
- [ ] Implement __array_ufunc__ for better numpy interop
- [ ] Add comparison operators (==, !=) for Mimage objects
- [ ] Better error messages with suggestions for common mistakes

### Testing Infrastructure
- [ ] Add property-based tests (hypothesis)
<!-- - [ ] Set up continuous integration -->
- [ ] Add benchmarks for performance regression testing