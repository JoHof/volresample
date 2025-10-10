# Copilot Instructions for mimage project

## Code Style
- Use Google-style docstrings for all functions/classes
- Always include type hints
- Maximum line length: 100 characters

## Architecture Rules
- Use composition over inheritance (no subclassing torch.Tensor or np.ndarray)
- Implement backend handlers for torch/numpy (see backends/ folder)
- Use lazy imports for optional dependencies (torch, nibabel)
- All transformations should return new Mimage instances (immutable)
- **3D Constraint**: Mimage ALWAYS has exactly 3 spatial dimensions (consistent with 4x4 affine)

## Functionality
- Coordinate transformations: index_to_coord, coord_to_index
- Axis manipulations: flip, permute
- Resampling with correct spacing/origin updates
- Support for additional non-spatial dimensions (batch, channel, time)
- Indexing behavior matches SimpleITK:
  - `[start:stop]`: Origin shifts by start * spacing
  - `[::step]`: Spacing scales by abs(step), origin unchanged
  - `[::-1]`: Direction flips, origin moves to last voxel center
  - Combined slicing applies both transformations

## Naming Conventions
- spatial_dims: tuple tracking which dimensions are spatial (always exactly 3, always sorted)
- affine: Affine instance managing transformations (always 4x4 for 3D space)
- origin: Physical location (mm) of the CENTER of first voxel [0,0,0]
- spacing: Voxel size (mm) in each spatial dimension
- direction: 3x3 orthonormal matrix defining axis orientations in physical space
- Backend handlers must implement: flip, permute, asarray, clone methods

## Spatial Dimensions Constraint
- Arrays must have at least 3 dimensions (ndim >= 3)
- Exactly 3 dimensions are designated as "spatial" via spatial_dims
- Spatial dimensions CANNOT be removed via indexing (raises ValueError)
- Additional dimensions (batch, channel, time) are allowed
- Use slicing to keep dimensions: `img[0:1]` not `img[0]` for spatial dims

## Error Handling
- Raise ImportError with clear message if optional dependency missing
- Raise ValueError if trying to remove spatial dimensions
- Validate shapes and dimensions before operations

## Testing
- Test both numpy and torch backends
- Test with 3D arrays (spatial only) and 4D+ arrays (with non-spatial dims)
- Verify spatial dimension removal raises appropriate errors
- Use SimpleITK as reference for coordinate transformations and indexing behavior
- Test all slicing patterns: simple, stride, reverse, negative indices, combined
- Verify coordinate transformations work correctly after complex slicing operations
- All tests should pass: run `pytest tests/` to verify (127 tests as of now)


