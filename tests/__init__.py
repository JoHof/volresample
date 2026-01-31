"""Unit tests for volresample package.

PyTorch Testing
---------------
When PyTorch is available, additional comparison tests are automatically enabled
to validate the Cython implementation against PyTorch's reference implementation.

To run tests with PyTorch comparison:
    pip install torch
    pytest tests/

To run tests without PyTorch (only basic functionality):
    pytest tests/ -v

Stats:
- With PyTorch: 115 tests pass (all 3D tests)
- Without PyTorch: 72 tests pass, 43 PyTorch comparison tests skipped

Note: volresample only supports 3D grid sampling. 2D tests have been removed.
"""
