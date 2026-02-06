"""
volresample: Fast 3D volume resampling with Cython
===================================================

A high-performance Python library for 3D medical image resampling using
optimized Cython implementations with OpenMP parallelization.

Main Functions
--------------
resample : Resample 3D or 4D volumes using various interpolation modes
grid_sample : Sample volumes using arbitrary sampling grids

Thread Configuration
--------------------
set_num_threads : Set the number of threads for parallel operations
get_num_threads : Get the current number of threads

Examples
--------
Basic resampling:

>>> import numpy as np
>>> import volresample
>>> data = np.random.rand(64, 64, 64).astype(np.float32)
>>> resampled = volresample.resample(data, (32, 32, 32), mode='linear')
>>> resampled.shape
(32, 32, 32)

Multi-channel resampling:

>>> data_4d = np.random.rand(4, 64, 64, 64).astype(np.float32)
>>> resampled_4d = volresample.resample(data_4d, (32, 32, 32), mode='linear')
>>> resampled_4d.shape
(4, 32, 32, 32)

Batched multi-channel resampling:

>>> data_5d = np.random.rand(2, 4, 64, 64, 64).astype(np.float32)
>>> resampled_5d = volresample.resample(data_5d, (32, 32, 32), mode='linear')
>>> resampled_5d.shape
(2, 4, 32, 32, 32)

Grid sampling:

>>> input = np.random.rand(1, 2, 32, 32, 32).astype(np.float32)
>>> grid = np.random.uniform(-1, 1, (1, 24, 24, 24, 3)).astype(np.float32)
>>> output = volresample.grid_sample(input, grid, mode='linear')
>>> output.shape
(1, 2, 24, 24, 24)

Thread configuration:

>>> volresample.set_num_threads(4)
>>> volresample.get_num_threads()
4
"""

__version__ = "0.1.0"

# Import thread configuration (always available, pure Python)
from volresample._config import get_num_threads, set_num_threads

try:
    from volresample._resample import grid_sample, resample
except ImportError as e:
    import warnings

    warnings.warn(
        f"Failed to import compiled Cython extensions: {e}. "
        "Please build the package with 'pip install -e .'",
        ImportWarning,
        stacklevel=2,
    )
    resample = None
    grid_sample = None

__all__ = ["resample", "grid_sample", "set_num_threads", "get_num_threads", "__version__"]
