"""Pytest configuration and shared helpers for volresample tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add tests directory to path so torch_reference can be imported
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    from torch_reference import TorchReference  # noqa: F401

    TORCH_AVAILABLE = TorchReference.available
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy.ndimage import zoom  # noqa: F401

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
requires_scipy = pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available")

ROOT = Path(__file__).resolve().parents[1]

# Tolerances
ATOL = 1e-4  # general torch comparison
ATOL_CUBIC = 5e-4  # cubic vs scipy


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_data(shape, *, seed=42):
    """Generate random float32 test data."""
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32)


def scipy_cubic(vol, out_shape, *, align_corners=False):
    """Reference cubic resampling via scipy.ndimage.zoom(order=3, mode='reflect')."""
    from scipy.ndimage import zoom

    factors = tuple(o / i for o, i in zip(out_shape, vol.shape[-3:]))
    return zoom(
        vol.astype(np.float64),
        factors,
        order=3,
        mode="reflect",
        grid_mode=not align_corners,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI options and hooks
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--skip-torch",
        action="store_true",
        default=False,
        help="Skip PyTorch comparison tests",
    )
    parser.addoption(
        "--require-torch",
        action="store_true",
        default=False,
        help="Fail if PyTorch is not available",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "torch: tests that require PyTorch backend for comparison")
    config.addinivalue_line("markers", "slow: slow running tests")
    if config.getoption("--require-torch"):
        try:
            from torch_reference import TorchReference

            if not TorchReference.available:
                raise ImportError("PyTorch is required but not available")
        except ImportError as e:
            raise RuntimeError(
                "PyTorch is required (--require-torch). Install with: pip install torch"
            ) from e


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--skip-torch"):
        return
    skip_marker = pytest.mark.skip(reason="PyTorch tests skipped (--skip-torch)")
    for item in items:
        if "torch" in item.nodeid.lower() or any(
            marker.name == "skipif" and "TORCH_AVAILABLE" in str(marker.args)
            for marker in item.iter_markers()
        ):
            item.add_marker(skip_marker)
