"""Pytest configuration for volresample tests."""

import sys
from pathlib import Path

# Add tests directory to path so torch_reference can be imported
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--skip-torch",
        action="store_true",
        default=False,
        help="Skip PyTorch comparison tests even if PyTorch is available",
    )
    parser.addoption(
        "--require-torch",
        action="store_true",
        default=False,
        help="Fail if PyTorch is not available (for CI validation)",
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "torch: tests that require PyTorch backend for comparison")
    config.addinivalue_line("markers", "slow: slow running tests")

    # Check PyTorch availability if required
    if config.getoption("--require-torch"):
        try:
            from torch_reference import TorchReference

            if not TorchReference.available:
                raise ImportError("PyTorch is required but not available")
        except ImportError as e:
            raise RuntimeError(
                "PyTorch is required for these tests (--require-torch flag set). "
                "Install with: pip install torch"
            ) from e


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if config.getoption("--skip-torch"):
        skip_torch = True
    else:
        skip_torch = False

    if skip_torch:
        import pytest

        skip_marker = pytest.mark.skip(reason="PyTorch tests skipped (--skip-torch flag)")
        for item in items:
            # Skip tests that have @pytest.mark.skipif(not TORCH_AVAILABLE, ...)
            if "torch" in item.nodeid.lower() or any(
                marker.name == "skipif" and "TORCH_AVAILABLE" in str(marker.args)
                for marker in item.iter_markers()
            ):
                item.add_marker(skip_marker)
