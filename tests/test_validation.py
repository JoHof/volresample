"""Input validation tests for resample() and grid_sample()."""

import os
import subprocess
import sys

import numpy as np
import pytest
from conftest import ROOT, TORCH_AVAILABLE

import volresample

if TORCH_AVAILABLE:
    from torch_reference import TorchReference  # noqa: F401


# ============================================================================
# resample() validation
# ============================================================================


def test_align_corners_invalid_modes():
    arr = np.ones((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="align_corners"):
        volresample.resample(arr, (2, 2, 2), mode="area", align_corners=True)


@pytest.mark.parametrize("mode", ["nearest", "linear", "area", "cubic"])
def test_resample_rejects_zero_output_dimension(mode):
    data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    with pytest.raises(ValueError):
        volresample.resample(data, (0, 2, 2), mode=mode)


def test_resample_invalid_size_tuple_does_not_crash():
    code = (
        "import numpy as np; import volresample; "
        "data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)\n"
        "try:\n"
        "    volresample.resample(data, (2, 2), mode='linear')\n"
        "except ValueError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    env = os.environ.copy()
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(volresample.__file__)))
    pythonpath_entries = [package_root]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


# ============================================================================
# grid_sample() validation
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_grid_sample_rejects_batch_mismatch(mode):
    rng = np.random.default_rng(12)
    inp = rng.normal(size=(1, 2, 3, 3, 3)).astype(np.float32)
    grid = rng.uniform(-1, 1, size=(2, 4, 4, 4, 3)).astype(np.float32)
    with pytest.raises(ValueError):
        volresample.grid_sample(inp, grid, mode=mode)
