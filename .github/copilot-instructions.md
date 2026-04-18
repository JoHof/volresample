# volresample Copilot Instructions

## Project Focus

- This repository is a performance-first 3D volume resampling library built around Cython and OpenMP.
- Speed is a top priority, but never at the expense of breaking established numerical behavior.
- Consistency with reference libraries is a top priority:
	- `resample(..., mode="nearest" | "linear" | "area")` should stay aligned with PyTorch behavior.
	- `grid_sample(...)` should stay aligned with PyTorch `F.grid_sample(..., align_corners=False)`.
	- `resample(..., mode="cubic")` should stay aligned with `scipy.ndimage.zoom(order=3, mode="reflect")`, using `grid_mode=True` when `align_corners=False` and `grid_mode=False` when `align_corners=True`.

## Environment And Commands

- When opening a new shell, run `nix develop` first.
- Inside the Nix shell, prefer `uv` for Python environment management and dependency installation.
- If dependencies are missing, use `uv sync --group dev`.
- Prefer targeted validation while iterating, then broader validation before finishing.

Useful commands:

```bash
nix develop
uv sync --group dev
pytest tests/
pytest tests/test_resample.py tests/test_cubic.py tests/test_grid_sample.py tests/test_validation.py
python tests/benchmark.py --profile quick --threads 4
ruff check .
ruff format .
```

## Codebase Map

- Public API lives in `src/volresample/__init__.py`.
- Thread configuration lives in `src/volresample/_config.py`.
- The compiled extension entry point is `src/volresample/_resample.pyx`.
- Core implementations are split across `src/volresample/cython_src/*.pyx` and included into `_resample.pyx` at compile time.
- Public typing surface lives in `src/volresample/_resample.pyi`.
- Tests are the main source of expected behavior:
	- `tests/test_resample.py` for nearest, linear, area, and shape coverage.
	- `tests/test_cubic.py` for SciPy parity and cubic edge cases.
	- `tests/test_grid_sample.py` for `grid_sample` parity.
	- `tests/test_validation.py` for argument validation and crash prevention.
- `tests/benchmark.py` is the main performance benchmark.
- Avoid editing generated artifacts in `build/`, `*.egg-info`, or generated C output unless explicitly required.

## Performance Guidance

- Prefer fixes that preserve or improve throughput on CPU.
- Treat `src/volresample/_resample.pyx` and `src/volresample/cython_src/*.pyx` as hot paths.
- Avoid adding Python-level overhead inside core loops.
- Preserve the current performance-oriented approach where applicable:
	- contiguous arrays,
	- precomputed indices and weights,
	- `nogil` sections,
	- OpenMP `prange`,
	- minimal branching in inner loops,
	- architecture-aware compiler flags in `setup.py`.
- Do not replace a compiled-path optimization with a cleaner but slower pure Python approach unless explicitly requested.
- If a change may affect performance, run at least a targeted quick benchmark.

## Correctness And Compatibility Rules

- Preserve support for 3D `(D, H, W)`, 4D `(C, D, H, W)`, and 5D `(N, C, D, H, W)` inputs in `resample`.
- Preserve support for 5D input plus 5D grid in `grid_sample`.
- Preserve dtype rules unless the task explicitly changes them:
	- `nearest` supports `uint8`, `int16`, and `float32`.
	- `linear`, `area`, and `cubic` are `float32` paths.
- Preserve validation behavior:
	- invalid shapes and malformed `size` arguments should raise Python exceptions, not crash,
	- zero output dimensions should be rejected,
	- `align_corners=True` is only valid for `linear` and `cubic`,
	- `grid_sample` should reject batch mismatches.
- Be especially careful with singleton axes, short axes, non-uniform scaling, and multi-channel or batched shapes. Those are explicitly covered by tests.

## Editing Expectations

- Prefer minimal, targeted changes.
- If you change public behavior or signatures, keep these in sync:
	- `src/volresample/__init__.py`
	- `src/volresample/_resample.pyi`
	- relevant tests
	- `README.md` when user-facing behavior changes
- If you touch Cython API behavior, update or add tests that compare against the reference library rather than only adding smoke tests.
- Type hints are encouraged for Python code and stubs.
- Docstrings are encouraged for public Python-facing functions and should use Google style.

## Validation Strategy

- For numerical behavior changes, prefer focused parity tests against PyTorch or SciPy first.
- For validation fixes, run the relevant validation tests and any regression tests covering the affected mode.
- For performance-sensitive changes, run `python tests/benchmark.py --profile quick --threads 4` unless the change is obviously documentation-only.
- Before finishing substantial code changes, run the smallest set of tests that credibly covers the edited behavior.

## Style Notes

- Target Python 3.9+ compatible code.
- Use Ruff for formatting and linting conventions.
- Keep the implementation pragmatic and compact; avoid unnecessary abstraction in hot code.
