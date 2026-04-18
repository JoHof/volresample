#!/usr/bin/env python3
"""Unified benchmark suite for volresample.

Default profile aims for a useful 30-60 second run on a typical laptop CPU.

Covers:
- resample() vs PyTorch for nearest, linear, and area
- cubic resample() vs SciPy for both align_corners variants
- grid_sample() vs PyTorch for representative interpolation/padding combinations
"""

from __future__ import annotations

import argparse
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

import volresample

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from torch_reference import TorchReference

try:
    import torch
except ImportError:
    torch = None

try:
    from scipy.ndimage import zoom
except ImportError:
    zoom = None


WARMUP_RUNS = 1
MIN_RUNS = 1
MAX_RUNS = 250
LINE_WIDTH = 122

PROFILE_TARGET_MS = {
    "quick": 600.0,
    "default": 1800.0,
    "full": 3000.0,
}


@dataclass(frozen=True)
class ResampleCase:
    label: str
    input_shape: tuple[int, ...]
    output_size: tuple[int, int, int]
    mode: str
    dtype: Any = np.float32
    align_corners: bool = False
    reference: str = "torch"
    seed: int = 42


@dataclass(frozen=True)
class GridSampleCase:
    label: str
    input_shape: tuple[int, int, int, int, int]
    grid_shape: tuple[int, int, int, int, int]
    mode: str
    padding_mode: str
    grid_kind: str = "random"
    grid_range: tuple[float, float] = (-1.25, 1.25)
    seed: int = 123


@dataclass
class Measurement:
    mean_ms: float
    std_ms: float
    runs: int
    output: np.ndarray


@dataclass
class ComparisonRow:
    case: str
    config: str
    shape: str
    reference_name: str
    reference_ms: float | None
    reference_std_ms: float | None
    volresample_ms: float
    volresample_std_ms: float
    speedup: float | None
    max_error: float | None
    runs: str


def product(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def resample_shape_text(input_shape: tuple[int, ...], output_size: tuple[int, int, int]) -> str:
    return f"{shape_text(input_shape)} -> {shape_text(output_size)}"


def grid_shape_text(
    input_shape: tuple[int, int, int, int, int],
    grid_shape: tuple[int, int, int, int, int],
) -> str:
    spatial = grid_shape[1:4]
    return f"{shape_text(input_shape)} -> {shape_text(spatial)}"


def dtype_name(dtype: Any) -> str:
    return np.dtype(dtype).name


def seed_from_text(text: str) -> int:
    total = 0
    for char in text:
        total = (total * 131 + ord(char)) % (2**32)
    return total


def generate_resample_input(shape: tuple[int, ...], dtype: Any, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if np.dtype(dtype) == np.uint8:
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    if np.dtype(dtype) == np.int16:
        return rng.integers(-32768, 32768, size=shape, dtype=np.int16)
    return rng.standard_normal(size=shape, dtype=np.float32)


def generate_grid_case_input(case: GridSampleCase) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(case.seed)
    input_data = rng.standard_normal(size=case.input_shape, dtype=np.float32)
    if case.grid_kind == "identity":
        depth, height, width = case.grid_shape[1:4]
        z = np.linspace(-1.0, 1.0, depth, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        base_grid = np.stack([xx, yy, zz], axis=-1)
        grid = np.broadcast_to(base_grid, case.grid_shape).copy()
    else:
        grid = rng.uniform(case.grid_range[0], case.grid_range[1], size=case.grid_shape).astype(
            np.float32
        )
    return input_data, grid


def scipy_cubic_reference(
    data: np.ndarray,
    output_size: tuple[int, int, int],
    align_corners: bool,
) -> np.ndarray:
    factors = tuple(out_dim / in_dim for out_dim, in_dim in zip(output_size, data.shape[-3:]))
    return zoom(
        data.astype(np.float64),
        factors,
        order=3,
        mode="reflect",
        grid_mode=not align_corners,
    ).astype(np.float32)


def time_callable(fn: Callable[[], np.ndarray], target_ms: float) -> Measurement:
    for _ in range(WARMUP_RUNS):
        fn()

    start = time.perf_counter()
    output = fn()
    sample_ms = (time.perf_counter() - start) * 1000.0

    if sample_ms <= 0.0:
        runs = MAX_RUNS
    else:
        runs = int(round(target_ms / sample_ms))
    runs = max(MIN_RUNS, min(MAX_RUNS, runs))

    times = [sample_ms]
    for _ in range(runs - 1):
        start = time.perf_counter()
        output = fn()
        times.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.fmean(times)
    std_ms = statistics.pstdev(times) if len(times) > 1 else 0.0
    return Measurement(mean_ms=mean_ms, std_ms=std_ms, runs=runs, output=output)


def max_abs_error(reference_output: np.ndarray, candidate_output: np.ndarray) -> float:
    return float(
        np.max(np.abs(reference_output.astype(np.float64) - candidate_output.astype(np.float64)))
    )


def format_time(mean_ms: float | None, std_ms: float | None) -> str:
    if mean_ms is None or std_ms is None:
        return "n/a"
    return f"{mean_ms:8.2f} +- {std_ms:5.2f}"


def format_speedup(speedup: float | None) -> str:
    if speedup is None:
        return "n/a"
    return f"{speedup:6.2f}x"


def format_error(error: float | None) -> str:
    if error is None:
        return "n/a"
    return f"{error:9.2e}"


def configure_live_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass


def live_print(*args: object, **kwargs: object) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def print_rule(char: str = "=") -> None:
    live_print(char * LINE_WIDTH)


def print_header(title: str, subtitle: str | None = None) -> None:
    print_rule("=")
    live_print(title)
    if subtitle:
        live_print(subtitle)
    print_rule("=")


def print_environment(profile: str, target_ms: float, threads: int) -> None:
    live_print(f"Profile          : {profile} ({target_ms:.0f} ms target per backend)")
    live_print(f"Warmup           : {WARMUP_RUNS} run")
    live_print(f"Adaptive repeats : {MIN_RUNS}-{MAX_RUNS} runs")
    live_print(f"Threads          : {threads}")
    live_print(f"Python           : {platform.python_version()}")
    live_print(f"NumPy            : {np.__version__}")
    live_print(f"PyTorch          : {'available' if TorchReference.available else 'missing'}")
    live_print(f"SciPy            : {'available' if zoom is not None else 'missing'}")
    live_print()
    live_print("Speedup is reference_time / volresample_time. Values above 1.0x favor volresample.")
    live_print("Error is max absolute difference between outputs.")


def print_progress_start(
    section: str,
    index: int,
    total: int,
    case_label: str,
    config: str,
    shape: str,
) -> None:
    live_print(f"[{section} {index}/{total}] {case_label} | {config} | {shape}")


def print_progress_done(
    reference_name: str,
    reference: Measurement,
    candidate: Measurement,
    speedup: float | None,
    error: float,
) -> None:
    speedup_text = format_speedup(speedup)
    live_print(
        f"  done: {reference_name} {reference.mean_ms:.2f} ms | "
        f"volresample {candidate.mean_ms:.2f} ms | {speedup_text} | err {error:.2e}"
    )


def print_table(title: str, rows: list[ComparisonRow]) -> None:
    live_print()
    live_print(title)
    print_rule("-")
    live_print(
        f"{'Case':<20} {'Config':<18} {'Shape':<31} {'Reference (ms)':>17} "
        f"{'volresample (ms)':>18} {'Speedup':>8} {'Max err':>11} {'Runs':>9}"
    )
    print_rule("-")

    for row in rows:
        live_print(
            f"{row.case:<20} {row.config:<18} {row.shape:<31} "
            f"{format_time(row.reference_ms, row.reference_std_ms):>17} "
            f"{format_time(row.volresample_ms, row.volresample_std_ms):>18} "
            f"{format_speedup(row.speedup):>8} {format_error(row.max_error):>11} {row.runs:>9}"
        )


def summarize_section(title: str, rows: list[ComparisonRow]) -> None:
    if not rows:
        return

    speedups = [row.speedup for row in rows if row.speedup is not None]
    errors = [row.max_error for row in rows if row.max_error is not None]
    best_row = max(
        (row for row in rows if row.speedup is not None), key=lambda row: row.speedup, default=None
    )

    live_print()
    live_print(title)
    print_rule("-")
    if speedups:
        live_print(f"Average speedup : {statistics.fmean(speedups):.2f}x")
        live_print(f"Median speedup  : {statistics.median(speedups):.2f}x")
    if best_row is not None:
        live_print(f"Best case       : {best_row.case} ({best_row.speedup:.2f}x)")
    if errors:
        live_print(f"Worst max error : {max(errors):.2e}")


def benchmark_resample_against_torch(
    cases: list[ResampleCase], target_ms: float
) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    if not TorchReference.available:
        live_print("Skipping resample vs PyTorch: PyTorch is not available.")
        return rows

    for index, case in enumerate(cases, start=1):
        data = generate_resample_input(
            case.input_shape,
            case.dtype,
            case.seed + seed_from_text(case.label),
        )
        output_size = case.output_size
        mode = case.mode
        align_corners = case.align_corners

        def run_reference(
            data: np.ndarray = data,
            output_size: tuple[int, int, int] = output_size,
            mode: str = mode,
            align_corners: bool = align_corners,
        ) -> np.ndarray:
            return TorchReference.resample(
                data,
                output_size,
                mode=mode,
                align_corners=align_corners,
            )

        def run_volresample(
            data: np.ndarray = data,
            output_size: tuple[int, int, int] = output_size,
            mode: str = mode,
            align_corners: bool = align_corners,
        ) -> np.ndarray:
            return volresample.resample(
                data,
                output_size,
                mode=mode,
                align_corners=align_corners,
            )

        config = case.mode
        if case.align_corners:
            config += ", align"
        if np.dtype(case.dtype) != np.float32:
            config += f", {dtype_name(case.dtype)}"
        shape = resample_shape_text(case.input_shape, case.output_size)

        print_progress_start("resample", index, len(cases), case.label, config, shape)
        reference = time_callable(run_reference, target_ms)
        candidate = time_callable(run_volresample, target_ms)
        error = max_abs_error(reference.output, candidate.output)

        rows.append(
            ComparisonRow(
                case=case.label,
                config=config,
                shape=shape,
                reference_name="PyTorch",
                reference_ms=reference.mean_ms,
                reference_std_ms=reference.std_ms,
                volresample_ms=candidate.mean_ms,
                volresample_std_ms=candidate.std_ms,
                speedup=reference.mean_ms / candidate.mean_ms if candidate.mean_ms > 0 else None,
                max_error=error,
                runs=f"{reference.runs}/{candidate.runs}",
            )
        )
        print_progress_done(
            "PyTorch",
            reference,
            candidate,
            reference.mean_ms / candidate.mean_ms if candidate.mean_ms > 0 else None,
            error,
        )

    return rows


def benchmark_cubic_against_scipy(
    cases: list[ResampleCase], target_ms: float
) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    if zoom is None:
        live_print("Skipping cubic vs SciPy: SciPy is not available.")
        return rows

    for index, case in enumerate(cases, start=1):
        data = generate_resample_input(
            case.input_shape,
            np.float32,
            case.seed + seed_from_text(case.label),
        )
        output_size = case.output_size
        align_corners = case.align_corners

        def run_reference(
            data: np.ndarray = data,
            output_size: tuple[int, int, int] = output_size,
            align_corners: bool = align_corners,
        ) -> np.ndarray:
            return scipy_cubic_reference(data, output_size, align_corners=align_corners)

        def run_volresample(
            data: np.ndarray = data,
            output_size: tuple[int, int, int] = output_size,
            align_corners: bool = align_corners,
        ) -> np.ndarray:
            return volresample.resample(
                data,
                output_size,
                mode="cubic",
                align_corners=align_corners,
            )

        config = "cubic"
        config += ", align" if case.align_corners else ", no-align"
        shape = resample_shape_text(case.input_shape, case.output_size)

        print_progress_start("cubic", index, len(cases), case.label, config, shape)
        reference = time_callable(run_reference, target_ms)
        candidate = time_callable(run_volresample, target_ms)
        error = max_abs_error(reference.output, candidate.output)

        rows.append(
            ComparisonRow(
                case=case.label,
                config=config,
                shape=shape,
                reference_name="SciPy",
                reference_ms=reference.mean_ms,
                reference_std_ms=reference.std_ms,
                volresample_ms=candidate.mean_ms,
                volresample_std_ms=candidate.std_ms,
                speedup=reference.mean_ms / candidate.mean_ms if candidate.mean_ms > 0 else None,
                max_error=error,
                runs=f"{reference.runs}/{candidate.runs}",
            )
        )
        print_progress_done(
            "SciPy",
            reference,
            candidate,
            reference.mean_ms / candidate.mean_ms if candidate.mean_ms > 0 else None,
            error,
        )

    return rows


def benchmark_grid_sample_against_torch(
    cases: list[GridSampleCase],
    target_ms: float,
) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    if not TorchReference.available:
        live_print("Skipping grid_sample vs PyTorch: PyTorch is not available.")
        return rows

    for index, case in enumerate(cases, start=1):
        input_data, grid = generate_grid_case_input(case)
        mode = case.mode
        padding_mode = case.padding_mode

        def run_reference(
            input_data: np.ndarray = input_data,
            grid: np.ndarray = grid,
            mode: str = mode,
            padding_mode: str = padding_mode,
        ) -> np.ndarray:
            return TorchReference.grid_sample(
                input_data,
                grid,
                mode=mode,
                padding_mode=padding_mode,
            )

        def run_volresample(
            input_data: np.ndarray = input_data,
            grid: np.ndarray = grid,
            mode: str = mode,
            padding_mode: str = padding_mode,
        ) -> np.ndarray:
            return volresample.grid_sample(
                input_data,
                grid,
                mode=mode,
                padding_mode=padding_mode,
            )

        config = f"{case.mode}, {case.padding_mode}"
        shape = grid_shape_text(case.input_shape, case.grid_shape)

        print_progress_start("grid", index, len(cases), case.label, config, shape)
        reference = time_callable(run_reference, target_ms)
        candidate = time_callable(run_volresample, target_ms)
        error = max_abs_error(reference.output, candidate.output)

        rows.append(
            ComparisonRow(
                case=case.label,
                config=config,
                shape=shape,
                reference_name="PyTorch",
                reference_ms=reference.mean_ms,
                reference_std_ms=reference.std_ms,
                volresample_ms=candidate.mean_ms,
                volresample_std_ms=candidate.std_ms,
                speedup=reference.mean_ms / candidate.mean_ms if candidate.mean_ms > 0 else None,
                max_error=error,
                runs=f"{reference.runs}/{candidate.runs}",
            )
        )
        print_progress_done(
            "PyTorch",
            reference,
            candidate,
            reference.mean_ms / candidate.mean_ms if candidate.mean_ms > 0 else None,
            error,
        )

    return rows


def build_resample_torch_cases() -> list[ResampleCase]:
    return [
        ResampleCase("3D nearest", (128, 128, 128), (64, 64, 64), "nearest"),
        ResampleCase("3D nearest u8", (128, 128, 128), (64, 64, 64), "nearest", np.uint8),
        ResampleCase("3D nearest i16", (128, 128, 128), (64, 64, 64), "nearest", np.int16),
        ResampleCase("3D nearest 512->256", (512, 512, 512), (256, 256, 256), "nearest"),
        ResampleCase("3D linear ds", (128, 128, 128), (64, 64, 64), "linear"),
        ResampleCase("3D linear up", (96, 96, 96), (144, 144, 144), "linear", align_corners=True),
        ResampleCase("3D linear 512->256", (512, 512, 512), (256, 256, 256), "linear"),
        ResampleCase("3D area", (160, 160, 160), (80, 80, 80), "area"),
        ResampleCase("3D area 512->64", (512, 512, 512), (64, 64, 64), "area"),
        ResampleCase("4D linear", (4, 96, 96, 96), (64, 64, 64), "linear"),
        ResampleCase("5D linear", (2, 4, 80, 80, 80), (48, 48, 48), "linear"),
    ]


def build_cubic_scipy_cases() -> list[ResampleCase]:
    return [
        ResampleCase("3D cubic ds", (128, 128, 128), (64, 64, 64), "cubic", reference="scipy"),
        ResampleCase(
            "3D cubic mixed",
            (96, 128, 80),
            (64, 160, 48),
            "cubic",
            align_corners=True,
            reference="scipy",
        ),
    ]


def build_grid_sample_cases() -> list[GridSampleCase]:
    return [
        GridSampleCase(
            "grid linear",
            (1, 2, 96, 96, 96),
            (1, 80, 80, 80, 3),
            "linear",
            "zeros",
            grid_range=(-1.15, 1.15),
        ),
        GridSampleCase(
            "grid nearest",
            (1, 2, 96, 96, 96),
            (1, 80, 80, 80, 3),
            "nearest",
            "zeros",
            "identity",
            (-0.95, 0.95),
        ),
        GridSampleCase(
            "grid reflect",
            (1, 2, 80, 96, 64),
            (1, 72, 88, 56, 3),
            "linear",
            "reflection",
            grid_range=(-1.50, 1.50),
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Professional benchmark suite for volresample.")
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Number of CPU threads to use for volresample and PyTorch (0 = current/default).",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_TARGET_MS),
        default="default",
        help="Benchmark profile. 'default' is curated for roughly 30-60 seconds.",
    )
    parser.add_argument(
        "--target-ms",
        type=float,
        default=None,
        help="Override target milliseconds per backend measurement.",
    )
    return parser.parse_args()


def configure_threads(requested_threads: int) -> int:
    if requested_threads > 0:
        volresample.set_num_threads(requested_threads)
    threads = volresample.get_num_threads()
    if torch is not None:
        torch.set_num_threads(threads)
    return threads


def main() -> int:
    configure_live_output()
    args = parse_args()
    target_ms = args.target_ms if args.target_ms is not None else PROFILE_TARGET_MS[args.profile]
    threads = configure_threads(args.threads)

    start = time.perf_counter()

    print_header(
        "volresample benchmark",
        "Compact, reference-checked CPU benchmark for resample() and grid_sample().",
    )
    print_environment(args.profile, target_ms, threads)

    resample_rows = benchmark_resample_against_torch(build_resample_torch_cases(), target_ms)
    cubic_rows = benchmark_cubic_against_scipy(build_cubic_scipy_cases(), target_ms)
    grid_rows = benchmark_grid_sample_against_torch(build_grid_sample_cases(), target_ms)

    if resample_rows:
        print_table("resample() vs PyTorch", resample_rows)
        summarize_section("resample() summary", resample_rows)

    if cubic_rows:
        print_table("cubic vs SciPy", cubic_rows)
        summarize_section("cubic summary", cubic_rows)

    if grid_rows:
        print_table("grid_sample() vs PyTorch", grid_rows)
        summarize_section("grid_sample() summary", grid_rows)

    all_rows = resample_rows + cubic_rows + grid_rows
    elapsed = time.perf_counter() - start

    live_print()
    print_header("overall summary")
    if all_rows:
        speedups = [row.speedup for row in all_rows if row.speedup is not None]
        errors = [row.max_error for row in all_rows if row.max_error is not None]
        best_row = max(
            (row for row in all_rows if row.speedup is not None), key=lambda row: row.speedup
        )
        live_print(f"Rows benchmarked  : {len(all_rows)}")
        if speedups:
            live_print(f"Average speedup  : {statistics.fmean(speedups):.2f}x")
        if errors:
            live_print(f"Worst max error  : {max(errors):.2e}")
        live_print(f"Best speedup row : {best_row.case} ({best_row.speedup:.2f}x)")
    else:
        live_print("No rows were benchmarked. Install PyTorch and SciPy for the full suite.")

    live_print(f"Total runtime    : {elapsed:.1f} s")
    print_rule("=")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
