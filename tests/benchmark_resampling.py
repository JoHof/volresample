#!/usr/bin/env python3
"""Benchmark script comparing volresample (Cython) and PyTorch implementations."""

import argparse
import time
from typing import Optional

import numpy as np

# Benchmark configuration
N_WARMUP_ITERATIONS = 1

import torch
from torch_reference import TorchReference

import volresample


def generate_test_data(
    shape: tuple[int, ...], seed: int = 42, dtype: type = np.float32
) -> np.ndarray:
    """Generate random test data with specified shape and dtype."""
    rng = np.random.RandomState(seed)

    if dtype == np.uint8:
        # Random uint8 values (0-255)
        return rng.randint(0, 256, shape).astype(np.uint8)
    elif dtype == np.int16:
        # Random int16 values (-32768 to 32767)
        return rng.randint(-32768, 32767, shape).astype(np.int16)
    else:
        # float32 normal distribution
        return rng.randn(*shape).astype(np.float32)


def benchmark_implementation(
    resample_func,
    data: np.ndarray,
    size: tuple[int, int, int],
    mode: str,
    n_warmup: int = 1,
    n_iterations: int = 50,
) -> tuple[float, float, np.ndarray]:
    """Benchmark a resampling implementation with repeated execution.

    Args:
        backend_name: Name of backend ('volresample' or 'torch')
        resample_func: Resample function to call
        data: Input data
        size: Target size
        mode: Interpolation mode ('nearest', 'linear', or 'area')
        n_warmup: Number of warmup iterations
        n_iterations: Number of timed iterations

    Returns:
        Tuple of (mean_time_ms, std_time_ms, output_array)
    """
    # Warmup
    for _ in range(n_warmup):
        result = resample_func(data, size, mode=mode)

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = resample_func(data, size, mode=mode)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = np.mean(times)
    std_time = np.std(times)

    return mean_time, std_time, result


def analyze_differences(torch_result: np.ndarray, cython_result: np.ndarray) -> dict[str, float]:
    """Analyze numerical differences between two results.

    Returns dict with max_diff, mean_diff, max_rel_error statistics and details about
    where differences occur.
    """
    diff = np.abs(torch_result - cython_result)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Compute relative error
    torch_magnitude = np.abs(torch_result)
    mask = torch_magnitude > 1e-6
    if np.any(mask):
        rel_errors = diff[mask] / torch_magnitude[mask]
        max_rel_error = np.max(rel_errors)
        mean_rel_error = np.mean(rel_errors)
    else:
        max_rel_error = 0.0
        mean_rel_error = 0.0

    # Find location of maximum difference
    max_idx = np.unravel_index(np.argmax(diff), diff.shape)

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "max_rel_error": max_rel_error,
        "mean_rel_error": mean_rel_error,
        "max_diff_location": max_idx,
        "torch_value_at_max": torch_result[max_idx],
        "cython_value_at_max": cython_result[max_idx],
    }


def benchmark_grid_sample(
    grid_sample_func,
    input_data: np.ndarray,
    grid: np.ndarray,
    mode: str,
    padding_mode: str,
    n_warmup: int = 1,
    n_iterations: int = 10,
) -> tuple[float, float, np.ndarray]:
    """Benchmark a grid_sample implementation with repeated execution.

    Args:
        grid_sample_func: Grid sample function to call
        input_data: Input data (N, C, D, H, W)
        grid: Grid for sampling (N, D_out, H_out, W_out, 3)
        mode: Interpolation mode ('linear' or 'nearest')
        padding_mode: Padding mode ('zeros', 'border', 'reflection')
        n_warmup: Number of warmup iterations
        n_iterations: Number of timed iterations

    Returns:
        Tuple of (mean_time_ms, std_time_ms, output_array)
    """
    # Warmup
    for _ in range(n_warmup):
        result = grid_sample_func(input_data, grid, mode=mode, padding_mode=padding_mode)

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = grid_sample_func(input_data, grid, mode=mode, padding_mode=padding_mode)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = np.mean(times)
    std_time = np.std(times)

    return mean_time, std_time, result


def run_benchmark(
    input_shape: tuple[int, ...],
    output_size: tuple[int, int, int],
    test_name: str,
    mode: str = "linear",
    dtype: type = np.float32,
    n_iterations: int = 5,
) -> Optional[dict[str, any]]:
    """Run benchmark for a specific test case.

    Args:
        input_shape: Shape of input data (3D or 4D)
        output_size: Target spatial dimensions
        test_name: Name for this test
        mode: Interpolation mode ('nearest', 'linear', or 'area')
        dtype: Data type for test data (uint8, int16, or float32)

    Returns:
        Dictionary with benchmark results or None if error
    """
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"  Mode: {mode}, Dtype: {dtype.__name__}")
    print(f"  Input: {input_shape} → Output: {output_size}")
    print(f"{'='*80}")

    # Generate test data
    data = generate_test_data(input_shape, dtype=dtype)

    # Benchmark volresample
    print("[volresample]", end=" ")
    try:
        cython_mean, cython_std, cython_result = benchmark_implementation(
            volresample.resample, data, output_size, mode, n_iterations=n_iterations
        )
        print(f"{cython_mean:.2f} ms (±{cython_std:.2f} ms)")
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    time.sleep(0.1)  # Small delay to avoid any interference

    # Benchmark PyTorch
    print("\n[PyTorch]   ", end=" ")
    try:
        torch_mean, torch_std, torch_result = benchmark_implementation(
            TorchReference.resample, data, output_size, mode, n_iterations=n_iterations
        )
        print(f"{torch_mean:.2f} ms (±{torch_std:.2f} ms)")
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    # Analyze differences
    print("\n[Equality Check]")
    diff_stats = analyze_differences(torch_result, cython_result)

    print(f"  Max absolute difference: {diff_stats['max_diff']:.6e}")
    print(f"  Mean absolute difference: {diff_stats['mean_diff']:.6e}")
    print(f"  Max relative error: {diff_stats['max_rel_error']:.6e}")

    # Show where max difference occurs if significant
    if diff_stats["max_diff"] > 1e-4:
        print("  ✗ FAIL: Results differ by more than tolerance (1e-05)")
        print(f"    Location of max diff: {diff_stats['max_diff_location']}")
        print(f"    PyTorch value: {diff_stats['torch_value_at_max']:.6f}")
        print(f"    Cython value:  {diff_stats['cython_value_at_max']:.6f}")
    else:
        print("  ✓ PASS: Results match within tolerance (1e-05)")

    # Speedup
    speedup = torch_mean / cython_mean
    print(f"\n[Result]  Speedup: {speedup:.2f}x")

    return {
        "test_name": test_name,
        "mode": mode,
        "dtype": dtype.__name__ if dtype != np.float32 else "float32",
        "input_shape": input_shape,
        "output_size": output_size,
        "torch_mean": torch_mean,
        "torch_std": torch_std,
        "cython_mean": cython_mean,
        "cython_std": cython_std,
        "speedup": speedup,
        "max_diff": diff_stats["max_diff"],
        "mean_diff": diff_stats["mean_diff"],
        "max_rel_error": diff_stats["max_rel_error"],
        "passed": diff_stats["max_diff"] < 1e-4,
    }


def run_grid_sample_benchmark(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    test_name: str,
    mode: str = "linear",
    padding_mode: str = "zeros",
    n_iterations: int = 5,
) -> Optional[dict[str, any]]:
    """Run benchmark for a grid_sample test case.

    Args:
        input_shape: Shape of input data (N, C, D, H, W)
        output_shape: Shape of output (N, D_out, H_out, W_out, 3)
        test_name: Name for this test
        mode: Interpolation mode ('linear' or 'nearest')
        padding_mode: Padding mode ('zeros', 'border', 'reflection')

    Returns:
        Dictionary with benchmark results or None if error
    """
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"  Mode: {mode}, Padding: {padding_mode}")
    print(f"  Input: {input_shape} → Output: {output_shape}")
    print(f"{'='*80}")

    # Generate test data
    rng = np.random.RandomState(42)
    input_data = rng.randn(*input_shape).astype(np.float32)
    grid = rng.uniform(-1, 1, output_shape).astype(np.float32)

    # Benchmark volresample
    print("[volresample]", end=" ")
    try:
        cython_mean, cython_std, cython_result = benchmark_grid_sample(
            volresample.grid_sample, input_data, grid, mode, padding_mode, n_iterations=n_iterations
        )
        print(f"{cython_mean:.2f} ms (±{cython_std:.2f} ms)")
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    time.sleep(0.1)

    # Benchmark PyTorch
    print("\n[PyTorch]   ", end=" ")
    try:
        torch_mean, torch_std, torch_result = benchmark_grid_sample(
            TorchReference.grid_sample,
            input_data,
            grid,
            mode,
            padding_mode,
            n_iterations=n_iterations,
        )
        print(f"{torch_mean:.2f} ms (±{torch_std:.2f} ms)")
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    # Analyze differences
    print("\n[Equality Check]")
    diff_stats = analyze_differences(torch_result, cython_result)

    print(f"  Max absolute difference: {diff_stats['max_diff']:.6e}")
    print(f"  Mean absolute difference: {diff_stats['mean_diff']:.6e}")
    print(f"  Max relative error: {diff_stats['max_rel_error']:.6e}")

    if diff_stats["max_diff"] > 1e-4:
        print("  ✗ FAIL: Results differ by more than tolerance (1e-04)")
        print(f"    Location of max diff: {diff_stats['max_diff_location']}")
        print(f"    PyTorch value: {diff_stats['torch_value_at_max']:.6f}")
        print(f"    Cython value:  {diff_stats['cython_value_at_max']:.6f}")
    else:
        print("  ✓ PASS: Results match within tolerance (1e-04)")

    # Speedup
    speedup = torch_mean / cython_mean
    print(f"\n[Result]  Speedup: {speedup:.2f}x")

    return {
        "test_name": test_name,
        "mode": mode,
        "padding_mode": padding_mode,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "torch_mean": torch_mean,
        "torch_std": torch_std,
        "cython_mean": cython_mean,
        "cython_std": cython_std,
        "speedup": speedup,
        "max_diff": diff_stats["max_diff"],
        "passed": diff_stats["max_diff"] < 1e-4,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Cython and PyTorch resampling.")
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Number of threads (0=use default: min(cpu_count, 4))",
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of benchmark iterations (default=5)"
    )
    args = parser.parse_args()
    n_iterations = args.iterations

    # Set thread count for both backends
    if args.threads > 0:
        volresample.set_num_threads(args.threads)
        torch.set_num_threads(args.threads)
    else:
        # Use volresample default for both
        num_threads = volresample.get_num_threads()
        torch.set_num_threads(num_threads)

    """Run benchmark suite."""
    print("\n" + "=" * 80)
    print("VOLUME RESAMPLING BENCHMARK")
    print("Comparing volresample (Cython) vs PyTorch implementations")
    print("=" * 80)

    # Show benchmark configuration
    print("\n[Benchmark Configuration]")
    print(f"  Warmup iterations: {N_WARMUP_ITERATIONS}")
    print(f"  Benchmark iterations: {n_iterations}")
    print(f"  Threads: {volresample.get_num_threads()}")

    # Check availability
    print("\n[Backend Availability]")
    print(f"  PyTorch: {TorchReference.available}")
    print(f"  volresample: {volresample.resample is not None}")

    if not TorchReference.available:
        print("\nERROR: PyTorch not available!")
        return

    if volresample.resample is None:
        print("\nERROR: volresample not available! Build with 'pip install -e .'")
        return

    # Define reduced test cases
    test_cases = [
        # Small 3D - all modes
        ((64, 64, 64), (32, 32, 32), "3D Small 64→32 nearest", "nearest"),
        ((64, 64, 64), (32, 32, 32), "3D Small 64→32 linear", "linear"),
        ((64, 64, 64), (32, 32, 32), "3D Small 64→32 area", "area"),
        # Small 3D with multi-channel
        ((4, 64, 64, 64), (32, 32, 32), "4D Small (4ch) 64→32 linear", "linear"),
        # Small 5D with batch and multi-channel
        ((2, 4, 64, 64, 64), (32, 32, 32), "5D Small (2x4ch) 64→32 linear", "linear"),
        # Small 5D with larger batch size
        ((20, 4, 64, 64, 64), (32, 32, 32), "5D Small (20x4ch) 64→32 linear", "linear"),
        # Small with different scaling (not half)
        ((64, 64, 64), (48, 48, 48), "3D Small 64→48 linear", "linear"),
        # Large 512->256 - all modes
        ((512, 512, 512), (256, 256, 256), "3D Large 512→256 nearest", "nearest"),
        ((512, 512, 512), (256, 256, 256), "3D Large 512→256 linear", "linear"),
        ((512, 512, 512), (256, 256, 256), "3D Large 512→256 area", "area"),
        # Large with different dtypes
        ((512, 512, 512), (256, 256, 256), "3D Large 512→256 nearest uint8", "nearest", np.uint8),
        ((512, 512, 512), (256, 256, 256), "3D Large 512→256 nearest int16", "nearest", np.int16),
    ]

    print("\n" + "=" * 80)
    print("RESAMPLE BENCHMARKS")
    print("=" * 80)

    results = []
    for test_case in test_cases:
        input_shape, output_size, test_name, mode = test_case[:4]
        dtype = test_case[4] if len(test_case) > 4 else np.float32
        result = run_benchmark(
            input_shape, output_size, test_name, mode, dtype=dtype, n_iterations=n_iterations
        )
        if result is not None:
            results.append(result)

    # Grid sample benchmarks
    print("\n\n" + "=" * 80)
    print("GRID SAMPLE BENCHMARKS")
    print("=" * 80)

    grid_sample_tests = [
        # Small 3D grid sample - both modes
        (
            (1, 2, 32, 32, 32),
            (1, 24, 24, 24, 3),
            "Grid Sample 3D Small nearest",
            "nearest",
            "zeros",
        ),
        ((1, 2, 32, 32, 32), (1, 24, 24, 24, 3), "Grid Sample 3D Small linear", "linear", "zeros"),
        # Large 3D grid sample - both modes
        (
            (1, 2, 128, 128, 128),
            (1, 96, 96, 96, 3),
            "Grid Sample 3D Large nearest",
            "nearest",
            "zeros",
        ),
        (
            (1, 2, 128, 128, 128),
            (1, 96, 96, 96, 3),
            "Grid Sample 3D Large linear",
            "linear",
            "zeros",
        ),
        (
            (1, 2, 128, 128, 128),
            (1, 96, 96, 96, 3),
            "Grid Sample 3D Large linear",
            "linear",
            "reflection",
        ),
    ]

    grid_results = []
    for input_shape, output_shape, test_name, mode, padding_mode in grid_sample_tests:
        result = run_grid_sample_benchmark(
            input_shape, output_shape, test_name, mode, padding_mode, n_iterations=n_iterations
        )
        if result is not None:
            grid_results.append(result)

    # Print resample summary
    print("\n" + "=" * 120)
    print("RESAMPLE SUMMARY")
    print("=" * 120)
    print(
        f"{'Test Name':<30} {'Mode':<8} {'PyTorch (ms)':<18} {'volresample (ms)':<18} {'Speedup':<10} {'Status':<8}"
    )
    print("-" * 120)

    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        speedup_str = f"{r['speedup']:.2f}x"
        torch_str = f"{r['torch_mean']:>7.2f} ±{r['torch_std']:>5.2f}"
        cython_str = f"{r['cython_mean']:>7.2f} ±{r['cython_std']:>5.2f}"
        print(
            f"{r['test_name']:<30} {r['mode']:<8} {torch_str:<18} {cython_str:<18} {speedup_str:<10} {status:<8}"
        )

    # Print grid sample summary
    if grid_results:
        print("\n" + "=" * 120)
        print("GRID SAMPLE SUMMARY")
        print("=" * 120)
        print(
            f"{'Test Name':<30} {'Mode':<10} {'Padding':<10} {'PyTorch (ms)':<18} {'volresample (ms)':<18} {'Speedup':<10} {'Status':<8}"
        )
        print("-" * 120)

        for r in grid_results:
            status = "✓ PASS" if r["passed"] else "✗ FAIL"
            speedup_str = f"{r['speedup']:.2f}x"
            torch_str = f"{r['torch_mean']:>7.2f} ±{r['torch_std']:>5.2f}"
            cython_str = f"{r['cython_mean']:>7.2f} ±{r['cython_std']:>5.2f}"
            print(
                f"{r['test_name']:<30} {r['mode']:<10} {r['padding_mode']:<10} {torch_str:<18} {cython_str:<18} {speedup_str:<10} {status:<8}"
            )

    # Overall statistics
    all_results = results + grid_results
    if all_results:
        print("\n" + "=" * 80)
        print("OVERALL STATISTICS")
        print("=" * 80)

        speedups = [r["speedup"] for r in all_results]
        avg_speedup = np.mean(speedups)

        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Tests passed: {sum(1 for r in all_results if r['passed'])}/{len(all_results)}")

        if all(r["passed"] for r in all_results):
            print("✓ ALL TESTS PASSED")
        else:
            print("✗ SOME TESTS FAILED")


if __name__ == "__main__":
    main()
