#!/usr/bin/env python3
"""Comprehensive benchmark script comparing Cython and PyTorch resampling implementations.

Tests all interpolation modes (nearest, linear, area) with repeated execution and
standard deviation reporting.
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Any, Optional
import sys
import os

# Benchmark configuration
N_WARMUP_ITERATIONS = 2
N_BENCHMARK_ITERATIONS = 10

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mimage.backends.resampling.resampling_torch import ResamplingTorchBackend
from mimage.backends.resampling.resampling_cython_wrapper import ResamplingCythonBackend


def generate_test_data(shape: Tuple[int, ...], seed: int = 42) -> np.ndarray:
    """Generate random test data with specified shape."""
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32)


def benchmark_implementation(
    backend: Any,
    data: np.ndarray,
    size: Tuple[int, int, int],
    mode: str,
    n_warmup: int = N_WARMUP_ITERATIONS,
    n_iterations: int = N_BENCHMARK_ITERATIONS,
) -> Tuple[float, float, np.ndarray]:
    """Benchmark a resampling implementation with repeated execution.
    
    Args:
        backend: Backend class with resample method
        data: Input data
        size: Target size
        mode: Interpolation mode ('nearest', 'linear', or 'area')
        n_warmup: Number of warmup iterations (default from global config)
        n_iterations: Number of timed iterations (default from global config)
        
    Returns:
        Tuple of (mean_time_ms, std_time_ms, output_array)
    """
    # Warmup
    for _ in range(n_warmup):
        result = backend.resample(data, size, mode=mode)
    
    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = backend.resample(data, size, mode=mode)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return mean_time, std_time, result


def analyze_differences(torch_result: np.ndarray, cython_result: np.ndarray) -> Dict[str, float]:
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
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'max_rel_error': max_rel_error,
        'mean_rel_error': mean_rel_error,
        'max_diff_location': max_idx,
        'torch_value_at_max': torch_result[max_idx],
        'cython_value_at_max': cython_result[max_idx],
    }


def run_benchmark(
    input_shape: Tuple[int, ...],
    output_size: Tuple[int, int, int],
    test_name: str,
    mode: str = "linear",
) -> Optional[Dict[str, any]]:
    """Run benchmark for a specific test case.
    
    Args:
        input_shape: Shape of input data (3D or 4D)
        output_size: Target spatial dimensions
        test_name: Name for this test
        mode: Interpolation mode ('nearest', 'linear', or 'area')
        
    Returns:
        Dictionary with benchmark results or None if error
    """
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"  Mode: {mode}")
    print(f"  Input: {input_shape} → Output: {output_size}")
    print(f"{'='*80}")
    
    # Generate test data
    data = generate_test_data(input_shape)
    
    # Benchmark Cython
    print("[Cython] ", end=" ")
    try:
        cython_mean, cython_std, cython_result = benchmark_implementation(
            ResamplingCythonBackend, data, output_size, mode
        )
        print(f"{cython_mean:.2f} ms (±{cython_std:.2f} ms)")
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    time.sleep(0.1)  # Small delay to avoid any interference

    # Benchmark PyTorch
    print("\n[PyTorch]", end=" ")
    try:
        torch_mean, torch_std, torch_result = benchmark_implementation(
            ResamplingTorchBackend, data, output_size, mode
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
    if diff_stats['max_diff'] > 1e-5:
        print(f"  ✗ FAIL: Results differ by more than tolerance (1e-05)")
        print(f"    Location of max diff: {diff_stats['max_diff_location']}")
        print(f"    PyTorch value: {diff_stats['torch_value_at_max']:.6f}")
        print(f"    Cython value:  {diff_stats['cython_value_at_max']:.6f}")
    else:
        print(f"  ✓ PASS: Results match within tolerance (1e-05)")
    
    # Speedup
    speedup = torch_mean / cython_mean
    print(f"\n[Result]  Speedup: {speedup:.2f}x")
    
    return {
        'test_name': test_name,
        'mode': mode,
        'input_shape': input_shape,
        'output_size': output_size,
        'torch_mean': torch_mean,
        'torch_std': torch_std,
        'cython_mean': cython_mean,
        'cython_std': cython_std,
        'speedup': speedup,
        'max_diff': diff_stats['max_diff'],
        'mean_diff': diff_stats['mean_diff'],
        'max_rel_error': diff_stats['max_rel_error'],
        'passed': diff_stats['max_diff'] < 1e-5,
    }


def main():
    """Run comprehensive benchmark suite for all interpolation modes."""
    print("\n" + "="*80)
    print("COMPREHENSIVE RESAMPLING BENCHMARK - ALL MODES")
    print("Comparing PyTorch vs Cython implementations")
    print("="*80)
    
    # Show benchmark configuration
    print("\n[Benchmark Configuration]")
    print(f"  Warmup iterations: {N_WARMUP_ITERATIONS}")
    print(f"  Benchmark iterations: {N_BENCHMARK_ITERATIONS}")
    print(f"  Reporting: mean ± std deviation")
    
    # Check availability
    print("\n[Backend Availability]")
    print(f"  PyTorch: {ResamplingTorchBackend.available}")
    print(f"  Cython:  {ResamplingCythonBackend.available}")
    
    if not ResamplingTorchBackend.available:
        print("\nERROR: PyTorch backend not available!")
        return
    
    if not ResamplingCythonBackend.available:
        print("\nERROR: Cython backend not available!")
        print("Build it with: cd src/mimage/backends/resampling && python setup_cython.py build_ext --inplace")
        return
    
    # Define all test cases
    test_cases = [
        # NEAREST mode tests
        ((64, 64, 64), (32, 32, 32), "3D - Nearest - Basic 64→32", "nearest"),
        ((4, 64, 64, 64), (32, 32, 32), "4D - Nearest - 4 channels 64→32", "nearest"),
        ((128, 128, 128), (64, 64, 64), "3D - Nearest - 128→64", "nearest"),
        ((8, 128, 128, 128), (64, 64, 64), "4D - Nearest - 8 channels 128→64", "nearest"),
        ((256, 256, 256), (128, 128, 128), "3D - Nearest - Large 256→128", "nearest"),
        ((512, 512, 512), (96, 96, 96), "3D - Nearest - Very large 512→96", "nearest"),
        ((512, 512, 512), (253, 253, 253), "3D - Nearest - 512→253", "nearest"),
        
        # LINEAR mode tests
        ((64, 64, 64), (32, 32, 32), "3D - Linear - Basic 64→32", "linear"),
        ((4, 64, 64, 64), (32, 32, 32), "4D - Linear - 4 channels 64→32", "linear"),
        ((128, 128, 128), (64, 64, 64), "3D - Linear - 128→64", "linear"),
        ((8, 128, 128, 128), (64, 64, 64), "4D - Linear - 8 channels 128→64", "linear"),
        ((256, 256, 256), (128, 128, 128), "3D - Linear - Large 256→128", "linear"),
        ((512, 512, 512), (96, 96, 96), "3D - Linear - Very large 512→96", "linear"),
        ((512, 512, 512), (253, 253, 253), "3D - Linear - 512→253", "linear"),
        
        # AREA mode tests
        ((64, 64, 64), (32, 32, 32), "3D - Area - Basic 64→32", "area"),
        ((4, 64, 64, 64), (32, 32, 32), "4D - Area - 4 channels 64→32", "area"),
        ((128, 128, 128), (64, 64, 64), "3D - Area - 128→64", "area"),
        ((8, 128, 128, 128), (64, 64, 64), "4D - Area - 8 channels 128→64", "area"),
        ((256, 256, 256), (128, 128, 128), "3D - Area - Large 256→128", "area"),
        ((512, 512, 512), (96, 96, 96), "3D - Area - Very large 512→96", "area"),
        ((512, 512, 512), (253, 253, 253), "3D - Area - 512→253", "area"),
    ]
    
    results = []
    for input_shape, output_size, test_name, mode in test_cases:
        result = run_benchmark(input_shape, output_size, test_name, mode)
        if result is not None:
            results.append(result)
    
    # Print summary
    print("\n" + "="*120)
    print("SUMMARY OF ALL TESTS")
    print("="*120)
    print(f"{'Test Name':<40} {'Mode':<8} {'PyTorch (ms)':<18} {'Cython (ms)':<18} {'Speedup':<10} {'Status':<8}")
    print("-"*120)
    
    for r in results:
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        speedup_str = f"{r['speedup']:.2f}x"
        torch_str = f"{r['torch_mean']:>7.2f} ±{r['torch_std']:>5.2f}"
        cython_str = f"{r['cython_mean']:>7.2f} ±{r['cython_std']:>5.2f}"
        print(f"{r['test_name']:<40} {r['mode']:<8} {torch_str:<18} {cython_str:<18} {speedup_str:<10} {status:<8}")
    
    # Group statistics by mode
    print("\n" + "="*120)
    print("STATISTICS BY MODE")
    print("="*120)
    
    for mode in ['nearest', 'linear', 'area']:
        mode_results = [r for r in results if r['mode'] == mode]
        if mode_results:
            speedups = [r['speedup'] for r in mode_results]
            torch_times = [r['torch_mean'] for r in mode_results]
            cython_times = [r['cython_mean'] for r in mode_results]
            
            avg_speedup = np.mean(speedups)
            std_speedup = np.std(speedups)
            avg_torch = np.mean(torch_times)
            avg_cython = np.mean(cython_times)
            
            print(f"{mode.upper():>7}: Speedup {avg_speedup:.2f}x (±{std_speedup:.2f}x) | "
                  f"PyTorch avg: {avg_torch:>7.2f} ms | Cython avg: {avg_cython:>7.2f} ms | "
                  f"{len(mode_results)} tests")
    
    # Overall correctness summary
    print("\n" + "="*80)
    print("CORRECTNESS SUMMARY")
    print("="*80)
    
    tolerance = 1e-5
    all_passed = all(r['passed'] for r in results)
    num_passed = sum(1 for r in results if r['passed'])
    num_total = len(results)
    
    print(f"Tests passed: {num_passed}/{num_total}")
    
    if all_passed:
        print(f"✓ ALL TESTS PASSED (max diff < {tolerance})")
    else:
        print(f"✗ SOME TESTS FAILED (max diff >= {tolerance})")
        for r in results:
            if not r['passed']:
                print(f"  FAILED: {r['test_name']} ({r['mode']}) - max diff: {r['max_diff']:.6e}")


if __name__ == "__main__":
    main()
