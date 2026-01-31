
def do_it():

    import os
    for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
        print(k, os.environ.get(k))
    import sys
    print(sys.executable)
    print(sys.version)
    from mimage.backends.resampling.resampling_cython_wrapper import ResamplingCythonBackend

    import torch
    from mimage.backends.resampling.resampling_torch import ResamplingTorchBackend
    from mimage.backends.resampling.resampling_cython_wrapper import ResamplingCythonBackend

    import numpy as np
    import time
    from typing import Any, Tuple, Dict

    torch.set_num_threads(1)

    F = torch.nn.functional
    # data_t = torch.randint(0, 10, (1,1,512,512,512), dtype=torch.uint8)
    # data_tnp = data_t[0,0].numpy()
    rng = np.random.RandomState(42)
    shape = (256, 256, 256)
    data_tnp = rng.randint(0, 256, shape).astype(np.uint8)
    data_t = torch.from_numpy(data_tnp).unsqueeze(0).unsqueeze(0)
    # for i in range(20):
    #     resampled_t = F.interpolate(data_t, size=(512,512,512), mode='trilinear')

    a = resampled_t = F.interpolate(data_t, size=(256,256,256), mode='nearest-exact')
    b = ResamplingCythonBackend.resample(data_t[0,0].numpy(), size=(256,256,256), mode='nearest', parallel_threads=4)
    (a[0,0] == b).all()

    def benchmark_implementation(
        backend: Any,
        data: np.ndarray,
        size: Tuple[int, int, int],
        mode: str,
        n_warmup: int = 1,
        n_iterations: int = 50,
        parallel_threads: int = 0,
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
        if backend.__name__ == "ResamplingTorchBackend":
            torch.set_num_threads(parallel_threads)
        # Warmup
        for _ in range(n_warmup):
            if backend.__name__ == "ResamplingTorchBackend":
                result = backend.resample(data, size, mode=mode)
            else:
                result = backend.resample(data, size, mode=mode, parallel_threads=parallel_threads)
        
        # Benchmark
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            if backend.__name__ == "ResamplingTorchBackend":
                result = backend.resample(data, size, mode=mode)
            else:
                result = backend.resample(data, size, mode=mode, parallel_threads=parallel_threads)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        if mode == 'nearest' and data.dtype in (np.uint8, torch.uint8):
            print(backend.__name__, data.shape, size, mode, data.dtype, times)    

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

    benchmark_implementation(ResamplingCythonBackend, data_tnp, size=(512,512,512), mode='nearest', parallel_threads=1)[0]
    print('blibb')
    benchmark_implementation(ResamplingTorchBackend, data_t[0,0], size=(512,512,512), mode='nearest', parallel_threads=1)[0]

if __name__ == "__main__":
    do_it()