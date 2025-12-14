"""
Runtime CPU feature detection and backend loading.

Automatically selects the best available backend based on CPU features.
"""

import platform
import os
import sys
from typing import Optional


def detect_cpu_features():
    """
    Detect available CPU instruction sets.
    
    Returns:
        dict: Dictionary with boolean flags for available features
    """
    features = {
        'avx512': False,
        'avx2': False,
        'arm_neon': False,
        'is_arm': False,
        'is_x86': False,
    }
    
    machine = platform.machine().lower()
    
    # Detect architecture
    if machine in ['arm64', 'aarch64', 'armv7l', 'armv8']:
        features['is_arm'] = True
        features['arm_neon'] = True  # Most modern ARM has NEON
    elif machine in ['x86_64', 'amd64', 'i386', 'i686']:
        features['is_x86'] = True
        
        # Detect x86 features
        if sys.platform == 'linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    features['avx2'] = 'avx2' in cpuinfo
                    features['avx512f'] = 'avx512f' in cpuinfo
            except:
                pass
        
        elif sys.platform == 'darwin':  # macOS
            try:
                import subprocess
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.features', 'machdep.cpu.leaf7_features'],
                    capture_output=True, text=True
                )
                cpuinfo = result.stdout.lower()
                features['avx2'] = 'avx2' in cpuinfo
                features['avx512f'] = 'avx512f' in cpuinfo
            except:
                pass
        
        elif sys.platform == 'win32':  # Windows
            try:
                import subprocess
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'caption'],
                    capture_output=True, text=True
                )
                # Basic detection - could be improved with cpuid
                cpuinfo = result.stdout.lower()
                # This is simplified - proper Windows detection needs more work
                features['avx2'] = True  # Assume modern Windows = AVX2
            except:
                pass
    
    return features


def get_backend_priority():
    """
    Get list of backend names in priority order based on CPU features.
    
    Note: These backends are only available if built on the target architecture
    or cross-compiled. On x86_64, only x86 backends will be built.
    On ARM, only ARM backends will be built.
    
    Returns:
        list: Backend names in order of preference
    """
    features = detect_cpu_features()
    backends = []
    
    if features['is_x86']:
        if features.get('avx512f'):
            backends.append('resampling_cython_avx512')  # Only built on x86_64
        if features['avx2']:
            backends.append('resampling_cython_avx2')    # Only built on x86_64
    
    if features['is_arm']:
        backends.append('resampling_cython_arm')         # Only built on ARM
    
    # Always have generic as fallback
    backends.append('resampling_cython_generic')
    
    return backends


def load_best_backend():
    """
    Load the best available Cython backend.
    
    For now, just use the standard compiled backend which is optimized for the build machine.
    In the future, this can be extended to support multiple architecture-specific builds.
    
    Returns:
        module: The loaded backend module
        
    Raises:
        ImportError: If no backend can be loaded
    """
    # Try the standard backend first
    try:
        from . import resampling_cython
        print(f"Loaded backend: resampling_cython (optimized for build machine)")
        return resampling_cython
    except ImportError:
        pass
    
    # Try architecture-specific builds (if available)
    backends = get_backend_priority()
    for backend_name in backends:
        try:
            import importlib
            module_path = f'mimage.backends.resampling.{backend_name}'
            module = importlib.import_module(module_path)
            print(f"Loaded backend: {backend_name}")
            return module
        except (ImportError, AttributeError) as e:
            continue
    
    # If we get here, no backend loaded
    raise ImportError(
        "No Cython backend available. Please build the extension with:\n"
        "  cd /home/johannes/projects/test\n"
        "  python setup_cython.py build_ext --inplace"
    )


# For convenience
def get_resample_function():
    """Get the resample function from the best available backend."""
    backend = load_best_backend()
    return backend.resample


if __name__ == '__main__':
    # Test detection
    print("CPU Feature Detection:")
    print("=" * 50)
    features = detect_cpu_features()
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    print("\nBackend Priority:")
    print("=" * 50)
    backends = get_backend_priority()
    for i, backend in enumerate(backends, 1):
        print(f"  {i}. {backend}")
    
    print("\nAttempting to load backend:")
    print("=" * 50)
    try:
        backend = load_best_backend()
        print(f"✓ Successfully loaded: {backend.__name__}")
    except ImportError as e:
        print(f"✗ Failed: {e}")
