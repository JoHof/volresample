"""Global thread configuration for volresample.

This module provides global thread settings similar to torch.set_num_threads().
"""

import os

# Global thread count - default will be set on first access
_num_threads = None


def _get_default_threads():
    """Get default number of threads: min(cpu_count, 4)."""
    try:
        cpu_count = os.cpu_count() or 1
    except Exception:
        cpu_count = 1
    return min(cpu_count, 4)


def set_num_threads(num_threads: int) -> None:
    """Set the number of threads used for parallel operations.
    
    This affects all volresample operations (resample, grid_sample).
    
    Args:
        num_threads: Number of threads to use. Must be >= 1.
        
    Examples:
        >>> import volresample
        >>> volresample.set_num_threads(4)
        >>> volresample.get_num_threads()
        4
    """
    global _num_threads
    if not isinstance(num_threads, int) or num_threads < 1:
        raise ValueError(f"num_threads must be a positive integer, got {num_threads}")
    _num_threads = num_threads


def get_num_threads() -> int:
    """Get the number of threads used for parallel operations.
    
    Returns:
        The current number of threads.
        
    Examples:
        >>> import volresample
        >>> volresample.get_num_threads()  # Returns default: min(cpu_count, 4)
        4
    """
    global _num_threads
    if _num_threads is None:
        _num_threads = _get_default_threads()
    return _num_threads
