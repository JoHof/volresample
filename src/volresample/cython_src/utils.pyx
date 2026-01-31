# Utility functions for resampling (included via main file)

cdef inline float clip(float val, float min_val, float max_val) nogil:
    """Clip value to [min_val, max_val]."""
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val
