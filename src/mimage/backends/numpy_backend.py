import numpy as np

class NumpyBackend:
    name='numpy'
    @staticmethod
    def flip(data, dims):
        return np.flip(data, dims)

    @staticmethod
    def permute(data, dims):
        return np.transpose(data, axes=dims)

    @staticmethod
    def asarray(data):
        return np.asarray(data)

    @staticmethod
    def clone(data):
        return np.copy(data)
    
    @staticmethod
    def clip(data, min_val=None, max_val=None):
        """Clip array values to specified range.
        
        Args:
            data: Input array
            min_val: Minimum value (None means no lower bound)
            max_val: Maximum value (None means no upper bound)
            
        Returns:
            Clipped array
        """
        return np.clip(data, min_val, max_val)