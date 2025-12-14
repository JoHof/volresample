from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
try:
    import torch
except ImportError:
    torch = None

if TYPE_CHECKING:
    import torch  

class TensorTorchBackend:
    name = 'torch'
    @staticmethod
    def flip(data: "torch.Tensor", dims: list[int]) -> "torch.Tensor":
        return torch.flip(data, dims)

    @staticmethod
    def permute(data: "torch.Tensor", dims: list[int]) -> "torch.Tensor":
        return data.permute(*dims)

    @staticmethod
    def asarray(data: "torch.Tensor") -> np.ndarray:
        return torch.as_tensor(data)
    
    @staticmethod
    def clone(data: "torch.Tensor") -> "torch.Tensor":
        return data.clone()
    
    @staticmethod
    def clip(data: "torch.Tensor", min_val=None, max_val=None) -> "torch.Tensor":
        """Clip tensor values to specified range.
        
        Args:
            data: Input tensor
            min_val: Minimum value (None means no lower bound)
            max_val: Maximum value (None means no upper bound)
            
        Returns:
            Clipped tensor
        """
        return torch.clamp(data, min=min_val, max=max_val)
    
