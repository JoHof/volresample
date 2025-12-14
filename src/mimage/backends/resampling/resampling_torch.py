"""PyTorch backend for image resampling using torch.nn.functional.interpolate."""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Any, overload
import numpy as np

try:
    import torch
except ImportError:
    torch = None

if TYPE_CHECKING:
    import torch


class ResamplingTorchBackend:
    """PyTorch-based resampling backend using F.interpolate."""

    available = torch is not None

    @staticmethod
    @overload
    def resample(
        data: np.ndarray, size: Tuple[int, int, int], mode: str = "linear"
    ) -> np.ndarray:
        ...

    @staticmethod
    @overload
    def resample(
        data: "torch.Tensor", size: Tuple[int, int, int], mode: str = "linear"
    ) -> "torch.Tensor":
        ...

    @staticmethod
    def resample(
        data: Any,  # torch.Tensor or np.ndarray
        size: Tuple[int, int, int],
        mode: str = "linear",
    ) -> Any:
        """Resample a 3D volume to a new size using interpolation.

        Supports 3D (D, H, W) and 4D (C, D, H, W) inputs. For tensors with additional
        dimensions, the caller must iterate over non-spatial dimensions.

        Behavior:
        - Input is cast to float32 before interpolation.
        - If mode == 'nearest', the output is cast back to the original dtype.
        - If input is a numpy array, the output is a numpy array; otherwise a torch tensor.

        Args:
            data: Input tensor/array of shape (D, H, W) or (C, D, H, W).
            size: Target size (new_D, new_H, new_W) for spatial dimensions.
            mode: Interpolation mode: 'nearest', 'linear', or 'area'.
                  Note: 'nearest' uses 'nearest-exact' under the hood.

        Returns:
            Resampled array/tensor. Shape is (new_D, new_H, new_W) for 3D inputs and
            (C, new_D, new_H, new_W) for 4D inputs. For 'nearest', dtype matches the
            input dtype; for other modes the output is float32.

        Raises:
            ValueError: If mode is not supported or data is not 3D/4D.
            ImportError: If PyTorch is not available.
        """
        if torch is None:
            raise ImportError("PyTorch is required for ResamplingTorchBackend")
        F = torch.nn.functional

        was_numpy = isinstance(data, np.ndarray)
        if was_numpy:
            orig_dtype = data.dtype
            data_t = torch.from_numpy(data)
            # Convert numpy dtype to torch dtype for later conversion
            torch_orig_dtype = data_t.dtype
        else:
            if not isinstance(data, torch.Tensor):
                raise ValueError("data must be a numpy.ndarray or torch.Tensor")
            orig_dtype = data.dtype
            torch_orig_dtype = data.dtype
            data_t = data

        if data_t.ndim not in (3, 4):
            raise ValueError(
                f"Data must be 3D or 4D (got {data_t.ndim}). "
                "Supported shapes: (D,H,W) or (C,D,H,W)."
            )

        mode_map = {
            "nearest": "nearest-exact",
            "linear": "trilinear",
            "area": "area",
        }
        if mode not in mode_map:
            raise ValueError("Unsupported interpolation mode. Use 'nearest', 'linear', or 'area'.")

        torch_mode = mode_map[mode]
        orig_ndim = data_t.ndim

        # Prepare shape for interpolate: (N, C, D, H, W)
        if orig_ndim == 3:
            data_t = data_t.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        else:
            data_t = data_t.unsqueeze(0)  # (1,C,D,H,W)

        # Always interpolate in float32 for numerical stability and API requirements.
        data_t = data_t.to(dtype=torch.float32)

        # Interpolate. Only pass align_corners for linear/trilinear modes.
        resampled_t = F.interpolate(data_t, size=size, mode=torch_mode)

        # Remove batch dimension; keep channel dimension for 4D inputs.
        out_t = resampled_t.squeeze(0)  # (1,*,D,H,W) -> (*,D,H,W)
        if orig_ndim == 3:
            out_t = out_t.squeeze(0)  # (1,D,H,W) -> (D,H,W)

        # If nearest, cast back to original dtype.
        if mode == "nearest":
            out_t = out_t.to(dtype=torch_orig_dtype)

        if was_numpy:
            return out_t.detach().cpu().numpy()

        return out_t
