"""PyTorch reference implementation for testing volresample."""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Any, overload
import numpy as np

try:
    import torch
except ImportError:
    torch = None

if TYPE_CHECKING:
    import torch


class TorchReference:
    """PyTorch-based reference implementation for testing."""

    available = torch is not None

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
        - For nearest mode with uint8 input: preserves uint8 dtype throughout
        - For other cases: input is cast to float32 before interpolation
        - If mode == 'nearest', the output is cast back to the original dtype (for non-uint8)
        - If input is a numpy array, the output is a numpy array; otherwise a torch tensor

        Args:
            data: Input tensor/array of shape (D, H, W) or (C, D, H, W).
                  Supported dtypes: uint8, int16, float32.
            size: Target size (new_D, new_H, new_W) for spatial dimensions.
            mode: Interpolation mode: 'nearest', 'linear', or 'area'.
                  Note: 'nearest' uses 'nearest-exact' under the hood.
                  For 'nearest' with uint8: stays uint8
                  For 'linear'/'area' with uint8: converts to float32

        Returns:
            Resampled array/tensor. Shape is (new_D, new_H, new_W) for 3D inputs and
            (C, new_D, new_H, new_W) for 4D inputs. 
            - For nearest with uint8 input: uint8 output
            - For nearest with other dtypes: preserves input dtype
            - For linear/area: float32 output

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

        # For uint8 nearest neighbor, keep in uint8 (nearest-exact supports it directly)
        # For other cases, convert to float32 for numerical stability
        if not (torch_orig_dtype == torch.uint8 and mode == "nearest"):
            data_t = data_t.to(dtype=torch.float32)

        # Interpolate. Only pass align_corners for linear/trilinear modes.
        resampled_t = F.interpolate(data_t, size=size, mode=torch_mode)

        # Remove batch dimension; keep channel dimension for 4D inputs.
        out_t = resampled_t.squeeze(0)  # (1,*,D,H,W) -> (*,D,H,W)
        if orig_ndim == 3:
            out_t = out_t.squeeze(0)  # (1,D,H,W) -> (D,H,W)

        # If nearest and not already uint8, cast back to original dtype.
        if mode == "nearest" and torch_orig_dtype != torch.uint8:
            out_t = out_t.to(dtype=torch_orig_dtype)

        if was_numpy:
            return out_t.detach().cpu().numpy()

        return out_t

    @staticmethod
    def grid_sample(
        input: Any,  # torch.Tensor or np.ndarray
        grid: Any,  # torch.Tensor or np.ndarray
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
    ) -> Any:
        """Apply grid sampling to input using the provided grid.

        Args:
            input: Input tensor/array of shape (N, C, D, H, W).
            grid: Sampling grid of shape (N, D_out, H_out, W_out, 3).
                  Values should be normalized to [-1, 1] range.
            mode: Interpolation mode: 'bilinear' or 'nearest'.
            padding_mode: Padding mode for outside grid values:
                         'zeros', 'border', or 'reflection'.
            align_corners: If True, corner pixels are aligned.
                          Default False to match grid_sample conventions.

        Returns:
            Sampled output of shape (N, C, D_out, H_out, W_out).
        """
        if torch is None:
            raise ImportError("PyTorch is required for ResamplingTorchBackend")
        F = torch.nn.functional

        was_numpy = isinstance(input, np.ndarray)
        if was_numpy:
            input_t = torch.from_numpy(input).float()
            grid_t = torch.from_numpy(grid).float()
        else:
            input_t = input.float()
            grid_t = grid.float()

        # PyTorch grid_sample expects grid values in range [-1, 1]
        output_t = F.grid_sample(
            input_t, grid_t, mode=mode, padding_mode=padding_mode, align_corners=align_corners
        )

        if was_numpy:
            return output_t.detach().cpu().numpy()

        return output_t
