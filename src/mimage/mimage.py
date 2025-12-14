from __future__ import annotations
import numpy as np
from .affine import Affine
from typing import Any, Optional, Union, List, Tuple, TYPE_CHECKING
from .utils import get_orientation_from_direction_cosines
from mimage.backends.tensor.tensor_torch import TensorTorchBackend
from mimage.backends.tensor.tensor_numpy import TensorNumpyBackend

# optional dependency
if TYPE_CHECKING:
    import torch

ArrayLike = Union["torch.Tensor", np.ndarray, List[float], Tuple[float, ...]]

_BACKEND_MAP = {
    TensorTorchBackend.name: TensorTorchBackend, # torch
    TensorNumpyBackend.name: TensorNumpyBackend, # numpy
}

class Mimage:
    """Medical image with spatial metadata and backend-agnostic data.
    
    Mimage enforces that images always have exactly 3 spatial dimensions,
    consistent with medical imaging standards and the 4x4 affine transformation.
    
    Key constraints:
    - Arrays must have at least 3 dimensions
    - Exactly 3 dimensions are designated as "spatial" (tracked by spatial_dims)
    - Additional dimensions can represent batches, channels, time, etc.
    - Spatial dimensions cannot be removed via indexing
    
    Attributes:
        data: The image data (torch.Tensor or np.ndarray), shape (..., D1, D2, D3, ...)
        affine (Affine): 4x4 affine transformation mapping 3D voxel indices to 3D physical space.
        spatial_dims (tuple[int, int, int]): Indices of the 3 spatial dimensions (always sorted).
    
    Examples:
        >>> # 3D image (all dimensions spatial)
        >>> img = Mimage(np.zeros((64, 128, 256)))  # spatial_dims = (0, 1, 2)
        >>> img.shape
        (64, 128, 256)
        
        >>> # 4D image (batch + spatial)
        >>> img = Mimage(np.zeros((10, 64, 128, 256)))  # spatial_dims = (1, 2, 3)
        >>> img.shape
        (10, 64, 128, 256)
        
        >>> # Slicing spatial dimensions (keeps dimension via slicing)
        >>> sliced = img[:, 10:20, :, :]  # OK - slice keeps dimensions
        >>> sliced.shape
        (10, 10, 128, 256)
        
        >>> # Cannot remove spatial dimensions
        >>> img[0, 0]  # ERROR - would remove spatial dimensions 1 and 2
        Traceback (most recent call last):
        ...
        ValueError: Cannot remove spatial dimensions via indexing...
        
        >>> # Correct way to extract data
        >>> plane = img.data[0, 0]  # Returns numpy/torch array, not Mimage
        >>> plane.shape
        (128, 256)
    """

    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        affine: Affine = None,
        spatial_dims: Tuple[int, int, int] = None,
        backend='auto' # 'auto', 'torch', 'numpy'
    ):
        if not backend in ('auto', 'torch', 'numpy'):
            raise ValueError("backend must be 'auto', 'torch', or 'numpy'")
        
        if backend in ('auto', 'torch'):
            try:
                import torch
            except ImportError:
                torch = None
            if torch is not None and isinstance(data, torch.Tensor) or backend == 'torch':
                backend = 'torch'
            else:
                backend = 'numpy'
                
        self._backend = _BACKEND_MAP[backend]
        self.backend = self._backend.name
                
        self.data = self._backend.asarray(data)
        
        # Enforce minimum 3 dimensions
        if self.data.ndim < 3:
            raise ValueError(
                f"Mimage requires data with at least 3 dimensions (got shape {self.data.shape}). "
                f"Medical images must have 3 spatial dimensions."
            )
        
        self.affine = affine if affine is not None else Affine()
        
        if spatial_dims is None:
            self._spatial_dims = tuple(range(self.data.ndim - 3, self.data.ndim))
        else:
            self._spatial_dims = tuple(sorted(spatial_dims))
        
        # Validate exactly 3 spatial dimensions
        if len(self._spatial_dims) != 3:
            raise ValueError(
                f"Mimage requires exactly 3 spatial dimensions, got {len(self._spatial_dims)}. "
                f"spatial_dims={self._spatial_dims} for array shape {self.data.shape}"
            )
        
        # Validate spatial_dims are valid indices
        if any(dim >= self.data.ndim or dim < 0 for dim in self._spatial_dims):
            raise ValueError(
                f"spatial_dims {self._spatial_dims} contains invalid indices for array shape {self.data.shape}"
            )
        

    def __repr__(self) -> str:
        return (
            f"Mimage(shape={tuple(self.data.shape)}, "
            f"spacing={self.spacing}, origin={self.origin}, direction=\n{self.direction})"
        )

    @property
    def spatial_dims(self) -> Tuple[int, int, int]:
        """Returns the spatial dimensions (always sorted)."""
        return self._spatial_dims

    @spatial_dims.setter
    def spatial_dims(self, value: Tuple[int, int, int]) -> None:
        """Sets spatial dimensions (automatically sorts them, must be exactly 3)."""
        if value is None:
            raise ValueError(
                "spatial_dims cannot be None. Mimage requires exactly 3 spatial dimensions."
            )
        
        value_tuple = tuple(sorted(value))
        
        if len(value_tuple) != 3:
            raise ValueError(
                f"spatial_dims must have exactly 3 elements, got {len(value_tuple)}"
            )
        
        self._spatial_dims = value_tuple

    def __getitem__(self, idx: int | slice | Tuple[Union[int, slice], ...]) -> "Mimage":
        """Indexing that updates affine and spatial_dims when axes are added/removed/sliced.

        Handles:
            - Integer indexing (removes dimension)
            - Slice indexing (keeps dimension, updates affine translation)
            - Stride slicing [::step] (scales spacing by abs(step))
            - Reverse slicing [::-1] (flips direction and adjusts origin)
            - None/np.newaxis (adds dimension)
            - Ellipsis (...)
        
        Behavior matches SimpleITK:
            - slice[start:stop]: Origin shifts by start * spacing
            - slice[::step]: Spacing scales by abs(step)
            - slice[::-1]: Direction flips, origin moves to last voxel
            - Combined slice[start:stop:step]: Both origin shift and spacing scale
        """
        new_data = self.data[idx]
        
        # If result is a scalar, just return it
        if not hasattr(new_data, "shape") or new_data.ndim == 0:
            return new_data

        # Normalize index to tuple
        if not isinstance(idx, tuple):
            idx = (idx,)

        # Track dimension mapping: old_dim -> new_dim
        old_dim = 0
        new_dim = 0
        dim_mapping = {}  # old -> new
        removed_dims = set()
        slice_info = {}  # old_dim -> (start_offset, step)

        for idx_elem in idx:
            if idx_elem is None:  # np.newaxis - adds a dimension
                new_dim += 1
                continue
            elif idx_elem is Ellipsis:
                # Ellipsis consumes remaining dimensions
                remaining = self.data.ndim - old_dim - (len(idx) - list(idx).index(Ellipsis) - 1)
                for _ in range(remaining):
                    dim_mapping[old_dim] = new_dim
                    slice_info[old_dim] = (0, 1)
                    old_dim += 1
                    new_dim += 1
                continue
            elif isinstance(idx_elem, int):
                # Integer indexing removes dimension
                removed_dims.add(old_dim)
                # For integer indexing, compute which slice it corresponds to
                slice_info[old_dim] = (idx_elem, 1)
                old_dim += 1
                continue
            elif isinstance(idx_elem, slice):
                # Slice keeps dimension
                dim_mapping[old_dim] = new_dim
                
                # Extract slice parameters
                start = idx_elem.start
                stop = idx_elem.stop
                step = idx_elem.step if idx_elem.step is not None else 1
                
                # Resolve negative indices and None
                dim_size = self.data.shape[old_dim]
                if start is None:
                    start = 0 if step > 0 else dim_size - 1
                elif start < 0:
                    start = dim_size + start
                
                if stop is None:
                    stop = dim_size if step > 0 else -1
                elif stop < 0:
                    stop = dim_size + stop
                
                # For reverse slicing with step < 0, the effective start is where we begin
                if step < 0:
                    # When reversing, origin should point to the first element we access
                    # which is at index 'start'
                    effective_start = start
                else:
                    effective_start = start
                
                slice_info[old_dim] = (effective_start, step)
                old_dim += 1
                new_dim += 1
                continue
            else:
                # Advanced indexing (array, list) - keeps dimension
                dim_mapping[old_dim] = new_dim
                slice_info[old_dim] = (0, 1)
                old_dim += 1
                new_dim += 1

        # Handle remaining dimensions not explicitly indexed
        while old_dim < self.data.ndim:
            dim_mapping[old_dim] = new_dim
            slice_info[old_dim] = (0, 1)
            old_dim += 1
            new_dim += 1

        # Build new affine matrix
        new_affine_mat = self.affine.matrix.copy()
        
        # Process each spatial dimension for origin shift, spacing scale, and direction flip
        for i, old_spatial_dim in enumerate(self.spatial_dims):
            if old_spatial_dim in slice_info:
                start_offset, step = slice_info[old_spatial_dim]
                
                # Update origin: shift by start_offset * spacing (original spacing before step scaling)
                if start_offset != 0:
                    new_affine_mat[:3, 3] += new_affine_mat[:3, i] * start_offset
                
                # Update spacing: scale by abs(step)
                if abs(step) != 1:
                    new_affine_mat[:3, i] *= abs(step)
                
                # Update direction: flip if step < 0
                if step < 0:
                    new_affine_mat[:3, i] *= -1

        # Update spatial_dims: map old spatial dims to new positions, skip removed
        new_spatial_dims = []
        for old_spatial_dim in self.spatial_dims:
            if old_spatial_dim not in removed_dims and old_spatial_dim in dim_mapping:
                new_spatial_dims.append(dim_mapping[old_spatial_dim])

        # ENFORCE: Must maintain exactly 3 spatial dimensions
        if len(new_spatial_dims) != 3:
            removed_spatial = [dim for dim in self.spatial_dims if dim in removed_dims]
            raise ValueError(
                f"Cannot remove spatial dimensions via indexing. "
                f"Attempted to remove dimension(s) {removed_spatial} from spatial_dims {self.spatial_dims}. "
                f"This would result in {len(new_spatial_dims)} spatial dimensions (need exactly 3). "
                f"Use slicing to keep dimensions: e.g., img[{removed_spatial[0] if removed_spatial else 0}:1] instead of img[{removed_spatial[0] if removed_spatial else 0}]"
            )

        return Mimage(
            new_data,
            affine=Affine(matrix=new_affine_mat),
            spatial_dims=tuple(new_spatial_dims),
        )

    def clone(self) -> "Mimage":
        """Create a deep copy of the Mimage.
        
        Returns:
            Mimage: New Mimage instance with copied data and affine.
        """
        data = self._backend.clone(self.data)
        return Mimage(data, affine=self.affine, spatial_dims=self.spatial_dims)

    def clip(self, min_val=None, max_val=None) -> "Mimage":
        """Clip (limit) the values in the array.
        
        Returns a new Mimage with values clipped to the specified range.
        Follows the immutable pattern - does not modify the original image.
        
        Args:
            min_val: Minimum value. If None, no lower bound is applied.
            max_val: Maximum value. If None, no upper bound is applied.
            
        Returns:
            Mimage: New Mimage instance with clipped data.
            
        Example:
            >>> img = Mimage(np.array([[[1, 2, 3], [4, 5, 6]]]))
            >>> clipped = img.clip(min_val=2, max_val=5)
            >>> # Values now in range [2, 5]: [[2, 2, 3], [4, 5, 5]]
        """
        clipped_data = self._backend.clip(self.data, min_val, max_val)
        return Mimage(clipped_data, affine=self.affine, spatial_dims=self.spatial_dims)

    def index_to_coord(self, coords: ArrayLike) -> np.ndarray:
        """Convert array indices to physical coordinates.
        
        The indices must correspond to the 3 spatial dimensions tracked by spatial_dims.
        
        Args:
            coords: Index coordinates (length 3) or (N, 3) array. These are indices
                    into the spatial dimensions only, in their current sorted order.
        
        Returns:
            np.ndarray: Physical coordinates (length 3) or (N, 3).
        
        Example:
            >>> img = Mimage(np.zeros((10, 20, 30)))  # spatial_dims = (0, 1, 2)
            >>> img.index_to_coord([5, 10, 15])  # indices for dims 0, 1, 2
            array([5., 10., 15.])  # physical coordinates (with default affine)
            
            >>> img4d = Mimage(np.zeros((2, 10, 20, 30)))  # spatial_dims = (1, 2, 3)
            >>> img4d[0].index_to_coord([5, 10, 15])  # still 3 values for spatial dims
            array([5., 10., 15.])
        """
        coords_np = np.asarray(coords, dtype=np.float32)
        return self.affine.index_to_coord(coords_np)

    def coord_to_index(self, coords: ArrayLike) -> np.ndarray:
        """Convert physical coordinates to array indices.
        
        The indices returned correspond to the 3 spatial dimensions tracked by spatial_dims.
        
        Args:
            coords: Physical coordinates (length 3) or (N, 3) array.
        
        Returns:
            np.ndarray: Index coordinates (length 3) or (N, 3) for the spatial dimensions.
        
        Example:
            >>> img = Mimage(np.zeros((10, 20, 30)), affine=Affine(spacing=[2, 2, 2]))
            >>> img.coord_to_index([10, 20, 30])  # physical coordinates
            array([5., 10., 15.])  # indices into spatial dimensions
        """
        coords_np = np.asarray(coords, dtype=np.float32)
        return self.affine.coord_to_index(coords_np)

    @property
    def orientation(self) -> str:
        return get_orientation_from_direction_cosines(self.direction)

    @property
    def spacing(self) -> np.ndarray:
        return self.affine.spacing

    @property
    def origin(self) -> np.ndarray:
        return self.affine.origin

    @property
    def direction(self) -> np.ndarray:
        return self.affine.direction

    @classmethod
    def from_sitk(cls, sitk_img, **kwargs) -> "Mimage":
        try:
            import SimpleITK as sitk
        except ImportError:
            raise ImportError("SimpleITK is required for from_sitk.")
        arr = sitk.GetArrayFromImage(sitk_img)
        origin = sitk_img.GetOrigin()
        spacing = sitk_img.GetSpacing()
        direction = np.asarray(sitk_img.GetDirection()).reshape(3, 3)
        affine = Affine(direction=direction, spacing=spacing, origin=origin).permute_axes((2, 1, 0))
        return cls(arr, affine=affine, **kwargs)

    def to_sitk(self):
        try:
            import SimpleITK as sitk
        except ImportError:
            raise ImportError("SimpleITK is required for to_sitk.")
        arr = np.ascontiguousarray(self.data)
        sitk_img = sitk.GetImageFromArray(arr)
        affine_sitk = self.affine.permute_axes((2, 1, 0))
        sitk_img.SetSpacing(tuple(affine_sitk.spacing.tolist()))
        sitk_img.SetOrigin(tuple(affine_sitk.origin.tolist()))
        sitk_img.SetDirection(tuple(affine_sitk.direction.flatten()))
        return sitk_img

    def permute(self, *dims) -> "Mimage":
        """Permute the dimensions of the image array.

        Args:
            *dims: The desired ordering of dimensions.

        Returns:
            Mimage: Permuted image with updated affine and sorted spatial_dims.
        """
        new_data = self._backend.permute(self.data, dims)
        
        # Update spatial_dims (guaranteed to exist and have length 3)
        spatial_rank = 3
        old_spatial = list(self.spatial_dims)
        
        # Find where each old spatial dim ends up after permutation
        new_spatial_unsorted = tuple(dims.index(d) for d in old_spatial)
        
        # Check if spatial dims are contiguous at the end after permutation
        if sorted(new_spatial_unsorted) == list(range(len(dims) - spatial_rank, len(dims))):
            # Spatial dims are contiguous at end, so we can update the affine
            # Create permutation for affine based on the new sorted order
            sorted_new_spatial = tuple(sorted(new_spatial_unsorted))
            # Map from old spatial axes (0,1,2) to new positions in sorted order
            affine_perm = tuple(new_spatial_unsorted.index(s) for s in sorted_new_spatial)
            aff = self.affine.permute_axes(affine_perm)
            spatial_dims = sorted_new_spatial
        else:
            # Spatial dims are scattered, don't update affine
            aff = self.affine
            spatial_dims = tuple(sorted(new_spatial_unsorted))
        
        return Mimage(new_data, affine=aff, spatial_dims=spatial_dims, backend=self.backend)

    def flip(self, dims) -> "Mimage":
        """Flip the image along specified dimensions.
        
        Args:
            dims: Dimensions to flip.
        
        Returns:
            Mimage: Flipped image with updated affine.
        """
        new_data = self._backend.flip(self.data, dims)
        
        # Update affine if flipping spatial dims
        spatial_axes = [ax for ax in dims if ax in self.spatial_dims]
        if spatial_axes:
            affine_axes = [self.spatial_dims.index(ax) for ax in spatial_axes]
            # Pass full spatial shape, not just flipped axes
            spatial_shape = tuple(self.data.shape[ax] for ax in self.spatial_dims)
            aff = self.affine.flip_axes(affine_axes, shape=spatial_shape)
        else:
            aff = self.affine
        
        return Mimage(new_data, affine=aff, spatial_dims=self.spatial_dims)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    @property
    def shape_spatial(self) -> Tuple[int, int, int]:
        """Returns the shape of the spatial dimensions only, in sorted order."""
        return tuple(self.data.shape[dim] for dim in self.spatial_dims)
