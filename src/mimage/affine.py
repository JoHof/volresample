import numpy as np
from typing import Optional, Tuple, Union

ArrayLike = Union[np.ndarray, list, tuple]

class Affine:
    """Represents a 4x4 affine transformation matrix for 3D spatial mappings.

    This class provides an intuitive interface for working with affine matrices
    in 3D medical imaging, supporting conversions between index and physical
    coordinates, and easy access to direction, spacing, and origin components.

    The affine matrix follows the convention:
        [ R * S | T ]
        [  0    | 1 ]
    where R is the direction (rotation) matrix (3x3), S is the diagonal spacing matrix,
    and T is the origin (translation) vector (length 3).
     
    R is defined such that its rows are orthonormal and represent the 
    direction cosines of the image axes in physical space.

    Example:
        >>> affine = Affine(direction=np.eye(3),
        ...                 spacing=[0.33, 1, 0.33],
        ...                 origin=[-90.3, 10, 1.44])
        >>> index = np.array([4, 0, 9])
        >>> coord = affine.index_to_coord(index)
        >>> print(coord)
        [-88.98  10.     4.41]

    Attributes:
        matrix (np.ndarray): The 4x4 affine matrix.
    """

    def __init__(
        self,
        matrix: Optional[np.ndarray] = None,
        *,
        direction: Optional[ArrayLike] = None,
        spacing: Optional[ArrayLike] = None,
        origin: Optional[ArrayLike] = None,
    ) -> None:
        """Initializes an affine transform from a full matrix or components.

        Args:
            matrix: Optional 4x4 affine matrix.
            direction: Orthonormal direction matrix (3x3), defaults to identity. Columns are the direction cosines of the image axes.
            spacing: Axis spacing vector of length 3, defaults to ones.
            origin: Origin vector of length 3, defaults to zeros.
        """
        if matrix is not None:
            self.matrix = np.asarray(matrix, dtype=float)
            if self.matrix.shape != (4, 4):
                raise ValueError("Affine matrix must be 4x4 for 3D.")
            self._direction, self._spacing, self._origin = self.decompose(self.matrix)
        else:
            self._direction = np.asarray(
                direction if direction is not None else np.eye(3), dtype=float
            )
            if self._direction.shape != (3, 3):
                raise ValueError("Direction matrix must be 3x3 for 3D.")
            self._spacing = np.asarray(
                spacing if spacing is not None else np.ones(3),
                dtype=float,
            )
            if self._spacing.shape != (3,):
                raise ValueError("Spacing must be a vector of length 3 for 3D.")
            if np.any(self._spacing <= 0):
                raise ValueError("Spacing values must be positive.")
            self._origin = np.asarray(
                origin if origin is not None else np.zeros(3),
                dtype=float,
            )
            if self._origin.shape != (3,):
                raise ValueError("Origin must be a vector of length 3 for 3D.")
            self.matrix = self.compose(self._direction, self._spacing, self._origin)

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------
    @property
    def direction(self) -> np.ndarray:
        """Returns the orthonormal direction (rotation) matrix (3x3)."""
        return self._direction.copy()

    @direction.setter
    def direction(self, new_direction: ArrayLike) -> None:
        new_direction = np.asarray(new_direction, dtype=float)
        self._validate_direction(new_direction)
        if new_direction.shape != (3, 3):
            raise ValueError("Direction matrix must be 3x3 for 3D.")
        self._direction = new_direction
        self._update_matrix()

    @property
    def spacing(self) -> np.ndarray:
        """Returns the voxel spacing vector (length 3)."""
        return self._spacing.copy()

    @spacing.setter
    def spacing(self, new_spacing: ArrayLike) -> None:
        new_spacing = np.asarray(new_spacing, dtype=float)
        if new_spacing.shape != (3,):
            raise ValueError("Spacing must be a vector of length 3 for 3D.")
        if np.any(new_spacing <= 0):
            raise ValueError("Spacing values must be positive.")
        self._spacing = new_spacing
        self._update_matrix()

    @property
    def origin(self) -> np.ndarray:
        """Returns the coordinate origin vector (length 3)."""
        return self._origin.copy()

    @origin.setter
    def origin(self, new_origin: ArrayLike) -> None:
        new_origin = np.asarray(new_origin, dtype=float)
        if new_origin.shape != (3,):
            raise ValueError("Origin must be a vector of length 3 for 3D.")
        self._origin = new_origin
        self._update_matrix()

    # --------------------------------------------------------------------------
    # Public methods
    # --------------------------------------------------------------------------
    def index_to_coord(self, index: ArrayLike) -> np.ndarray:
        """Converts index coordinates to physical coordinates.

        Args:
            index: Vector of voxel indices (length 3), or (N,3) array-like.

        Returns:
            np.ndarray: Physical-space coordinates (length 3) or (N,3).
        """
        index = np.asarray(index, dtype=float)
        if index.ndim == 1:
            if index.shape[0] != 3:
                raise ValueError("Index must be a vector of length 3 for 3D.")
            return index @ self.linear_part.T + self.origin
        elif index.ndim == 2:
            if index.shape[1] != 3:
                raise ValueError("Index array must have shape (N, 3) for 3D.")
            return index @ self.linear_part.T + self.origin
        else:
            raise ValueError("Index must be a vector of length 3 or an (N,3) array.")

    def coord_to_index(self, coord: ArrayLike) -> np.ndarray:
        """Converts physical coordinates to index coordinates.

        Args:
            coord: Vector of physical coordinates (length 3), or (N,3) array-like.

        Returns:
            np.ndarray: Index-space coordinates (length 3) or (N,3).
        """
        coord = np.asarray(coord, dtype=float)
        inv_linear = np.linalg.inv(self.linear_part)
        if coord.ndim == 1:
            if coord.shape[0] != 3:
                raise ValueError("Coordinate must be a vector of length 3 for 3D.")
            return (coord - self.origin) @ inv_linear.T
        elif coord.ndim == 2:
            if coord.shape[1] != 3:
                raise ValueError("Coordinate array must have shape (N, 3) for 3D.")
            return (coord - self.origin) @ inv_linear.T
        else:
            raise ValueError("Coordinate must be a vector of length 3 or an (N,3) array.")
    def clone(self) -> "Affine":
        """Returns a deep copy of the affine."""
        return Affine(self.matrix.copy())

    def inverse(self) -> "Affine":
        """Returns the inverse affine transform."""
        return Affine(np.linalg.inv(self.matrix))

    def compose_with(self, other: "Affine") -> "Affine":
        """Composes this affine with another affine: self ∘ other."""
        return Affine(self.matrix @ other.matrix)
    
    def permute_axes(self, axes: Tuple[int, ...]) -> "Affine":
        """Returns a new Affine adapted for a transposed/permuted 3D image array.

        This adjusts the affine so that it remains spatially consistent
        with an array that has been permuted according to `axes`.

        For example:
            If a 3D array is permuted from (x, y, z) → (z, y, x),
            call `affine.permute_axes((2, 1, 0))`.

        Args:
            axes: A permutation tuple of length 3, describing
                how array axes are reordered.

        Returns:
            Affine: New affine corresponding to the permuted array.
        """
        if not (isinstance(axes, tuple) and len(axes) == 3 and sorted(axes) == [0, 1, 2]):
            raise ValueError(f"Invalid axes permutation for 3D affine: {axes}")

        # Permute direction and spacing
        new_spacing = self._spacing[list(axes)]
        new_direction = self._direction[:, list(axes)]
        return Affine(direction=new_direction, spacing=new_spacing, origin=self._origin)

    def flip_axes(self, axes: list[int], shape: tuple[int, ...] = None) -> "Affine":
        """Return a new Affine with the specified axes flipped, updating the origin.

        Args:
            axes (list[int]): List of axis indices (0, 1, or 2) to flip.
            shape (tuple[int, ...], optional): Shape of the image array. Required to update origin.

        Returns:
            Affine: New affine with axes flipped.
        """
        new_matrix = self.matrix.copy()
        if shape is None:
            raise ValueError("Shape must be provided to update origin when flipping axes.")
        for ax in axes:
            new_matrix[:3, ax] *= -1
            new_matrix[:3, 3] -= new_matrix[:3, ax] * (shape[ax] - 1)
        return Affine(matrix=new_matrix)

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------
    @property
    def linear_part(self) -> np.ndarray:
        """The top-left 3x3 linear component."""
        return self.matrix[:-1, :-1]

    def _update_matrix(self) -> None:
        """Rebuilds the full affine matrix from components."""
        self.matrix = self.compose(self._direction, self._spacing, self._origin)

    @staticmethod
    def compose(
        direction: np.ndarray, spacing: np.ndarray, origin: np.ndarray
    ) -> np.ndarray:
        """Builds a 4x4 affine matrix."""
        if direction.shape != (3, 3):
            raise ValueError("Direction matrix must be 3x3 for 3D.")
        if spacing.shape != (3,):
            raise ValueError("Spacing must be a vector of length 3 for 3D.")
        if origin.shape != (3,):
            raise ValueError("Origin must be a vector of length 3 for 3D.")
        affine = np.eye(4)
        affine[:-1, :-1] = direction @ np.diag(spacing)
        affine[:-1, -1] = origin
        return affine

    @staticmethod
    def decompose(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts direction, spacing, and origin from a 4x4 affine matrix."""
        if matrix.shape != (4, 4):
            raise ValueError("Affine matrix must be 4x4 for 3D.")
        linear = matrix[:-1, :-1]
        spacing = np.linalg.norm(linear, axis=0)
        direction = linear @ np.diag(1 / spacing)
        origin = matrix[:-1, -1]
        return direction, spacing, origin

    @staticmethod
    def _validate_direction(direction: np.ndarray) -> None:
        """Validates that the direction matrix is orthonormal."""
        if direction.shape != (3, 3):
            raise ValueError("Direction matrix must be 3x3 for 3D.")
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-6):
            raise ValueError("Direction matrix must be orthonormal.")

    # --------------------------------------------------------------------------
    # Dunder methods
    # --------------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Affine(origin={self._origin}, spacing={self._spacing})"

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        """Matrix multiplication with an array."""
        return self.matrix @ np.asarray(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Affine):
            return False
        return np.allclose(self.matrix, other.matrix)
