import numpy as np

def get_orientation_from_direction_cosines(direction: np.ndarray) -> str:
    """Get DICOM 3-letter orientation string from a 3x3 direction cosine matrix.

    The cosine is defined such that its columns are orthonormal and represent the 
    direction cosines of the image axes in physical space. The physical space is defined to be in the DICOM SPL. That is: 
        - The first physical coordinate is the x-axis (right->left)
        - The second physical coordinate is the y-axis (anterior->posterior)
        - The third physical coordinate is the z-axis (inferior->superior)

    Args:
        direction (np.ndarray): 3x3 direction cosine matrix (columns = image axes).

    Returns:
        str: 3-letter orientation string, e.g., 'RAS', 'LPI', etc.
    """
    if direction.shape != (3, 3):
        raise ValueError("Direction matrix must be 3x3")

    axes_labels = ['L', 'R', 'P', 'A', 'S', 'I']
    axes_vectors = np.array([[1, 0, 0],  # L
                             [-1, 0, 0], # R
                             [0, 1, 0],  # P
                             [0, -1, 0], # A
                             [0, 0, 1],  # S
                             [0, 0, -1]])# I

    orientation = ""
    for col in range(3):
        dir_vector = direction[:, col]
        # Compute dot product with each standard axis
        dots = axes_vectors @ dir_vector
        # Pick the axis best aligned with the direction vector
        max_idx = np.argmax(dots)
        orientation += axes_labels[max_idx]

    return orientation

