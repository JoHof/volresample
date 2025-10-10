from mimage.affine import Affine
from .mimage import Mimage
import numpy as np
import SimpleITK as sitk

def sitk_to_mimage(sitk_img: sitk.Image) -> Mimage:
    """Convert a SimpleITK image to a Mimage object.

    Args:
        sitk_img (sitk.Image): The SimpleITK image to convert.

    Returns:
        Mimage: The resulting Mimage object.
    """
    o = sitk_img.GetOrigin()
    s = sitk_img.GetSpacing()
    d = np.asarray(sitk_img.GetDirection()).reshape(3, 3)
    arr = sitk_img.GetArrayFromImage(sitk_img)
    affine = Affine(direction=d, spacing=s, origin=o).transpose_axes((2, 1, 0))
    return Mimage(arr, affine=affine)

def mimage_to_sitk(mimage: Mimage) -> sitk.Image:
    """Convert a Mimage object to a SimpleITK image.

    Args:
        mimage (Mimage): The Mimage object to convert.

    Returns:
        sitk.Image: The resulting SimpleITK image.
    """
    arr = mimage.detach().cpu().numpy()
    arr = np.ascontiguousarray(arr)
    sitk_img = sitk.GetImageFromArray(arr)
    # Set spacing, origin, direction from mimage properties

    affine_sitk = mimage.affine.transpose_axes((2, 1, 0))  # Ensure correct axis order
    sitk_img.SetSpacing(tuple(affine_sitk.spacing.tolist()))
    sitk_img.SetOrigin(tuple(affine_sitk.origin.tolist()))
    sitk_img.SetDirection(tuple(affine_sitk.direction.flatten()))
    return sitk_img