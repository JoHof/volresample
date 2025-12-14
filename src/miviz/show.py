import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from scipy import ndimage

def overlay(
    img: np.ndarray,
    mask: np.ndarray,
    label_to_idx: dict = None,
    overlay_cmap: ListedColormap = None,
    alpha: float = 0.7,
    cmap: str = 'jet'
) -> np.ndarray:
    """
    Overlay a mask on a single-channel image using a colormap.
    Args:
        img: 2D numpy array, normalized to [0,1]
        mask: 2D numpy array, same shape as img
        label_to_idx: dict mapping label values to colormap indices (default: label==index)
        overlay_cmap: ListedColormap for mask overlay (default: 'jet')
        alpha: blending factor for mask
        cmap: colormap name if overlay_cmap is not provided
    Returns:
        RGB image as np.ndarray, shape (H, W, 3)
    """
    # Convert grayscale to RGB
    img_rgb = np.stack([img]*3, axis=-1)
    # Default label_to_idx: label==index for all unique nonzero labels
    if label_to_idx is None:
        labels = sorted(set(np.unique(mask)) - {0})
        label_to_idx = {label: label for label in labels}
    # Default colormap: 'jet' with enough colors for max label
    if overlay_cmap is None:
        n_labels = max(label_to_idx.values(), default=0) + 1
        cmap_base = plt.get_cmap(cmap, n_labels)
        cmap_list = cmap_base(np.arange(n_labels))
        cmap_list[0, -1] = 0.0  # make alpha=0 for background
        overlay_cmap = ListedColormap(cmap_list)
    mask_idx = np.zeros_like(mask, dtype=int)
    for lbl, idx in label_to_idx.items():
        mask_idx[mask == lbl] = idx
    mask_rgb = overlay_cmap(mask_idx)[..., :3]  # ignore alpha channel
    mask_alpha = (mask_idx > 0).astype(float) * alpha
    # Blend where mask is present
    out = img_rgb * (1 - mask_alpha[..., None]) + mask_rgb * mask_alpha[..., None]
    return out

def imprint_text(
    img: np.ndarray,
    text: str,
    position: str = "lower_left",
    color_fg=(0, 0, 0),
    color_bg=(255, 255, 255),
    font_scale: float = None,
    thickness: int = None,
    margin: int = 5,
    alpha: float = 1.0,
    font=None
) -> np.ndarray:
    """
    Imprint text onto an image at a specified position.

    Args:
        img: Image as numpy array (H, W, 3), float [0,1] or uint8.
        text: Text to imprint.
        position: One of 'lower_left', 'upper_left', 'upper_right', 'lower_right', or (x, y) tuple.
        color_fg: Foreground (text) color, tuple of 3 floats (0-1) or ints (0-255).
        color_bg: Background rectangle color, tuple of 3 floats (0-1) or ints (0-255).
        font_scale: Font scale (None = auto).
        thickness: Font thickness (None = auto).
        margin: Margin from edge in pixels.
        alpha: Alpha blending for imprint (1.0 = opaque).
        font: cv2 font (default: cv2.FONT_HERSHEY_SIMPLEX)
    Returns:
        Image with text imprinted.
    """
    import cv2

    img = img.copy()
    h, w = img.shape[:2]
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        float_input = True
    else:
        float_input = False
    if font is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
    if font_scale is None:
        font_scale = max(0.25, h / 256)
    if thickness is None:
        thickness = max(2, h // 128)

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    rect_w, rect_h = text_size

    # Determine position
    if isinstance(position, str):
        if position == "lower_left":
            text_x = margin
            text_y = h - margin
        elif position == "upper_left":
            text_x = margin
            text_y = margin + rect_h
        elif position == "upper_right":
            text_x = w - rect_w - margin
            text_y = margin + rect_h
        elif position == "lower_right":
            text_x = w - rect_w - margin
            text_y = h - margin
        else:
            raise ValueError(f"Unknown position: {position}")
    else:
        text_x, text_y = position

    # Draw background rectangle
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (text_x - 2, text_y - rect_h - 2),
        (text_x + rect_w + 2, text_y + 2),
        color_bg,
        thickness=-1
    )
    # Blend background if alpha < 1
    if alpha < 1.0:
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    else:
        img = overlay

    # Draw text
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color_fg, thickness, cv2.LINE_AA)
    if float_input:
        img = img.astype(np.float32) / 255.0

    return img


import matplotlib.pyplot as plt

def imprint_figure(
    img: np.ndarray,
    text: str,
    position: str = "lower_left",
    color_fg=(0, 0, 0),
    color_bg=(1, 1, 1),
    font_size: float = None,
    margin: int = 5,
    ax=None,
    **kwargs
):
    """
    Show an image with a text label overlayed using matplotlib (not rendered into pixels).

    Args:
        img: Image as numpy array (H, W, 3) or (H, W).
        text: Text to overlay.
        position: One of 'lower_left', 'upper_left', 'upper_right', 'lower_right', or (x, y) tuple in pixels.
        color_fg: Foreground (text) color, tuple of 3 floats (0-1) or ints (0-255).
        color_bg: Background rectangle color, tuple of 3 floats (0-1) or ints (0-255).
        font_size: Font size in points (None = auto).
        margin: Margin from edge in pixels.
        ax: Optional matplotlib axis to plot on.
        **kwargs: Passed to plt.imshow.
    Returns:
        The matplotlib axis with the image and text.
    """
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.imshow(img, **kwargs)
    ax.axis("off")

    h, w = img.shape[:2]
    if font_size is None:
        font_size = max(10, h // 16)

    # Convert color to matplotlib format
    def to_float(c):
        return tuple(float(x) / 255.0 if isinstance(x, int) and x > 1 else float(x) for x in c)
    color_fg = to_float(color_fg)
    color_bg = to_float(color_bg)

    # Determine position
    if isinstance(position, str):
        if position == "lower_left":
            x = margin
            y = h - margin
            va = "bottom"
            ha = "left"
        elif position == "upper_left":
            x = margin
            y = margin
            va = "top"
            ha = "left"
        elif position == "upper_right":
            x = w - margin
            y = margin
            va = "top"
            ha = "right"
        elif position == "lower_right":
            x = w - margin
            y = h - margin
            va = "bottom"
            ha = "right"
        else:
            raise ValueError(f"Unknown position: {position}")
    else:
        x, y = position
        va = "bottom"
        ha = "left"

    # Draw background rectangle for text
    from matplotlib.patches import Rectangle
    import matplotlib.transforms as mtransforms

    # Estimate text width/height in pixels
    renderer = fig.canvas.get_renderer()
    t = ax.text(
        x, y, text, fontsize=font_size, color=color_fg, va=va, ha=ha,
        bbox=dict(facecolor=color_bg, edgecolor='none', boxstyle='round,pad=0.2', alpha=1.0)
    )
    return ax

# def mimshow(vol: np.ndarray,
#             msk: list[np.ndarray] | np.ndarray = [],
#             n_slices: int = 8,
#             slices_per_row: int = 8,
#             labels: list[int] | None = None,
#             clip: list[float] = [-1024, 300],
#             alpha: float = 0.7,
#             outline_alpha: float = 0.0,
#             cmap: str = 'jet',
#             padding: tuple = (2, 2),
#             background: float = 0,
#             width: float = 10,
#             stack_native: bool = False,
#             only_nonzero: bool = False,
#             plot: bool = True,
#             imprint_index: bool = False):
#     """
#     Visualize a subset of slices from a 3D medical image volume with optional segmentation mask overlays.
#     Uses montage for display.

#     Args:
#         vol: 3D numpy array (Z, H, W), the volume to display
#         msk: segmentation masks, can be a single array or a list of arrays
#         n_slices: number of slices to display
#         slices_per_row: number of slices to display per row
#         labels: list of label values for segmentation masks
#         clip: intensity clipping range for display
#         alpha: transparency for mask fill overlay (0 disables fill)
#         outline_alpha: transparency for mask outline overlay (0 disables outline)
#         cmap: colormap for mask overlay
#         padding: tuple, (pad_rows, pad_cols), padding between images in the montage
#         background: value to use for padding/background in the montage
#         width: width of the plot in inches (height is set automatically)
#         stack_native: if True, for each slice, stack the native image (no overlay) above the overlay image
#         only_nonzero: if True, only show slices where at least one mask has nonzero label
#         plot: if True, plot with matplotlib, else return the image matrix
#         imprint_index: if True, imprint the slice index onto each slice
#     """

#     # Handle masks: accept single np.ndarray or list
#     if isinstance(msk, np.ndarray):
#         msk = [msk]
#     elif msk is None:
#         msk = []

#     # Clip and normalize volume to [0,1]
#     vol_disp = np.clip(vol, clip[0], clip[1])
#     vol_disp = (vol_disp - clip[0]) / (clip[1] - clip[0])

#     # Efficiently determine which slices to show
#     if only_nonzero and msk and any(m is not None for m in msk):
#         # Stack all masks for efficient computation
#         mask_stack = np.stack([m for m in msk if m is not None], axis=0)  # (n_masks, Z, H, W)
#         mask_sum = np.any(mask_stack != 0, axis=(0, 2, 3))  # (Z,)
#         valid_indices = np.where(mask_sum)[0]
#         if len(valid_indices) == 0:
#             z_indices = []
#         elif len(valid_indices) <= n_slices:
#             z_indices = valid_indices
#         else:
#             lin_idx = np.linspace(0, len(valid_indices) - 1, n_slices, dtype=int)
#             z_indices = valid_indices[lin_idx]
#     else:
#         z_indices = np.linspace(0, vol.shape[0] - 1, n_slices, dtype=int)

#     # Collect labels if not provided
#     if labels is None and len(msk) > 0:
#         # Use only the selected slices for label detection
#         unique_labels = set()
#         for m in msk:
#             if m is not None:
#                 unique_labels.update(np.unique(m[z_indices]))
#         labels = sorted(unique_labels)
#     if labels is None:
#         labels = []
#     labels = [l for l in labels if l != 0]  # ignore background 0

#     # Build colormap with transparency for background
#     if labels:
#         cmap_base = plt.get_cmap(cmap, len(labels) + 1)  # +1 for background slot
#         cmap_list = cmap_base(np.arange(len(labels) + 1))
#         cmap_list[0, -1] = 0.0  # make alpha=0 for background
#         overlay_cmap = ListedColormap(cmap_list)
#         norm = Normalize(vmin=0, vmax=len(labels))
#         label_to_idx = {label: i+1 for i, label in enumerate(labels)}  # shift by +1 so 0 stays background
#     else:
#         overlay_cmap, norm, label_to_idx = None, None, {}

#     # Prepare RGB slices with overlays (vectorized where possible)
#     rgb_slices = []
#     for i, z in enumerate(z_indices):
#         img = vol_disp[z]
#         img_rgb = np.repeat(img[..., None], 3, axis=-1)

#         overlays = []
#         # Always add the native image first
#         if stack_native:
#             overlays.append(img_rgb)

#         # If msk is a list, add overlays for each mask
#         if isinstance(msk, list) and len(msk) > 1:
#             for j, m in enumerate(msk):
#                 img_overlay = img_rgb.copy()
#                 if m is not None and overlay_cmap is not None:
#                     mask_sum = m[z]
#                     # Fill mask
#                     if alpha > 0 and mask_sum.max() > 0:
#                         mask_idx = np.zeros_like(mask_sum, dtype=int)
#                         for lbl, idx in label_to_idx.items():
#                             mask_idx[mask_sum == lbl] = idx
#                         mask_rgb = overlay_cmap(mask_idx)[..., :3]
#                         mask_alpha = (mask_idx > 0).astype(float) * alpha
#                         img_overlay = img_overlay * (1 - mask_alpha[..., None]) + mask_rgb * mask_alpha[..., None]
#                     # Outline mask
#                     if outline_alpha > 0 and mask_sum.max() > 0:
#                         outline_mask = np.zeros_like(mask_sum)
#                         for lbl in label_to_idx:
#                             if lbl == 0:
#                                 continue
#                             binary = (mask_sum == lbl)
#                             eroded = ndimage.binary_erosion(binary)
#                             outline = binary & ~eroded
#                             outline_mask[outline] = lbl
#                         outline_idx = np.zeros_like(outline_mask, dtype=int)
#                         for lbl, idx in label_to_idx.items():
#                             outline_idx[outline_mask == lbl] = idx
#                         outline_rgb = overlay_cmap(outline_idx)[..., :3]
#                         outline_alpha_map = (outline_idx > 0).astype(float) * outline_alpha
#                         img_overlay = img_overlay * (1 - outline_alpha_map[..., None]) + outline_rgb * outline_alpha_map[..., None]
#                 # Only imprint index on the last overlay
#                 if imprint_index and j == len(msk) - 1:
#                     img_overlay = imprint_text(img_overlay, str(z), position="lower_left")
#                 overlays.append(img_overlay)
#             stacked = np.vstack(overlays)
#             rgb_slices.append(stacked)
#         else:
#             # Single overlay (or no mask)
#             img_overlay = img_rgb.copy()
#             if msk and overlay_cmap is not None:
#                 mask_sum = np.zeros_like(img, dtype=int)
#                 for m in msk:
#                     if m is not None:
#                         mask_sum = np.maximum(mask_sum, m[z])
#                 # Fill mask
#                 if alpha > 0 and mask_sum.max() > 0:
#                     mask_idx = np.zeros_like(mask_sum, dtype=int)
#                     for lbl, idx in label_to_idx.items():
#                         mask_idx[mask_sum == lbl] = idx
#                     mask_rgb = overlay_cmap(mask_idx)[..., :3]
#                     mask_alpha = (mask_idx > 0).astype(float) * alpha
#                     img_overlay = img_overlay * (1 - mask_alpha[..., None]) + mask_rgb * mask_alpha[..., None]
#                 # Outline mask
#                 if outline_alpha > 0 and mask_sum.max() > 0:
#                     outline_mask = np.zeros_like(mask_sum)
#                     for lbl in label_to_idx:
#                         if lbl == 0:
#                             continue
#                         binary = (mask_sum == lbl)
#                         eroded = ndimage.binary_erosion(binary)
#                         outline = binary & ~eroded
#                         outline_mask[outline] = lbl
#                     outline_idx = np.zeros_like(outline_mask, dtype=int)
#                     for lbl, idx in label_to_idx.items():
#                         outline_idx[outline_mask == lbl] = idx
#                     outline_rgb = overlay_cmap(outline_idx)[..., :3]
#                     outline_alpha_map = (outline_idx > 0).astype(float) * outline_alpha
#                     img_overlay = img_overlay * (1 - outline_alpha_map[..., None]) + outline_rgb * outline_alpha_map[..., None]
#             if imprint_index:
#                 img_overlay = imprint_text(img_overlay, str(z), position="lower_left")
#             if stack_native:
#                 stacked = np.vstack([img_rgb, img_overlay])
#                 rgb_slices.append(stacked)
#             else:
#                 rgb_slices.append(img_overlay)
#     if len(rgb_slices) == 0:
#         print("No slices to display.")
#         return
#     rgb_slices = np.stack(rgb_slices, axis=0)

#     # Create montage
#     grid_shape = (int(np.ceil(len(rgb_slices) / slices_per_row)), slices_per_row)
#     mntg = montage(rgb_slices, grid_shape=grid_shape, padding=padding, fill=background)

#     # Set figure size based on width
#     fig_width = width
#     fig_height = width * (mntg.shape[0] / mntg.shape[1]) if mntg.shape[1] > 0 else width
#     if plot:
#         plt.figure(figsize=(fig_width, fig_height))
#         plt.imshow(mntg)
#         plt.axis("off")
#         plt.show()
#     else:
#         return mntg

def mimshow(vol: np.ndarray,
            msk: list[np.ndarray] | np.ndarray = [],
            n_slices: int = 8,
            slices_per_row: int = 8,
            labels: list[int] | None = None,
            clip: list[float] = [-1024, 300],
            alpha: float = 0.7,
            outline_alpha: float = 0.0,
            outline_thickness: int = 2,
            cmap: str = 'jet',
            padding: tuple = (2, 2),
            background: float = 0,
            width: float = 10,
            stack_native: bool = False,
            only_nonzero: bool = False,
            plot: bool = True,
            show_slice_index: bool = False,
            row_labels: list[str] = None):
    """
    Visualize a subset of slices from a 3D medical image volume with optional segmentation mask overlays.
    Uses montage for display.

    Args:
        vol: 3D numpy array (Z, H, W), the volume to display
        msk: segmentation masks, can be a single array or a list of arrays
        n_slices: number of slices to display
        slices_per_row: number of slices to display per row
        labels: list of label values for segmentation masks
        clip: intensity clipping range for display
        alpha: transparency for mask fill overlay (0 disables fill)
        outline_alpha: transparency for mask outline overlay (0 disables outline)
        cmap: colormap for mask overlay
        padding: tuple, (pad_rows, pad_cols), padding between images in the montage
        background: value to use for padding/background in the montage
        width: width of the plot in inches (height is set automatically)
        stack_native: if True, for each slice, stack the native image (no overlay) above the overlay image
        only_nonzero: if True, only show slices where at least one mask has nonzero label
        plot: if True, plot with matplotlib, else return the image matrix
        show_slice_index: if True, imprint the slice index onto each slice (with matplotlib text, not pixels)
        row_labels: list of str, optional. If provided, add text on each row (left, outside of the montage)
    """

    # Handle masks: accept single np.ndarray or list
    if isinstance(msk, np.ndarray):
        msk = [msk]
    elif msk is None:
        msk = []

    # Clip and normalize volume to [0,1]
    vol_disp = np.clip(vol, clip[0], clip[1])
    vol_disp = (vol_disp - clip[0]) / (clip[1] - clip[0])

    # Efficiently determine which slices to show
    if only_nonzero and msk and any(m is not None for m in msk):
        mask_stack = np.stack([m for m in msk if m is not None], axis=0)  # (n_masks, Z, H, W)
        mask_sum = np.any(mask_stack != 0, axis=(0, 2, 3))  # (Z,)
        valid_indices = np.where(mask_sum)[0]
        if len(valid_indices) == 0:
            z_indices = []
        elif len(valid_indices) <= n_slices:
            z_indices = valid_indices
        else:
            lin_idx = np.linspace(0, len(valid_indices) - 1, n_slices, dtype=int)
            z_indices = valid_indices[lin_idx]
    else:
        z_indices = np.linspace(0, vol.shape[0] - 1, n_slices, dtype=int)

    # Collect labels if not provided
    if labels is None and len(msk) > 0:
        unique_labels = set()
        for m in msk:
            if m is not None:
                unique_labels.update(np.unique(m[z_indices], sorted=False))
        labels = sorted(unique_labels)
    if labels is None:
        labels = []
    labels = [l for l in labels if l != 0]  # ignore background 0

    # Build colormap with transparency for background
    if labels:
        cmap_base = plt.get_cmap(cmap, len(labels) + 1)  # +1 for background slot
        cmap_list = cmap_base(np.arange(len(labels) + 1))
        cmap_list[0, -1] = 0.0  # make alpha=0 for background
        overlay_cmap = ListedColormap(cmap_list)
        norm = Normalize(vmin=0, vmax=len(labels))
        label_to_idx = {label: i+1 for i, label in enumerate(labels)}  # shift by +1 so 0 stays background
    else:
        overlay_cmap, norm, label_to_idx = None, None, {}

    # Prepare RGB slices with overlays (vectorized where possible)
    rgb_slices = []
    for i, z in enumerate(z_indices):
        img = vol_disp[z]
        img_rgb = np.repeat(img[..., None], 3, axis=-1)

        overlays = []
        if stack_native:
            overlays.append(img_rgb)

        for j, m in enumerate(msk):
            img_overlay = img_rgb.copy()
            if m is not None and overlay_cmap is not None:
                mask_sum = m[z]
                # Fill mask
                if alpha > 0 and mask_sum.max() > 0:
                    mask_idx = np.zeros_like(mask_sum, dtype=int)
                    for lbl, idx in label_to_idx.items():
                        mask_idx[mask_sum == lbl] = idx
                    mask_rgb = overlay_cmap(mask_idx)[..., :3]
                    mask_alpha = (mask_idx > 0).astype(float) * alpha
                    img_overlay = img_overlay * (1 - mask_alpha[..., None]) + mask_rgb * mask_alpha[..., None]
                # Outline mask
                if outline_alpha > 0 and mask_sum.max() > 0:
                    outline_mask = np.zeros_like(mask_sum)
                    for lbl in label_to_idx:
                        if lbl == 0:
                            continue
                        binary = (mask_sum == lbl)
                        eroded = ndimage.binary_erosion(binary, iterations=outline_thickness)
                        outline = binary & ~eroded
                        outline_mask[outline] = lbl
                    outline_idx = np.zeros_like(outline_mask, dtype=int)
                    for lbl, idx in label_to_idx.items():
                        outline_idx[outline_mask == lbl] = idx
                    outline_rgb = overlay_cmap(outline_idx)[..., :3]
                    outline_alpha_map = (outline_idx > 0).astype(float) * outline_alpha
                    img_overlay = img_overlay * (1 - outline_alpha_map[..., None]) + outline_rgb * outline_alpha_map[..., None]
            overlays.append(img_overlay)

        if not overlays:
            overlays.append(img_rgb)
        stacked = np.vstack(overlays)
        rgb_slices.append(stacked)
    if len(rgb_slices) == 0:
        print("No slices to display.")
        return
    rgb_slices = np.stack(rgb_slices, axis=0)

    # Create montage
    n_rows_per_slice = (1 if stack_native else 0) + len(msk)
    grid_shape = (int(np.ceil(len(rgb_slices) / slices_per_row)), slices_per_row)
    mntg = montage(rgb_slices, grid_shape=grid_shape, padding=padding, fill=background)

    # Set figure size based on width
    fig_width = width
    fig_height = width * (mntg.shape[0] / mntg.shape[1]) if mntg.shape[1] > 0 else width

    if plot:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.imshow(mntg)
        ax.axis("off")

        # Compute grid and image sizes
        img_h, img_w = rgb_slices[0].shape[:2]
        pad_h, pad_w = padding

        # Avoid division by zero
        if n_rows_per_slice == 0:
            n_rows_per_slice = 1

        # compute target fontsize
        # pixels = points × (DPI / 72)
        dpi = fig.dpi                        # dots per inch
        fw, fh = fig.get_size_inches() * dpi
        h_scale = fh / mntg.shape[0]
        single_slice_height = rgb_slices[0].shape[0] / n_rows_per_slice
        target_fontsize_pt = (single_slice_height * h_scale * .07) / (dpi / 72)

        # Overlay index or other text on each sub-image (using matplotlib text, not pixels)
        if show_slice_index:
            for idx in range(len(rgb_slices)):
                row = idx // slices_per_row
                col = idx % slices_per_row
                y0 = row * (img_h + pad_h)
                x0 = col * (img_w + pad_w)
                z = z_indices[idx]
                bbox_margin_px = int((.1 * (target_fontsize_pt) * dpi / 72) / h_scale) + 1
                ax.text(
                    x0 + bbox_margin_px, y0 + img_h - bbox_margin_px,
                    str(z),
                    color='black', fontsize=target_fontsize_pt,
                    va='bottom', ha='left',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.1')
                )
        plt.show()
    else:
        return mntg

def montage(
    arr: np.ndarray,
    grid_shape: tuple = None,
    padding: tuple = (0, 0),
    fill: float = 0
) -> np.ndarray:
    """
    Create a 2D montage from a stack of 2D images (arr: shape [N, H, W] or [N, H, W, C]).
    Args:
        arr: 3D or 4D numpy array (N, H, W) or (N, H, W, C)
        grid_shape: (n_rows, n_cols), optional. If None, tries to make a square grid.
        padding: (pad_rows, pad_cols), padding between images.
        fill: value to use for padding.
    Returns:
        2D or 3D numpy array with montage.
    """
    arr = np.asarray(arr)
    assert arr.ndim in (3, 4), "Input must be 3D (N, H, W) or 4D (N, H, W, C)"
    n_imgs, h, w = arr.shape[:3]
    is_color = arr.ndim == 4
    if grid_shape is None:
        n_cols = int(np.ceil(np.sqrt(n_imgs)))
        n_rows = int(np.ceil(n_imgs / n_cols))
    else:
        n_rows, n_cols = grid_shape
    pad_rows, pad_cols = padding
    out_h = n_rows * h + (n_rows - 1) * pad_rows
    out_w = n_cols * w + (n_cols - 1) * pad_cols
    if is_color:
        c = arr.shape[3]
        out = np.full((out_h, out_w, c), fill, dtype=arr.dtype)
    else:
        out = np.full((out_h, out_w), fill, dtype=arr.dtype)
    for idx in range(n_imgs):
        row = idx // n_cols
        col = idx % n_cols
        if row >= n_rows:
            break
        y = row * (h + pad_rows)
        x = col * (w + pad_cols)
        if is_color:
            out[y:y+h, x:x+w, :] = arr[idx]
        else:
            out[y:y+h, x:x+w] = arr[idx]
    return out
