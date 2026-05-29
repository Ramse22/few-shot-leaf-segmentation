import os
import numpy as np
from PIL import Image

os.chdir(os.path.dirname(os.path.realpath(__file__)))


def divide_image_into_patches(image, n_patches_h=3, n_patches_w=4):
    """
    Divide an image into n_patches_h x n_patches_w equal patches (12 total)
    """
    h, w = image.shape[:2]
    patch_h = h // n_patches_h
    patch_w = w // n_patches_w

    patches = []
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            y1 = i * patch_h
            y2 = (i + 1) * patch_h
            x1 = j * patch_w
            x2 = (j + 1) * patch_w

            patch = image[y1:y2, x1:x2]
            patches.append(patch)

    return patches


def add_padding_to_patches(patches, window_size=128, padding_factor=1.5):
    """
    Add padding to patches to allow safe rotation sampling.
    padding_factor = 1.5 means 50% more padding than window_size/2
    """
    pad_size = int(window_size / 2 * padding_factor)
    padded_patches = []

    for patch in patches:
        if len(patch.shape) == 3:  # RGB image
            padded = np.pad(
                patch,
                pad_width=[[pad_size, pad_size], [pad_size, pad_size], [0, 0]],
                mode="constant",
                constant_values=0,
            )
        else:  # Grayscale mask
            padded = np.pad(
                patch,
                pad_width=[[pad_size, pad_size], [pad_size, pad_size]],
                mode="constant",
                constant_values=0,
            )
        padded_patches.append(padded)

    return padded_patches


def create_test_patches():
    """
    Divide C_1_4_19_bot image into 12 patches (3x4 grid) with padding and save to
    data_marion/test_patches/.

    Saved folders are always: image, vein_mask.
    Additional folders are saved only if source files exist:
    - leaf_mask (from data/leaf_preds)
    - leaf_pred (from data/leaf_preds)
    - leaf_pred_sam3 (from data_marion/leaf_preds_jlag_sam3)
    """
    # Create output directories
    output_dir = "../data_marion/test_patches"
    os.makedirs(f"{output_dir}/image", exist_ok=True)
    os.makedirs(f"{output_dir}/vein_mask", exist_ok=True)
    os.makedirs(f"{output_dir}/leaf_mask", exist_ok=True)
    os.makedirs(f"{output_dir}/leaf_pred", exist_ok=True)
    os.makedirs(f"{output_dir}/leaf_pred_sam3", exist_ok=True)

    # Load mandatory sources
    print("Loading C_1_4_19_bot image...")
    image = np.array(Image.open("../data/images/C_1_4_19_bot.jpeg"))
    vein_mask = np.array(Image.open("../data/vein_masks/C_1_4_19_bot.png"))

    # Optional sources
    leaf_mask_path = "../data/leaf_masks/C_1_4_19_bot.png"
    leaf_pred_path = "../data/leaf_preds/C_1_4_19_bot.png"
    leaf_pred_sam3_path = "../data_marion/leaf_preds_jlag_sam3/C_1_4_19_bot.png"

    leaf_mask = np.array(Image.open(leaf_mask_path)) if os.path.exists(leaf_mask_path) else None
    leaf_pred = np.array(Image.open(leaf_pred_path)) if os.path.exists(leaf_pred_path) else None
    leaf_pred_sam3 = (
        np.array(Image.open(leaf_pred_sam3_path)) if os.path.exists(leaf_pred_sam3_path) else None
    )

    print(f"Image shape:          {image.shape}")
    print(f"Vein mask shape:      {vein_mask.shape}")
    if leaf_mask is not None:
        print(f"Leaf mask shape:      {leaf_mask.shape}")
    else:
        print("Leaf mask missing:    data/leaf_masks")
    if leaf_pred is not None:
        print(f"Leaf pred shape:      {leaf_pred.shape}")
    else:
        print("Leaf pred missing:    data/leaf_preds")
    if leaf_pred_sam3 is not None:
        print(f"Leaf pred SAM3 shape: {leaf_pred_sam3.shape}")
    else:
        print("Leaf pred SAM3 missing: data_marion/leaf_preds_jlag_sam3")

    # Sanity check — all must have same spatial dimensions
    assert image.shape[:2] == vein_mask.shape[:2], "Image/vein mask size mismatch"
    if leaf_mask is not None:
        assert image.shape[:2] == leaf_mask.shape[:2], "Image/leaf mask size mismatch"
    if leaf_pred is not None:
        assert image.shape[:2] == leaf_pred.shape[:2], "Image/leaf pred size mismatch"
    if leaf_pred_sam3 is not None:
        assert image.shape[:2] == leaf_pred_sam3.shape[:2], "Image/SAM3 leaf pred size mismatch"

    window_size = 128

    # Divide into 12 patches (3x4 grid)
    print("\nDividing into 3x4 grid (12 patches)...")
    image_patches = divide_image_into_patches(image, n_patches_h=3, n_patches_w=4)
    vein_patches = divide_image_into_patches(vein_mask, n_patches_h=3, n_patches_w=4)
    leaf_patches = (
        divide_image_into_patches(leaf_mask, n_patches_h=3, n_patches_w=4)
        if leaf_mask is not None
        else None
    )
    leaf_pred_patches = (
        divide_image_into_patches(leaf_pred, n_patches_h=3, n_patches_w=4)
        if leaf_pred is not None
        else None
    )
    leaf_sam3_patches = (
        divide_image_into_patches(leaf_pred_sam3, n_patches_h=3, n_patches_w=4)
        if leaf_pred_sam3 is not None
        else None
    )

    # Add padding to patches
    print("Adding padding to patches...")
    image_patches_padded = add_padding_to_patches(
        image_patches, window_size=window_size, padding_factor=1.5
    )
    vein_patches_padded = add_padding_to_patches(
        vein_patches, window_size=window_size, padding_factor=1.5
    )
    leaf_patches_padded = (
        add_padding_to_patches(leaf_patches, window_size=window_size, padding_factor=1.5)
        if leaf_patches is not None
        else None
    )
    leaf_pred_patches_padded = (
        add_padding_to_patches(leaf_pred_patches, window_size=window_size, padding_factor=1.5)
        if leaf_pred_patches is not None
        else None
    )
    leaf_sam3_patches_padded = (
        add_padding_to_patches(leaf_sam3_patches, window_size=window_size, padding_factor=1.5)
        if leaf_sam3_patches is not None
        else None
    )

    # Save all patches
    print(f"\nSaving padded patches to {output_dir}...")
    for i, (img_patch, vein_patch) in enumerate(zip(image_patches_padded, vein_patches_padded)):
        patch_name = f"patch_{i:02d}"

        Image.fromarray(img_patch).save(f"{output_dir}/image/{patch_name}.jpeg")
        Image.fromarray(vein_patch).save(f"{output_dir}/vein_mask/{patch_name}.png")
        if leaf_patches_padded is not None:
            Image.fromarray(leaf_patches_padded[i]).save(f"{output_dir}/leaf_mask/{patch_name}.png")
        if leaf_pred_patches_padded is not None:
            Image.fromarray(leaf_pred_patches_padded[i]).save(f"{output_dir}/leaf_pred/{patch_name}.png")
        if leaf_sam3_patches_padded is not None:
            Image.fromarray(leaf_sam3_patches_padded[i], mode="L").save(
                f"{output_dir}/leaf_pred_sam3/{patch_name}.png"
            )

        print(f"  Saved patch {i + 1}/12 - Shape after padding: {img_patch.shape}")

if __name__ == "__main__":
    create_test_patches()