import os
import numpy as np
from PIL import Image

def divide_image_into_patches(image, n_patches_h=3, n_patches_w=4):
    """
    Divide an image into n_patches_h x n_patches_w equal patches (10 total)
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

def create_test_patches():
    """
    Divide C_1_4_19_bot image into 10 patches and save to data_marion/test_patches/
    """
    # Create output directories
    output_dir = "../data_marion/test_patches"
    os.makedirs(f"{output_dir}/image", exist_ok=True)
    os.makedirs(f"{output_dir}/vein_mask", exist_ok=True)
    os.makedirs(f"{output_dir}/leaf_mask", exist_ok=True)
    
    # Load image, vein mask, leaf mask
    print("Loading C_1_4_19_bot image...")
    image = np.array(Image.open("../data/images/C_1_4_19_bot.jpeg"))
    vein_mask = np.array(Image.open("../data/vein_masks/C_1_4_19_bot.png"))
    leaf_mask = np.array(Image.open("../data_marion/leaf_preds/C_1_4_19_bot.png"))
    
    print(f"Image shape: {image.shape}")
    print(f"Vein mask shape: {vein_mask.shape}")
    print(f"Leaf mask shape: {leaf_mask.shape}")
    
    # Divide into 10 patches (3x4 grid)
    print("\nDividing into 3x4 grid (10 patches)...")
    image_patches = divide_image_into_patches(image, n_patches_h=3, n_patches_w=4)
    vein_patches = divide_image_into_patches(vein_mask, n_patches_h=3, n_patches_w=4)
    leaf_patches = divide_image_into_patches(leaf_mask, n_patches_h=3, n_patches_w=4)
    
    # Save patches
    print(f"\nSaving patches to {output_dir}...")
    for i, (img_patch, vein_patch, leaf_patch) in enumerate(
        zip(image_patches, vein_patches, leaf_patches)
    ):
        patch_name = f"patch_{i:02d}"
        
        # Save image
        Image.fromarray(img_patch).save(f"{output_dir}/image/{patch_name}.jpeg")
        
        # Save vein mask
        Image.fromarray(vein_patch).save(f"{output_dir}/vein_mask/{patch_name}.png")
        
        # Save leaf mask (ROI)
        Image.fromarray(leaf_patch).save(f"{output_dir}/leaf_mask/{patch_name}.png")
        
        print(f"  Saved patch {i+1}/10")
    
    print(f"\n✓ Created {len(image_patches)} test patches!")
    print(f"  Location: {output_dir}/")

if __name__ == "__main__":
    create_test_patches()