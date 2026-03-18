import os, glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Load saved masks
pred_path = '../data_marion/leaf_preds_2/'
image_path = '../data_marion/images/'

# Get list of saved predictions
pred_files = sorted(glob.glob(pred_path + '*.png'))

print(f"Found {len(pred_files)} saved predictions\n")

# Visualize a few examples
num_examples = 3  # Change this to show more/fewer
fig_size = 12

for pred_idx, pred_file in enumerate(pred_files[:num_examples]):
    # Get the image name from the mask filename (without extension)
    base_name = os.path.splitext(os.path.basename(pred_file))[0]
    
    # Try different image extensions
    image_file = None
    for ext in ['.jpeg', '.jpg', '.png', '.tiff', '.tif']:
        candidate = image_path + base_name + ext
        if os.path.exists(candidate):
            image_file = candidate
            break
    
    # Load image and mask
    if image_file:
        image = np.array(Image.open(image_file), dtype=float) / 255
        mask = np.array(Image.open(pred_file), dtype=float)[:, :, 0] / 255
        
        # Create overlay plot
        fig = plt.figure(figsize=(fig_size, fig_size * image.shape[0] / image.shape[1]))
        
        # Plot image with contour overlay
        plt.imshow(image)
        contour = measure.find_contours(mask, 0.5)
        if contour:
            for cont in contour:
                plt.plot(cont[:, 1], cont[:, 0], 'r-', linewidth=2)
        
        plt.title(f"Example {pred_idx + 1}: {base_name}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Image file not found for: {base_name}")

#Leaf vein visualization

vein_pred_path = '../data_marion/vein_fl_preds2/'
image_path = '../data_marion/images/'

for tiff_file in glob.glob(image_path + '*.tiff'):
    img = Image.open(tiff_file)
    jpeg_path = tiff_file.replace('.tiff', '.jpeg')
    img.convert('RGB').save(jpeg_path, 'JPEG')
    print(f"Converted {os.path.basename(tiff_file)}")

# Get vein predictions
vein_files = sorted(glob.glob(vein_pred_path + '*.png'))
print(f"\nFound {len(vein_files)} vein predictions")

# Visualize with overlay (like in TracerInference)
num_examples = 5
fig_size = 15

for vein_idx, vein_file in enumerate(vein_files[-num_examples:]):
    base_name = os.path.splitext(os.path.basename(vein_file))[0]
    
    # Try different image extensions
    image_file = None
    for ext in ['.jpeg', '.jpg', '.png', '.tiff', '.tif']:
        candidate = image_path + base_name + ext
        if os.path.exists(candidate):
            image_file = candidate
            break
    
    if image_file:
        image = np.array(Image.open(image_file), dtype=np.float32) / 255
        vein_mask = np.array(Image.open(vein_file), dtype=np.float32)[:, :, 0] / 255
        
        # Create overlay with red veins
        image[vein_mask > 0.5] = [1, 0, 0]
        
        fig = plt.figure(figsize=(image.shape[1]/image.shape[0]*fig_size, fig_size))
        plt.imshow(image)
        plt.title(f"Vein Segmentation: {base_name}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Image file not found for: {base_name}")