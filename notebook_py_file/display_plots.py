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
