import os, sys, glob, pdb, random, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from PIL import Image
from importlib import reload
import yaml

os.chdir(os.path.dirname(os.path.realpath(__file__)))

sys.path.append("../")
import models.BuildCNN as BuildCNN
import models.VeinGrower as VeinGrower
from utils.GetLowestGPU import GetLowestGPU

if "device" not in locals():
    device = torch.device(GetLowestGPU(verbose=2))

# Get config file from command line argument or use default
config_file = sys.argv[1] if len(sys.argv) > 1 else "../config_inference.yaml"

with open(config_file, "r") as f:
    config = yaml.safe_load(f)

#### Initialize grower ####

# Load vein grower parameters from config
window_size = config["vein_grower"]["window_size"]
layers = config["vein_grower"]["layers"]
output_shape = config["vein_grower"]["output_shape"]
loss_type = config["loss"]["type"]
weights_path = config["inference"]["model_weights_path"]
output_activation = torch.nn.Softmax2d()

print(f"Loading CNN model from: {weights_path}")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model weights not found: {weights_path}")

# Load CNN model
print("Loading CNN model...")
reload(BuildCNN)
model = BuildCNN.CNN(
    window_size=window_size,
    layers=layers,
    output_shape=output_shape,
    output_activation=output_activation,
).to(device)
weights = torch.load(weights_path, map_location=device)
model.load_state_dict(weights)
model.eval()

# Initialize vein grower
print("Initializing vein grower...")
reload(VeinGrower)
grower = VeinGrower.VeinGrower(
    window_size=window_size, model=model, device=device, verbose=True
)

#### Inference parameters ####

# Load inference parameters from config
image_path = config["inference"]["image_path"]
roi_path = config["inference"]["roi_path"]
pred_path = config["inference"]["pred_path"]
prob_path = config["inference"]["prob_path"]

os.makedirs(pred_path, exist_ok=True)
os.makedirs(prob_path, exist_ok=True)

roi_extension = config["data"]["roi_extension"]
pred_extension = config["inference"]["pred_extension"]
prob_extension = config["inference"]["prob_extension"]
n_locs = config["inference"]["n_locs"]
batch_size = config["inference"]["batch_size"]
threshold = config["inference"]["threshold"]
post_process = config["inference"]["post_process"]
max_number = config["inference"]["max_number"]
verbose = config["inference"]["verbose"]
save = config["inference"]["save"]
show = config["inference"]["show"]
fig_size = config["inference"]["fig_size"]

# Get image paths
image_names = []
for ext in ["jpeg", "tiff", "tif", "jpg", "png"]:
    image_names.extend([os.path.basename(f) for f in glob.glob(image_path + "*" + ext)])
image_names = list(set(image_names))  # Remove duplicates
image_names.sort()

print(f"Found {len(image_names)} images to process")
if max_number is not None:
    image_names = image_names[:max_number]
    print(f"Processing {len(image_names)} images (limited by max_number)")

#### Loop over all leaf images ####

for image_idx, image_name in enumerate(image_names):
    if verbose:
        print(f"\n[{image_idx + 1}/{len(image_names)}] Processing {image_name}...")

    # Load image
    image = np.array(Image.open(image_path + image_name), dtype=np.float32) / 255

    # Load ROI if available
    if roi_path is not None:
        roi_candidates = [
            roi_path + os.path.splitext(image_name)[0] + "." + roi_extension,
            roi_path + image_name,
        ]
        roi_file = None
        for candidate in roi_candidates:
            if os.path.exists(candidate):
                roi_file = candidate
                break

        if roi_file:
            roi_array = np.array(Image.open(roi_file), dtype=np.float32) / 255

            # Handle both 2D (grayscale) and 3D (RGB) masks
            if roi_array.ndim == 3:  # RGB mask
                roi = roi_array[:, :, 0] > 0.5
            else:  # Already 2D grayscale
                roi = roi_array > 0.5
        else:
            if verbose:
                print(f"  Warning: ROI file not found for {image_name}")
            roi = None
    else:
        roi = None

    # Segment the venation
    if verbose:
        print("  Growing veins...")
    t0 = time.time()
    prob, mask = grower.grow(
        image=image,
        roi=roi,
        start_locs=None,
        n_locs=n_locs,
        batch_size=batch_size,
        threshold=threshold,
        post_process=post_process,
    )
    t1 = time.time()
    if verbose:
        print(f"  Completed in {t1 - t0:1.2f} seconds")

    # Get positive class
    prob = prob[0]

    # Save mask
    if save:
        if verbose:
            print("  Saving mask...")
        save_mask = np.concatenate(
            [mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=-1
        )
        pil_mask = Image.fromarray(np.uint8(255 * save_mask))
        name = pred_path + os.path.splitext(image_name)[0] + "." + pred_extension
        pil_mask.save(name, quality=100, subsampling=0)

    # Save probability map
    if save:
        if verbose:
            print("  Saving probability map...")
        prob_single = prob[0] if len(prob.shape) == 3 else prob
        save_prob = np.concatenate(
            [prob_single[:, :, None], prob_single[:, :, None], prob_single[:, :, None]],
            axis=-1,
        )
        pil_prob = Image.fromarray(np.uint8(255 * save_prob))
        name = prob_path + os.path.splitext(image_name)[0] + "." + prob_extension
        pil_prob.save(name, quality=100, subsampling=0)

    # Plot overlay
    if show:
        if verbose:
            print("  Plotting overlay...")
        image_overlay = image.copy()
        image_overlay[mask] = [1, 0, 0]
        fig = plt.figure(
            figsize=(
                image_overlay.shape[1] / image_overlay.shape[0] * fig_size,
                fig_size,
            )
        )
        plt.imshow(image_overlay)
        plt.title(f"{image_name} - Vein predictions")
        plt.tight_layout()
        plt.show()

print(f"\n✓ Inference complete! Results saved to:")
print(f"  - Masks: {pred_path}")
print(f"  - Probabilities: {prob_path}")
