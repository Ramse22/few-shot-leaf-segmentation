import os, sys, glob, pdb, random, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from PIL import Image
from importlib import reload
from skimage import measure
import yaml

os.chdir(os.path.dirname(os.path.realpath(__file__)))

sys.path.append("../")
import models.BuildCNN as BuildCNN
import models.VeinGrower as VeinGrower
from utils.GetLowestGPU import GetLowestGPU

if "device" not in locals():
    device = torch.device(GetLowestGPU(verbose=2))

#### Load config ####

config_path = sys.argv[1] if len(sys.argv) > 1 else "../configs/config_inf.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

#### initialize grower ####

# options
window_size = config["data"]["window_size"]
loss = config["model"]["loss"]
weights_path = config["data"]["weights_path"]
layers = config["model"]["layers"]
output_shape = config["model"]["output_shape"]
output_activation = getattr(torch.nn, config["model"]["output_activation"])()

# load CNN model
print("loading cnn model...")
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

# initialize vein grower
print("initializing vein grower...")
reload(VeinGrower)
grower = VeinGrower.VeinGrower(
    window_size=window_size, model=model, device=device, verbose=True
)


#### Grower inference ####

# options
image_path = config["data"]["image_path"]
roi_path = config["data"]["roi_path"]

# Determine prediction/probability output locations.
# If the config's pred/prob paths point to the generic results root (e.g. "../results/"),
# save preds/probs inside the same run folder as the weights file so outputs live with weights/logs.
weights_dir = (
    os.path.dirname(weights_path) if os.path.isfile(weights_path) else weights_path
)
pred_root_cfg = config["data"].get("pred_path", "../results/")
prob_root_cfg = config["data"].get("prob_path", "../results/")
if os.path.normpath(pred_root_cfg).endswith("results"):
    pred_path = os.path.join(weights_dir, f"vein_{loss}_preds/")
else:
    pred_path = os.path.join(pred_root_cfg, f"vein_{loss}_preds/")

if os.path.normpath(prob_root_cfg).endswith("results"):
    prob_path = os.path.join(weights_dir, f"vein_{loss}_probs/")
else:
    prob_path = os.path.join(prob_root_cfg, f"vein_{loss}_probs/")

os.makedirs(pred_path, exist_ok=True)
os.makedirs(prob_path, exist_ok=True)

image_extension = "*"
roi_extension = config["data"]["roi_extension"]
pred_extension = config["data"]["pred_extension"]
prob_extension = config["data"]["prob_extension"]
n_locs = config["inference"]["n_locs"]
batch_size = config["inference"]["batch_size"]
threshold = config["inference"]["threshold"]
post_process = config["inference"]["post_process"]
max_number = config["inference"]["max_number"]
verbose = config["inference"]["verbose"]
save = config["inference"]["save"]
show = config["inference"]["show"]
fig_size = config["inference"]["fig_size"]

# get image paths
image_names = []
for ext in ["jpeg", "tiff", "tif", "jpg", "png"]:
    image_names.extend([os.path.basename(f) for f in glob.glob(image_path + "*" + ext)])
image_names = list(set(image_names))  # Remove duplicates
image_names.sort()

# loop over all leaf images
for image_idx, image_name in enumerate(image_names):
    # don't exceed maximum
    if max_number is not None:
        if image_idx >= max_number:
            break

    # load image
    if verbose:
        print(f"Loading {image_name}...")
    image = np.array(Image.open(image_path + image_name), dtype=np.float32) / 255
    if roi_path is not None:
        roi_candidates = [
            roi_path + image_name.replace(image_extension, roi_extension),
            roi_path + os.path.splitext(image_name)[0] + "." + roi_extension,
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
            roi = None
    else:
        roi = None

    # segment the venation
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
        print("Iteration completed in {0:1.2f} seconds".format(t1 - t0))

    # get positive class
    prob = prob[0]

    # save mask
    if save:
        if verbose:
            print("Saving mask...")
        save_mask = np.concatenate(
            [mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=-1
        )
        pil_mask = Image.fromarray(np.uint8(255 * save_mask))
        name = pred_path + os.path.splitext(image_name)[0] + "." + pred_extension
        pil_mask.save(name, quality=100, subsampling=0)

    # save prob
    if save:
        if verbose:
            print("Saving prob...")
        prob = prob[0] if len(prob.shape) == 3 else prob
        save_prob = np.concatenate(
            [prob[:, :, None], prob[:, :, None], prob[:, :, None]], axis=-1
        )
        pil_prob = Image.fromarray(np.uint8(255 * save_prob))
        name = prob_path + os.path.splitext(image_name)[0] + "." + prob_extension
        pil_prob.save(name, quality=100, subsampling=0)

    # plot overlay
    if show:
        if verbose:
            print("Plotting overlay...")
        image[mask] = [1, 0, 0]
        fig = plt.figure(figsize=(image.shape[1] / image.shape[0] * fig_size, fig_size))
        plt.imshow(image)
        plt.show()

    if verbose:
        print()
