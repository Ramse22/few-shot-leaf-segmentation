import os, sys, glob, pdb, random, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from PIL import Image
from importlib import reload
import yaml
from skimage import measure

os.chdir(os.path.dirname(os.path.realpath(__file__)))

sys.path.append("../")
import models.BuildCNN as BuildCNN
import models.VeinGrower as VeinGrower
from utils.GetLowestGPU import GetLowestGPU


def build_output_activation(name):
    if name is None:
        return None

    name = str(name).lower()
    if name == "softmax":
        return torch.nn.Softmax2d()
    if name == "sigmoid":
        return torch.nn.Sigmoid()

    raise ValueError(f"Unsupported output activation: {name}")


if "device" not in locals():
    device = torch.device(GetLowestGPU(verbose=2))

# Get config file from command line argument or use default
config_file = sys.argv[1] if len(sys.argv) > 1 else "../config_inference.yaml"

with open(config_file, "r") as f:
    config = yaml.safe_load(f)

#### initialize grower ####

# options
loss = config["loss"]["type"]
window_size = config["vein_grower"]["window_size"]
layers = config["vein_grower"]["layers"]
output_shape = config["vein_grower"]["output_shape"]
output_activation = build_output_activation(
    config["vein_grower"].get("output_activation")
)
inference_config = config.get("inference", {})
weights_path = inference_config.get(
    "model_weights_path",
    f"../weights_marion/vein_grower_{loss}_{window_size}_best_val_model.save",
)

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
pred_path = inference_config["pred_path"]
prob_path = inference_config["prob_path"]

os.makedirs(pred_path, exist_ok=True)
os.makedirs(prob_path, exist_ok=True)

image_extension = config["data"].get("image_extension", "*")
roi_extension = config["data"].get("roi_extension", "png")
pred_extension = inference_config.get("pred_extension", "png")
prob_extension = inference_config.get("prob_extension", "png")
n_locs = inference_config.get("n_locs", 10000)  # number of seed pixels
batch_size = inference_config.get("batch_size", 2048)
threshold = inference_config.get("threshold", None)
post_process = inference_config.get("post_process", True)
max_number = inference_config.get(
    "max_number", inference_config.get("num_predictions", None)
)  # number of images to segment, set to None for all images
verbose = inference_config.get("verbose", True)
save = inference_config.get("save", True)
show = inference_config.get("show", True)
fig_size = inference_config.get("fig_size", 15)

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
