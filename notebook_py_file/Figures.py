import os, glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure

os.chdir(os.path.dirname(os.path.abspath(__file__)))

### Paths ###
results_path = "../results/"
mask_path    = "../data/vein_masks/"
image_path   = "../data/images/"
save_path    = "../figures_marion/"
os.makedirs(save_path, exist_ok=True)

# config filename: display label for each approach
APPROACHES = {
    "config.yaml":          "Our approach",
    "config_jlag.yaml":     "JLAG",
    "config_original.yaml": "Original",
}

### some functions ###

# intersection over union
def iou(a, b):
    i = (a.astype(bool) * b.astype(bool)).sum()
    u = (a.astype(bool) + b.astype(bool)).clip(0, 1).sum()
    return i / u

def reduce_dim(mask):
    return mask[:, :, 0] if len(mask.shape) == 3 else mask


# Figure 1 - Example vein prediction for each config/approach

example_name   = "C_1_1_2_bot"
leaf_pred_path = "../data_marion/leaf_preds_jlag_sam3/"
rmin, rmax  = 600, 3200
cmin, cmax  = 600, 2000
figsize     = 5
n_tick      = 400
plot_width  = figsize
plot_height = (rmax - rmin) / (cmax - cmin) * figsize

# find the first run directory for each approach that has predictions
approach_runs = {label: [] for label in APPROACHES.values()}
for run_dir in sorted(glob.glob(results_path + "*/")):
    for config_name, label in APPROACHES.items():
        if os.path.exists(run_dir + config_name):
            approach_runs[label].append(run_dir)
            break

fig_runs = [
    (label, runs[0])
    for label, runs in approach_runs.items()
    if runs and os.path.exists(runs[0] + "vein_fl_preds/" + example_name + ".png")
]

if fig_runs:
    n = len(fig_runs)
    fig, axes = plt.subplots(1, n, figsize=[n * plot_width, plot_height], constrained_layout=True)
    if n == 1:
        axes = [axes]

    image      = np.array(Image.open(image_path + example_name + ".jpeg"), dtype=float) / 255
    image_crop = image[rmin:rmax, cmin:cmax]

    leaf_file = leaf_pred_path + example_name + ".png"
    contour   = None
    if os.path.exists(leaf_file):
        leaf_crop = (np.array(Image.open(leaf_file), dtype=float)[:, :, 0] > 128)[rmin:rmax, cmin:cmax]
        contour   = measure.find_contours(leaf_crop, 0.5)[0]

    for ax, (label, run_dir) in zip(axes, fig_runs):
        vein_crop  = (np.array(Image.open(run_dir + "vein_fl_preds/" + example_name + ".png"), dtype=float)[:, :, 0] > 128)[rmin:rmax, cmin:cmax]
        plot_image = image_crop.copy()
        plot_image[vein_crop] = [1, 0, 0]
        plt.sca(ax)
        plt.imshow(plot_image, vmin=0, vmax=1, extent=[cmin, cmax, rmax, rmin])
        if contour is not None:
            plt.plot(contour[:, 1] + cmin, contour[:, 0] + rmin, "b-", linewidth=2)
        plt.xticks([i * n_tick for i in range(8)], [i * n_tick for i in range(8)], fontsize=12)
        plt.yticks([i * n_tick for i in range(8)], [i * n_tick for i in range(8)], fontsize=12)
        plt.xlim([cmin, cmax - 1])
        plt.ylim([rmax, rmin])
        ax.set_title(label, fontsize=15)

    plt.savefig(save_path + "figure_1.png", bbox_inches="tight", dpi=200)
    plt.show()


# Figure - image seule / + masque reel / + masque predit + contour (run 42, "our approach")
# Uses C_1_14_18_bot since it already has a prediction in this run's vein_fl_preds/

run42_normal_dir  = results_path + "20260601-112025-063664/"
run42_example     = "C_1_14_18_bot"
run42_pred_file   = run42_normal_dir + "vein_fl_preds/" + run42_example + ".png"
run42_real_file   = mask_path + run42_example + ".png"

if os.path.exists(run42_pred_file):
    rmin2, rmax2 = 150, 3200
    cmin2, cmax2 = 900, 1900
    plot_height2 = (rmax2 - rmin2) / (cmax2 - cmin2) * figsize

    image2      = np.array(Image.open(image_path + run42_example + ".jpeg"), dtype=float) / 255
    image_crop2 = image2[rmin2:rmax2, cmin2:cmax2]

    leaf_file2 = leaf_pred_path + run42_example + ".png"
    contour2   = None
    if os.path.exists(leaf_file2):
        leaf_arr2  = np.array(Image.open(leaf_file2), dtype=float)
        leaf_crop2 = (leaf_arr2[:, :, 0] if leaf_arr2.ndim == 3 else leaf_arr2) > 128
        leaf_crop2 = leaf_crop2[rmin2:rmax2, cmin2:cmax2]
        contour2   = measure.find_contours(leaf_crop2, 0.5)[0]

    vein_arr2  = np.array(Image.open(run42_pred_file), dtype=float)
    vein_crop2 = (vein_arr2[:, :, 0] if vein_arr2.ndim == 3 else vein_arr2) > 128
    vein_crop2 = vein_crop2[rmin2:rmax2, cmin2:cmax2]

    # masque reel, meme convention de binarisation que pour le calcul d'IoU plus bas
    real_crop2 = None
    if os.path.exists(run42_real_file):
        real_arr2  = np.array(Image.open(run42_real_file), dtype=float)
        real_bin2  = reduce_dim(real_arr2) / 255 > 0.5
        real_crop2 = real_bin2[rmin2:rmax2, cmin2:cmax2]

    n_panels = 2 + (real_crop2 is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=[n_panels * plot_width, plot_height2], constrained_layout=True)

    panel_idx = 0

    # Panneau : image seule
    plt.sca(axes[panel_idx])
    plt.imshow(image_crop2, vmin=0, vmax=1, extent=[cmin2, cmax2, rmax2, rmin2])
    axes[panel_idx].set_title("Image de base", fontsize=15)
    panel_idx += 1

    # Panneau : image + masque reel (si dispo)
    if real_crop2 is not None:
        plot_real2 = image_crop2.copy()
        plot_real2[real_crop2] = [1, 0, 0]
        plt.sca(axes[panel_idx])
        plt.imshow(plot_real2, vmin=0, vmax=1, extent=[cmin2, cmax2, rmax2, rmin2])
        axes[panel_idx].set_title("+ masque reel (veines)", fontsize=15)
        panel_idx += 1

    # Panneau : image + masque predit + contour feuille
    plot_image2 = image_crop2.copy()
    plot_image2[vein_crop2] = [1, 0, 0]
    plt.sca(axes[panel_idx])
    plt.imshow(plot_image2, vmin=0, vmax=1, extent=[cmin2, cmax2, rmax2, rmin2])
    if contour2 is not None:
        plt.plot(contour2[:, 1] + cmin2, contour2[:, 0] + rmin2, "b-", linewidth=2)
    axes[panel_idx].set_title("Our approach - run 42 (" + run42_example + ")", fontsize=15)

    for ax in axes:
        ax.set_xticks([cmin2 + i * n_tick for i in range(int((cmax2 - cmin2) / n_tick) + 1)])
        ax.set_yticks([rmin2 + i * n_tick for i in range(int((rmax2 - rmin2) / n_tick) + 1)])
        ax.tick_params(labelsize=12)
        ax.set_xlim([cmin2, cmax2 - 1])
        ax.set_ylim([rmax2, rmin2])

    plt.savefig(save_path + "figure_run42_normal.png", bbox_inches="tight", dpi=200)
    plt.show()


# IoU per run (between vein_mask and vein_pred for each run/approach/seed)

iou_by_approach = {label: [] for label in APPROACHES.values()}

for run_dir in sorted(glob.glob(results_path + "*/")):
    # identify approach from config yaml present in this run
    label = None
    for config_name, approach_label in APPROACHES.items():
        if os.path.exists(run_dir + config_name):
            label = approach_label
            break
    if label is None:
        continue

    pred_dir = run_dir + "vein_fl_preds/"
    if not os.path.exists(pred_dir):
        continue

    for mask_file in sorted(glob.glob(mask_path + "*.png")):
        pred_file = pred_dir + os.path.basename(mask_file)
        if not os.path.exists(pred_file):
            continue
        gt   = reduce_dim((np.array(Image.open(mask_file)) / 255) > 0.5)
        pred = reduce_dim((np.array(Image.open(pred_file)) / 255) > 0.5)
        iou_by_approach[label].append(iou(gt, pred))


# Boxplot of IoU for each approach (across seeds)

approach_labels = [label for label in APPROACHES.values() if iou_by_approach[label]]
approach_data   = [iou_by_approach[label] for label in approach_labels]

if approach_data:
    fig, ax = plt.subplots(figsize=[max(5, 3 * len(approach_labels)), 5])
    ax.boxplot(approach_data, tick_labels=approach_labels)
    plt.ylabel("IoU (vein mask vs. prediction)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0, 1])
    plt.grid(axis="y")
    plt.savefig(save_path + "figure_boxplot_iou.png", bbox_inches="tight", dpi=200)
    plt.show()