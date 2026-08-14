import os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from skimage import measure

os.chdir(os.path.dirname(os.path.abspath(__file__)))

### Paths ###
results_path = "../results/"
mask_path    = "../data/vein_masks/"
image_path   = "../data/images/"
save_path    = "../figures_marion/"
os.makedirs(save_path, exist_ok=True)

AUTH_PRED_DIRS = {
    42: "../data_marion/vein_fl_preds_42/",
    43: "../data_marion/vein_fl_preds_43/",
    44: "../data_marion/vein_fl_preds_44/",
    45: "../data_marion/vein_fl_preds_45/",
    46: "../data_marion/vein_fl_preds_46/",
    47: "../data_marion/vein_fl_preds_47/",
}

APPROACHES = {
    "config.yaml":          "Our approach",
    "config_original.yaml": "Original",
}

def reduce_dim(mask):
    return mask[:, :, 0] if len(mask.shape) == 3 else mask

example_name   = "C_1_1_2_bot"
leaf_pred_path = "../data_marion/leaf_preds_jlag_sam3/"
rmin, rmax  = 600, 3200
cmin, cmax  = 600, 2000
figsize     = 5
n_tick      = 400
plot_width  = figsize
plot_height = (rmax - rmin) / (cmax - cmin) * figsize

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

    real_crop2 = None
    if os.path.exists(run42_real_file):
        real_arr2  = np.array(Image.open(run42_real_file), dtype=float)
        real_bin2  = reduce_dim(real_arr2) / 255 > 0.5
        real_crop2 = real_bin2[rmin2:rmax2, cmin2:cmax2]

    n_panels = 2 + (real_crop2 is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=[n_panels * plot_width, plot_height2], constrained_layout=True)

    panel_idx = 0
    plt.sca(axes[panel_idx])
    plt.imshow(image_crop2, vmin=0, vmax=1, extent=[cmin2, cmax2, rmax2, rmin2])
    axes[panel_idx].set_title("Base image", fontsize=15)
    panel_idx += 1

    if real_crop2 is not None:
        plot_real2 = image_crop2.copy()
        plot_real2[real_crop2] = [1, 0, 0]
        plt.sca(axes[panel_idx])
        plt.imshow(plot_real2, vmin=0, vmax=1, extent=[cmin2, cmax2, rmax2, rmin2])
        axes[panel_idx].set_title("+ ground truth mask (veins)", fontsize=15)
        panel_idx += 1

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


    def plot_mask_on_leaf(base_image, masks_colors, contour, title):
        plot_img = base_image.copy()
        for mask, color in masks_colors:
            plot_img[mask] = color
        fig, ax = plt.subplots(figsize=[plot_width, plot_height2], constrained_layout=True)
        plt.imshow(plot_img, vmin=0, vmax=1, extent=[cmin2, cmax2, rmax2, rmin2])
        if contour is not None:
            plt.plot(contour[:, 1] + cmin2, contour[:, 0] + rmin2, "k-", linewidth=2)
        ax.set_title(title, fontsize=15)
        ax.set_xticks([cmin2 + i * n_tick for i in range(int((cmax2 - cmin2) / n_tick) + 1)])
        ax.set_yticks([rmin2 + i * n_tick for i in range(int((rmax2 - rmin2) / n_tick) + 1)])
        ax.tick_params(labelsize=12)
        ax.set_xlim([cmin2, cmax2 - 1])
        ax.set_ylim([rmax2, rmin2])
        return fig, ax

    rmin3, rmax3 = 1700, 2500
    cmin3, cmax3 = 1100, 1500

    if real_crop2 is not None:
        both      = real_crop2 & vein_crop2
        mask_only = real_crop2 & ~vein_crop2
        pred_only = ~real_crop2 & vein_crop2

        fig, ax = plot_mask_on_leaf(
            image_crop2,
            [(mask_only, [0, 1, 0]), (pred_only, [1, 0, 0]), (both, [0, 0, 1])],
            contour2,
            "Ground truth vs. prediction comparison - run 42 (" + run42_example + ")",
        )
        legend_handles = [
            mpatches.Patch(color=[0, 0, 1], label="Les deux"),
            mpatches.Patch(color=[0, 1, 0], label="Masque seul"),
            mpatches.Patch(color=[1, 0, 0], label="Prediction seule"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=11, framealpha=0.9)
        rect = plt.Rectangle((cmin3, rmin3), cmax3 - cmin3, rmax3 - rmin3,
                              linewidth=2, edgecolor="black", facecolor="none", linestyle="-")
        ax.add_patch(rect)
        plt.savefig(save_path + "figure_run42_diff.png", bbox_inches="tight", dpi=200)
        plt.show()

        plot_mask_on_leaf(np.ones_like(image_crop2), [(both, [0, 0, 1])], contour2,
                          "Both (ground truth & prediction) - run 42 (" + run42_example + ")")
        plt.savefig(save_path + "figure_run42_both.png", bbox_inches="tight", dpi=200)
        plt.show()

        plot_mask_on_leaf(np.ones_like(image_crop2), [(mask_only, [0, 1, 0])], contour2,
                          "Mask only - run 42 (" + run42_example + ")")
        plt.savefig(save_path + "figure_run42_mask_only.png", bbox_inches="tight", dpi=200)
        plt.show()

        plot_mask_on_leaf(np.ones_like(image_crop2), [(pred_only, [1, 0, 0])], contour2,
                          "Prediction only - run 42 (" + run42_example + ")")
        plt.savefig(save_path + "figure_run42_pred_only.png", bbox_inches="tight", dpi=200)
        plt.show()

        both3      = both[rmin3 - rmin2:rmax3 - rmin2, cmin3 - cmin2:cmax3 - cmin2]
        mask_only3 = mask_only[rmin3 - rmin2:rmax3 - rmin2, cmin3 - cmin2:cmax3 - cmin2]
        pred_only3 = pred_only[rmin3 - rmin2:rmax3 - rmin2, cmin3 - cmin2:cmax3 - cmin2]
        image_crop3 = image_crop2[rmin3 - rmin2:rmax3 - rmin2, cmin3 - cmin2:cmax3 - cmin2]

        plot_width3  = 6
        plot_height3 = (rmax3 - rmin3) / (cmax3 - cmin3) * plot_width3

        plot_img3 = image_crop3.copy()
        plot_img3[mask_only3] = [0, 1, 0]
        plot_img3[pred_only3] = [1, 0, 0]
        plot_img3[both3]      = [0, 0, 1]

        fig, ax = plt.subplots(figsize=[plot_width3, plot_height3], constrained_layout=True)
        plt.imshow(plot_img3, vmin=0, vmax=1, extent=[cmin3, cmax3, rmax3, rmin3])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("black"); spine.set_linewidth(2)
        plt.savefig(save_path + "figure_run42_diff_zoom.png", bbox_inches="tight", dpi=300)
        plt.show()

        fig, ax = plt.subplots(figsize=[plot_width3, plot_height3], constrained_layout=True)
        plt.imshow(image_crop3, vmin=0, vmax=1, extent=[cmin3, cmax3, rmax3, rmin3])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("black"); spine.set_linewidth(2)
        plt.savefig(save_path + "figure_run42_diff_zoom_leaf.png", bbox_inches="tight", dpi=300)
        plt.show()

        vein_crop3 = vein_crop2[rmin3 - rmin2:rmax3 - rmin2, cmin3 - cmin2:cmax3 - cmin2]
        fig, ax = plt.subplots(figsize=[plot_width3, plot_height3], constrained_layout=True)
        plt.imshow(image_crop3, vmin=0, vmax=1, extent=[cmin3, cmax3, rmax3, rmin3])
        plt.imshow(vein_crop3, cmap="Reds", alpha=0.2, extent=[cmin3, cmax3, rmax3, rmin3])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("black"); spine.set_linewidth(2)
        plt.savefig(save_path + "figure_run42_zoom_pred_transparent_ours.png", bbox_inches="tight", dpi=300)
        plt.show()

        auth_pred_file = AUTH_PRED_DIRS[42] + run42_example + ".png"
        if os.path.exists(auth_pred_file):
            auth_arr2   = np.array(Image.open(auth_pred_file), dtype=float)
            auth_crop2  = (auth_arr2[:, :, 0] if auth_arr2.ndim == 3 else auth_arr2) > 128
            auth_crop2  = auth_crop2[rmin2:rmax2, cmin2:cmax2]

            auth_both      = real_crop2 & auth_crop2
            auth_mask_only = real_crop2 & ~auth_crop2
            auth_pred_only = ~real_crop2 & auth_crop2

            plot_mask_on_leaf(np.ones_like(image_crop2), [(auth_both, [0, 0, 1])], contour2,
                              "Both (ground truth & prediction) - Authors' method (" + run42_example + ")")
            plt.savefig(save_path + "figure_run42_both_authors.png", bbox_inches="tight", dpi=200)
            plt.show()

            plot_mask_on_leaf(np.ones_like(image_crop2), [(auth_mask_only, [0, 1, 0])], contour2,
                              "Mask only - Authors' method (" + run42_example + ")")
            plt.savefig(save_path + "figure_run42_mask_only_authors.png", bbox_inches="tight", dpi=200)
            plt.show()

            plot_mask_on_leaf(np.ones_like(image_crop2), [(auth_pred_only, [1, 0, 0])], contour2,
                              "Prediction only - Authors' method (" + run42_example + ")")
            plt.savefig(save_path + "figure_run42_pred_only_authors.png", bbox_inches="tight", dpi=200)
            plt.show()