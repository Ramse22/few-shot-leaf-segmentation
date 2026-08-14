import os, glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

os.chdir(os.path.dirname(os.path.abspath(__file__)))

### Paths ###
results_path = "../results/"
mask_path    = "../data/vein_masks/"
logs_dir     = "../logs_marion/"
save_path    = "../figures_marion/"
os.makedirs(save_path, exist_ok=True)

APPROACHES = {
    "config.yaml":          "Our approach",
    "config_original.yaml": "Original",
}

AUTH_PRED_DIRS = {
    42: "../data_marion/vein_fl_preds_42/",
    43: "../data_marion/vein_fl_preds_43/",
    44: "../data_marion/vein_fl_preds_44/",
    45: "../data_marion/vein_fl_preds_45/",
    46: "../data_marion/vein_fl_preds_46/",
    47: "../data_marion/vein_fl_preds_47/",
}

AUTH_LOGS = {
    42: logs_dir + "vein_grower_42.txt",
    43: logs_dir + "vein_grower_43.txt",
    44: logs_dir + "vein_grower_44.txt",
    45: logs_dir + "vein_grower_45.txt",
    46: logs_dir + "vein_grower_46.txt",
    47: logs_dir + "vein_grower_47.txt",
}

def iou(a, b):
    i = (a.astype(bool) * b.astype(bool)).sum()
    u = (a.astype(bool) + b.astype(bool)).clip(0, 1).sum()
    return i / u

def reduce_dim(mask):
    return mask[:, :, 0] if len(mask.shape) == 3 else mask

def load_our_log(log_file):
    train, val = [], []
    with open(log_file) as f:
        for i, line in enumerate(f):
            if i == 0: continue
            parts = line.strip().split(",")
            train.append(float(parts[1]))
            val.append(float(parts[2]))
    return train, val

def load_auth_log(log_file):
    train, val = [], []
    with open(log_file) as f:
        for i, line in enumerate(f):
            if i == 0: continue
            parts = line.strip().split(",")
            train.append(float(parts[1]))
            val.append(float(parts[2]))
    return train, val

# ===== IoU =====

iou_by_approach = {label: [] for label in APPROACHES.values()}
iou_by_approach["Authors' method"] = []
iou_rows = []

for run_dir in sorted(glob.glob(results_path + "*/")):
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
        gt    = reduce_dim((np.array(Image.open(mask_file)) / 255) > 0.5)
        pred  = reduce_dim((np.array(Image.open(pred_file)) / 255) > 0.5)
        score = iou(gt, pred)
        iou_by_approach[label].append(score)
        iou_rows.append((label, os.path.basename(mask_file), f"{score:.6f}"))

for seed, pred_dir in AUTH_PRED_DIRS.items():
    if not os.path.exists(pred_dir):
        continue
    for mask_file in sorted(glob.glob(mask_path + "*.png")):
        pred_file = pred_dir + os.path.basename(mask_file)
        if not os.path.exists(pred_file):
            continue
        gt    = reduce_dim((np.array(Image.open(mask_file)) / 255) > 0.5)
        pred  = reduce_dim((np.array(Image.open(pred_file)) / 255) > 0.5)
        score = iou(gt, pred)
        iou_by_approach["Authors' method"].append(score)
        iou_rows.append(("Authors' method", os.path.basename(mask_file), f"{score:.6f}"))

with open(save_path + "iou_scores.csv", "w") as f:
    f.write("method,image,iou\n")
    for row in iou_rows:
        f.write(f"{row[0]},{row[1]},{row[2]}\n")
print(f"Saved {len(iou_rows)} IoU scores to {save_path}iou_scores.csv")

# ===== Loss and epochs per run =====

our_best_val, our_final_val, our_epochs = [], [], []
for run_dir in sorted(glob.glob(results_path + "*/")):
    for config_name in APPROACHES.keys():
        log_file = run_dir + "vein_grower.csv"
        if os.path.exists(run_dir + config_name) and os.path.exists(log_file):
            train, val = load_our_log(log_file)
            if val:
                our_best_val.append(min(val))
                our_final_val.append(val[-1])
                our_epochs.append(len(val))
            break

auth_best_val, auth_final_val, auth_epochs = [], [], []
for seed, log_file in AUTH_LOGS.items():
    if not os.path.exists(log_file):
        continue
    train, val = load_auth_log(log_file)
    if val:
        auth_best_val.append(min(val))
        auth_final_val.append(val[-1])
        auth_epochs.append(len(val))

# ===== FIGURE 1: IoU boxplot =====
approach_labels = [label for label in list(APPROACHES.values()) + ["Authors' method"] if iou_by_approach[label]]
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

# ===== FIGURE 2: Best val loss boxplot =====
fig, ax = plt.subplots(figsize=[8, 5])
ax.boxplot([our_best_val, auth_best_val], tick_labels=["Our approach", "Authors' method"])
plt.ylabel("Best validation loss", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="y")
plt.savefig(save_path + "figure_boxplot_best_val_loss.png", bbox_inches="tight", dpi=200)
plt.show()

# ===== FIGURE 3: Number of epochs boxplot =====
fig, ax = plt.subplots(figsize=[8, 5])
ax.boxplot([our_epochs, auth_epochs], tick_labels=["Our approach", "Authors' method"])
plt.ylabel("Number of epochs", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="y")
plt.savefig(save_path + "figure_boxplot_epochs.png", bbox_inches="tight", dpi=200)
plt.show()