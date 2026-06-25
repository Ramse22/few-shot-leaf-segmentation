import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from importlib import reload

import torch.multiprocessing as mp

mp.set_sharing_strategy("file_system")

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../")

import utils.ImageLoader as ImageLoader
import utils.VeinGenerator as VeinGenerator
from utils.GetLowestGPU import GetLowestGPU
import models.BuildCNN as BuildCNN
from torch.utils.data import DataLoader

if "device" not in locals():
    device = torch.device(GetLowestGPU(verbose=2))

#### Load data ####

image_path = "../data/images/"
mask_path = "../data/vein_masks/"
roi_path = "../data/leaf_preds/"
window_size = 128

reload(ImageLoader)
IL = ImageLoader.ImageLoader(
    image_path=image_path,
    mask_path=mask_path,
    roi_path=roi_path,
    image_ext=".jpeg",
    mask_ext=".png",
    roi_ext=".png",
    window_size=window_size,
    verbose=False,
)
images, masks, rois = IL.load_data()
file_names = IL.file_names
rois = [(rois[i] + masks[i]).clip(0, 1) for i in range(len(rois))]

val_img_idx = [file_names.index(l) for l in ["C_1_14_18_bot.png", "C_1_8_1_bot.png"]]

reload(VeinGenerator)
train_dataset = VeinGenerator.VeinGenerator(
    images=[images[i] for i in range(len(images)) if i not in val_img_idx],
    masks=[masks[i] for i in range(len(masks)) if i not in val_img_idx],
    rois=[rois[i] for i in range(len(rois)) if i not in val_img_idx],
    window_size=window_size,
    augment=True,
    dilate=50,
)

#### Build model ####

reload(BuildCNN)
cnn = BuildCNN.CNN(
    window_size=window_size,
    layers=[3, 32, 32, 32, 32, 64, 128],
    output_shape=[2, 3, 3],
    output_activation=torch.nn.Softmax2d(),
).to(device)


def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    pred = pred.clamp(min=1e-7, max=1.0 - 1e-7)
    pt_1 = torch.where(target == 1, pred, torch.ones_like(pred))
    pt_0 = torch.where(target == 0, pred, torch.zeros_like(pred))
    return (
        -torch.mean(alpha * ((1.0 - pt_1) ** gamma) * torch.log(pt_1))
        - torch.mean((1.0 - alpha) * (pt_0**gamma) * torch.log(1.0 - pt_0))
    )


#### LR Finder (replicates Lightning's lr_find) ####

min_lr = 1e-8
max_lr = 1.0
num_steps = 100
batch_size = 256
beta = 0.98  # exponential smoothing factor (same as Lightning default)

loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
opt = torch.optim.Adam(cnn.parameters(), lr=min_lr)

lrs, losses, avg_loss = [], [], 0.0
best_loss = None
cnn.train()
data_iter = iter(loader)

print("Running LR finder...")
for step in range(num_steps):
    try:
        x, y = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        x, y = next(data_iter)

    lr = min_lr * (max_lr / min_lr) ** (step / num_steps)
    for pg in opt.param_groups:
        pg["lr"] = lr

    x = x.to(device).contiguous()
    y = y.to(device).contiguous()
    opt.zero_grad()
    loss = focal_loss(cnn(x), y)
    loss.backward()
    opt.step()

    # bias-corrected exponential smoothing (same as Lightning)
    avg_loss = beta * avg_loss + (1 - beta) * loss.item()
    smoothed = avg_loss / (1 - beta ** (step + 1))

    lrs.append(lr)
    losses.append(smoothed)

    if best_loss is None or smoothed < best_loss:
        best_loss = smoothed

    if smoothed > 4 * best_loss:
        print(f"Loss diverged at step {step}, stopping early.")
        break

print(f"Completed {len(lrs)} steps.")

#### Suggest LR (steepest descent, same as Lightning's .suggestion()) ####

losses_arr = np.array(losses)
lrs_arr = np.array(lrs)
suggested_idx = np.argmin(np.gradient(losses_arr))
suggested_lr = lrs_arr[suggested_idx]

print(f"\nSuggested LR: {suggested_lr:.2e}")
print(f"Usage: opt = torch.optim.Adam(cnn.parameters(), lr={suggested_lr:.2e})")

#### Plot ####

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(lrs_arr, losses_arr, linewidth=2, label="Loss (smoothed)")
ax.axvline(suggested_lr, color="red", linestyle="--", label=f"Suggested: {suggested_lr:.2e}")
ax.set_xscale("log")
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Loss")
ax.set_title("LR Finder")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lr_finder_results.png", dpi=150, bbox_inches="tight")
plt.show()
