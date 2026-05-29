import os, sys, time
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

if "device" not in locals():
    device = torch.device(GetLowestGPU(verbose=2))

#### Load images ####

image_path = "../data/images/"
mask_path = "../data/vein_masks/"
roi_path = "../data/leaf_preds/"
image_extension = ".jpeg"
mask_extension = ".png"
roi_extension = ".png"
window_size = 128

reload(ImageLoader)
IL = ImageLoader.ImageLoader(
    image_path=image_path,
    mask_path=mask_path,
    roi_path=roi_path,
    image_ext=image_extension,
    mask_ext=mask_extension,
    roi_ext=roi_extension,
    window_size=window_size,
    verbose=False,
)

images, masks, rois = IL.load_data()
file_names = IL.file_names
rois = [(rois[i] + masks[i]).clip(0, 1) for i in range(len(rois))]

#### Make data loader ####

val_img_idx = [file_names.index(l) for l in ["C_1_14_18_bot.png", "C_1_8_1_bot.png"]]
dilate = 50

reload(VeinGenerator)
train_dataset = VeinGenerator.VeinGenerator(
    images=[images[i] for i in range(len(images)) if i not in val_img_idx],
    masks=[masks[i] for i in range(len(masks)) if i not in val_img_idx],
    rois=[rois[i] for i in range(len(rois)) if i not in val_img_idx],
    window_size=window_size,
    augment=True,
    dilate=dilate,
)

#### Learning Rate Finder ####

initial_lr = 1e-5
final_lr = 1.0
num_iterations = 100
batch_size = 256

reload(BuildCNN)
cnn = BuildCNN.CNN(
    window_size=window_size,
    layers=[3, 32, 32, 32, 32, 64, 128],
    output_shape=[2, 3, 3],
    output_activation=torch.nn.Softmax2d(),
).to(device)
opt = torch.optim.Adam(cnn.parameters(), lr=initial_lr)

gamma, alpha = 2.0, 0.25


def FocalLoss(pred, target):
    pred = pred.clamp(min=1e-7, max=1.0 - 1e-7)
    pt_1 = torch.where(target == 1, pred, torch.ones_like(pred))
    pt_0 = torch.where(target == 0, pred, torch.zeros_like(pred))
    out = -torch.mean(alpha * ((1.0 - pt_1) ** gamma) * torch.log(pt_1))
    out = out - torch.mean((1.0 - alpha) * (pt_0**gamma) * torch.log(1.0 - pt_0))
    return out


from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=16
)

lrs = []
losses = []
best_loss = None

print("Running LR finder...")
cnn.train()
iteration = 0

for x_true, y_true in train_loader:
    if iteration >= num_iterations:
        break

    lr = initial_lr * (final_lr / initial_lr) ** (iteration / num_iterations)
    for param_group in opt.param_groups:
        param_group["lr"] = lr

    x_true = x_true.to(device).contiguous()
    y_true = y_true.to(device).contiguous()

    opt.zero_grad()
    y_pred = cnn(x_true)
    loss = FocalLoss(y_pred, y_true)
    loss.backward()
    opt.step()

    loss_value = loss.cpu().detach().numpy()
    lrs.append(lr)
    losses.append(loss_value)

    if best_loss is None:
        best_loss = loss_value
    else:
        # Stop if loss explodes (increases by 10x from best)
        if loss_value > 10 * best_loss:
            print(f"Loss exploded at iteration {iteration}.")
            break
        if loss_value < best_loss:
            best_loss = loss_value

    iteration += 1

print(f"Completed {iteration} iterations.\n")

#### Plot Results ####

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(lrs, losses, "b-", linewidth=2, label="Loss")
ax.set_xlabel("Learning Rate (log scale)", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Learning Rate Finder", fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xscale("log")
ax.legend()

plt.tight_layout()
plt.savefig("lr_finder_results.png", dpi=150, bbox_inches="tight")
plt.show()

#### Recommendations ####

# Find the learning rate with minimum loss (PyTorch Lightning approach)
best_loss_idx = np.argmin(losses)
best_loss_value = losses[best_loss_idx]
best_loss_lr = lrs[best_loss_idx]

# Recommended LR is typically 10x lower than the best LR found
# (to stay in the improving region but with margin)
recommended_lr = best_loss_lr / 10

print("=" * 60)
print(f"Best loss: {best_loss_value:.4e} at LR = {best_loss_lr:.2e}")
print(f"Recommended LR (10x lower): {recommended_lr:.2e}")
print(f"\nUsage in your main script:")
print(f"  opt = torch.optim.Adam(cnn.parameters(), lr={recommended_lr:.2e})")
print("=" * 60)
