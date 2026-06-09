import os, sys, glob, pdb, random, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchinfo
from importlib import reload
import yaml
import shutil
from pathlib import Path
from datetime import datetime


import torch.multiprocessing as mp

mp.set_sharing_strategy("file_system")

os.chdir(os.path.dirname(os.path.realpath(__file__)))

sys.path.append("../")
import utils.ImageLoader as ImageLoader
import utils.VeinGenerator as VeinGenerator
from utils.GetLowestGPU import GetLowestGPU
import utils.ModelWrapperGenerator as MW
import models.BuildCNN as BuildCNN

if "device" not in locals():
    device = torch.device(GetLowestGPU(verbose=2))

#### Load config ####

config_path = "../configs/config.yaml" #Path(sys.argv[1]) # when running from command line. Set string to config name to run locally.
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

run_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
weights_root = Path(config["data"]["weights_path"])
logs_root = Path(config["data"]["logs_path"])
weights_run_dir = weights_root / run_name
logs_run_dir = logs_root / run_name
weights_run_dir.mkdir(parents=True, exist_ok=True)
logs_run_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(config_path, weights_run_dir / Path(config_path).name)

#### Load images ####

# options
image_path = config["data"]["image_path"]
mask_path = config["data"]["mask_path"]
roi_path = config["data"]["roi_path"]
image_extension = config["data"]["image_extension"]
mask_extension = config["data"]["mask_extension"]
roi_extension = config["data"]["roi_extension"]
window_size = config["data"]["window_size"]
verbose = True
plot = True
figsize = 5

# initialize loader
reload(ImageLoader)
IL = ImageLoader.ImageLoader(
    image_path=image_path,
    mask_path=mask_path,
    roi_path=roi_path,
    image_ext=image_extension,
    mask_ext=mask_extension,
    roi_ext=roi_extension,
    window_size=window_size,
    verbose=verbose,
)

# load data
print("Loading data...")
time.sleep(0.3)
images, masks, rois = IL.load_data()
file_names = IL.file_names

# add mask to roi to include petiole
rois = [(rois[i] + masks[i]).clip(0, 1) for i in range(len(rois))]

# plot
if plot:
    print("Plotting examples...")
    size = [images[0].shape[0], images[0].shape[1]]
    fig = plt.figure(
        figsize=(6 * size[1] / size[0] * figsize, np.ceil(len(IL) / 2) * figsize)
    )
    for i in range(len(IL)):
        ax = fig.add_subplot(int(np.ceil(len(IL) / 2)), 6, 3 * i + 1)
        plt.imshow(images[i], aspect="auto")
        ax = fig.add_subplot(int(np.ceil(len(IL) / 2)), 6, 3 * i + 2)
        plt.imshow(masks[i], aspect="auto", cmap="gray")
        plt.title(file_names[i] + ", index = {0}".format(i))
        ax = fig.add_subplot(int(np.ceil(len(IL) / 2)), 6, 3 * i + 3)
        plt.imshow(rois[i], aspect="auto", cmap="gray")
    plt.tight_layout(pad=3)
    plt.show()


#### Make data loader ####

# options
split_cfg = config["data_split"]
random_split_cfg = split_cfg.get("random_split")
if random_split_cfg is not None:
    seed = int(random_split_cfg.get("seed", 42))
    val_fraction = float(random_split_cfg.get("val_fraction", 0.2))
    val_count = int(np.ceil(len(file_names) * val_fraction))
    val_count = max(1, min(val_count, max(1, len(file_names) - 1)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(file_names))
    val_img_idx = sorted(perm[:val_count].tolist())
    print(
        f"Random split with seed={seed}: val={len(val_img_idx)}, train={len(file_names) - len(val_img_idx)}"
    )
else:
    val_names = split_cfg.get("val_img_names", [])
    missing_names = [name for name in val_names if name not in file_names]
    if len(missing_names) > 0:
        raise ValueError(
            "Some val_img_names are missing from loaded files: "
            + ", ".join(missing_names)
        )
    val_img_idx = [file_names.index(name) for name in val_names]

dilate = split_cfg["dilate"]
plot = split_cfg["plot"]
augment = split_cfg.get("augment", True)
n_samples = split_cfg.get("n_samples", None)

# instantiate data loaders
reload(VeinGenerator)
train_dataset = VeinGenerator.VeinGenerator(
    images=[images[i] for i in range(len(images)) if i not in val_img_idx],
    masks=[masks[i] for i in range(len(masks)) if i not in val_img_idx],
    rois=[rois[i] for i in range(len(rois)) if i not in val_img_idx],
    window_size=window_size,
    augment=augment,
    dilate=dilate,
    n_samples=n_samples,
)
val_dataset = VeinGenerator.VeinGenerator(
    images=[images[i] for i in range(len(images)) if i in val_img_idx],
    masks=[masks[i] for i in range(len(masks)) if i in val_img_idx],
    rois=[rois[i] for i in range(len(rois)) if i in val_img_idx],
    window_size=window_size,
    augment=False,
    dilate=dilate,
    n_samples=n_samples,
)
print("Train: {0:,}, Val: {1:,}".format(len(train_dataset), len(val_dataset)))
print()

# plot example input/output tiles
if plot:
    print("Plotting training examples:")
    fig = plt.figure(figsize=(15, 15))
    N = len(train_dataset)
    for i in range(32):
        rand_idx = np.random.choice(N)
        input, output = train_dataset[rand_idx]
        ax = fig.add_subplot(8, 8, 2 * i + 1)
        plt.imshow(train_dataset.image2numpy(input), vmin=0, vmax=1)
        plt.plot([62, 66, 66, 62, 62], [62, 62, 66, 66, 62], "k-", linewidth=1)
        ax = fig.add_subplot(8, 8, 2 * i + 2)
        plt.imshow(train_dataset.mask2numpy(output), cmap="gray", vmin=0, vmax=1)
        plt.plot([0.5, 0.5], [-0.5, 2.5], c="gray")
        plt.plot([1.5, 1.5], [-0.5, 2.5], c="gray")
        plt.plot([-0.5, 2.5], [0.5, 0.5], c="gray")
        plt.plot([-0.5, 2.5], [1.5, 1.5], c="gray")
    plt.tight_layout(pad=0.5)
    plt.show()


#### Train vein growing CNN ####

# options
loss = config["model"]["loss"]
layers = config["model"]["layers"]
output_shape = config["model"]["output_shape"]
output_activation = getattr(torch.nn, config["model"]["output_activation"])()
save_name = config["model"]["save_name"]
dropout = config["model"]["dropout"]

# initialize model and optimizer
reload(BuildCNN)
cnn = BuildCNN.CNN(
    window_size=window_size,
    layers=layers,
    output_shape=output_shape,
    output_activation=output_activation,
    dropout_rate=dropout,
).to(device)

opt_cfg = config.get("optimizer", {})
opt = torch.optim.Adam(
    cnn.parameters(),
    lr=float(opt_cfg.get("opt_lr", opt_cfg.get("lr", 5e-3))),
)


# focal loss
gamma = config["focal_loss"]["gamma"]
alpha = config["focal_loss"]["alpha"]


def FocalLoss(pred, target):
    pred = pred.clamp(min=1e-7, max=1.0 - 1e-7)
    pt_1 = torch.where(target == 1, pred, torch.ones_like(pred))
    pt_0 = torch.where(target == 0, pred, torch.zeros_like(pred))
    out = -torch.mean(alpha * ((1.0 - pt_1) ** gamma) * torch.log(pt_1))
    out = out - torch.mean((1.0 - alpha) * (pt_0**gamma) * torch.log(1.0 - pt_0))
    return out


if loss == "fl":
    loss_fn = FocalLoss
elif loss == "bce":
    loss_fn = torch.nn.BCELoss()
else:
    raise ValueError(f"Unsupported loss '{loss}'. Expected 'fl' or 'bce'.")

# Reduce learning rate when validation loss plateaus
scheduler_before_Plateau = {
    "mode": config["scheduler"]["mode"],
    "factor": float(config["scheduler"]["factor"]),
    "patience": int(config["scheduler"]["patience"]),
    "threshold": float(config["scheduler"]["threshold"]),
}
if (
    "verbose"
    in torch.optim.lr_scheduler.ReduceLROnPlateau.__init__.__code__.co_varnames
):
    scheduler_before_Plateau["verbose"] = True

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **scheduler_before_Plateau)

# wrap model
reload(MW)
model = MW.ModelWrapper(
    model=cnn,
    optimizer=opt,
    loss=loss_fn,
    scheduler=scheduler,
    save_name=str(weights_run_dir / save_name),
    log_name=str(logs_run_dir / f"{save_name}.csv"),
    device=device,
)

# model summary
torchinfo.summary(cnn, input_size=(1, 3, window_size, window_size), device=device)


#### run model ####

epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
workers = config["training"]["workers"]
early_stopping = config["training"]["early_stopping"]

model.fit(
    train_dataset=train_dataset,
    validation_dataset=val_dataset,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
    early_stopping=early_stopping,
    workers=workers,
)

#### Plot ####

rel_save_thresh = config["training"]["rel_save_thresh"]

# load errors
total_train_losses, total_val_losses = [], []
with open(model.log_name, "r") as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        line = line.split(",")
        total_train_losses.append(float(line[1]))
        total_val_losses.append(float(line[2]))

# find where errors decreased
train_idx, train_loss, val_idx, val_loss = [], [], [], []
best_train, best_val = 1e12, 1e12
for i in range(len(total_train_losses)):
    rel_diff = best_train - total_train_losses[i]
    rel_diff /= best_train
    if rel_diff > rel_save_thresh:
        best_train = total_train_losses[i]
        train_idx.append(i)
        train_loss.append(best_train)
    rel_diff = best_val - total_val_losses[i]
    rel_diff /= best_val
    if rel_diff > rel_save_thresh:
        best_val = total_val_losses[i]
        val_idx.append(i)
        val_loss.append(best_val)
idx = np.argmin(val_loss)

# plot errors and improvements
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 2, 1)
plt.plot(total_train_losses, "b")
plt.plot(total_val_losses, "r")
plt.plot(val_idx[idx], val_loss[idx], "ko")
plt.legend([r"Train error", r"Val error", "Best model"])
plt.xlabel(r"Epochs")
plt.ylabel(r"Total Loss")
plt.title(r"Convergence")
plt.grid()
ax = fig.add_subplot(1, 2, 2)
plt.plot(train_idx, train_loss, "b.-")
plt.plot(val_idx, val_loss, "r.-")
plt.legend([r"Train error", r"Val error"])
plt.xlabel("Epochs")
plt.ylabel(r"Total Loss")
plt.title(r"Improvements")
plt.grid()
plt.tight_layout(h_pad=2, w_pad=2)
plt.show()

# plot log-scaled errors and improvements
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 2, 1)
plt.semilogy(total_train_losses, "b")
plt.semilogy(total_val_losses, "r")
plt.semilogy(val_idx[idx], val_loss[idx], "ko")
plt.legend([r"Train error", r"Val error", "Best model"])
plt.xlabel(r"Epochs")
plt.ylabel(r"Total Loss")
plt.title(r"Log Convergence")
plt.grid()
ax = fig.add_subplot(1, 2, 2)
plt.semilogy(train_idx, train_loss, "b.-")
plt.semilogy(val_idx, val_loss, "r.-")
plt.legend([r"Train error", r"Val error"])
plt.xlabel("Epochs")
plt.ylabel(r"Total Loss")
plt.title(r"Log Improvements")
plt.grid()
plt.tight_layout(h_pad=2, w_pad=2)
plt.show()


# load model weights
model.load_best_val(device=device)

# plot example inputs/outputs/predictions
fig = plt.figure(figsize=(15, 15))
for i in range(9 * 3):
    # predict on random validation tile
    rand_idx = np.random.choice(len(val_dataset))
    tile, true = val_dataset[rand_idx]
    pred = model.predict(tile[None].to(device))[0]
    tile = val_dataset.image2numpy(tile)
    true = true.detach().cpu().numpy()[0]
    pred = pred.detach().cpu().numpy()[0]

    # apply thresholds to probabilities
    n_chunks = 5
    pred = pred[:, :, None] > np.arange(1, n_chunks + 1)[None, None] / n_chunks
    pred = pred.sum(-1) / (n_chunks - 1)

    # plot tile, ground truth, and prediction
    ax = fig.add_subplot(9, 9, i * 3 + 1)
    plt.imshow(tile, vmin=0, vmax=1)
    plt.plot([62, 66, 66, 62, 62], [62, 62, 66, 66, 62], "k-", linewidth=1)
    if i // 3 == 0:
        plt.title("Input tile")

    ax = fig.add_subplot(9, 9, i * 3 + 2)
    plt.imshow(true, cmap="gray", vmin=0, vmax=1)
    plt.plot([0.5, 0.5], [-0.5, 2.5], c="gray")
    plt.plot([1.5, 1.5], [-0.5, 2.5], c="gray")
    plt.plot([-0.5, 2.5], [0.5, 0.5], c="gray")
    plt.plot([-0.5, 2.5], [1.5, 1.5], c="gray")
    if i // 3 == 0:
        plt.title("Ground truth")

    ax = fig.add_subplot(9, 9, i * 3 + 3)
    plt.imshow(pred, cmap="plasma", vmin=0, vmax=1)
    plt.plot([0.5, 0.5], [-0.5, 2.5], c="gray")
    plt.plot([1.5, 1.5], [-0.5, 2.5], c="gray")
    plt.plot([-0.5, 2.5], [0.5, 0.5], c="gray")
    plt.plot([-0.5, 2.5], [1.5, 1.5], c="gray")
    if i // 3 == 0:
        plt.title("Prediction")

plt.tight_layout(pad=0.5)
plt.show()
