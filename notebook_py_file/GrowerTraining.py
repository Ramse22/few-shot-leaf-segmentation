import os, sys, glob, pdb, random, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchinfo
from importlib import reload
import yaml
import torch.multiprocessing as mp

mp.set_sharing_strategy("file_system")

os.chdir(os.path.dirname(os.path.realpath(__file__)))

sys.path.append("../")
import utils.ImageLoader as ImageLoader
import utils.VeinGenerator as VeinGenerator
from utils.GetLowestGPU import GetLowestGPU
import utils.ModelWrapperGenerator as MW
import models.BuildCNN as BuildCNN

# Get config file from command line argument or use default
config_file = sys.argv[1] if len(sys.argv) > 1 else "../config.yaml"

with open(config_file, "r") as f:
    config = yaml.safe_load(f)

if "device" not in locals():
    device = torch.device(GetLowestGPU(verbose=2))

#### Load images ####

# options
# Load config values
image_path = config["data"]["image_path"]
mask_path = config["data"]["mask_path"]
roi_path = config["data"]["roi_path"]
image_extension = config["data"]["image_extension"]
mask_extension = config["data"]["mask_extension"]
roi_extension = config["data"]["roi_extension"]
window_size = config["vein_grower"]["window_size"]
verbose = True
plot = True
val_split = config["data"]["val_split"]
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

dilate = config["data"]["dilate"]
plot = True

# Get reproducible validation split from ImageLoader
val_img_idx = IL.val_img_idx

# instantiate data loaders
reload(VeinGenerator)
train_dataset = VeinGenerator.VeinGenerator(
    images=[images[i] for i in range(len(images)) if i not in val_img_idx],
    masks=[masks[i] for i in range(len(masks)) if i not in val_img_idx],
    rois=[rois[i] for i in range(len(rois)) if i not in val_img_idx],
    window_size=window_size,
    augment=True,
    dilate=dilate,
)
val_dataset = VeinGenerator.VeinGenerator(
    images=[images[i] for i in range(len(images)) if i in val_img_idx],
    masks=[masks[i] for i in range(len(masks)) if i in val_img_idx],
    rois=[rois[i] for i in range(len(rois)) if i in val_img_idx],
    window_size=window_size,
    augment=False,
    dilate=dilate,
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

# Load config parameters values
loss_type = config["loss"]["type"]
layers = config["vein_grower"]["layers"]
output_shape = config["vein_grower"]["output_shape"]
output_activation = torch.nn.Softmax2d()
learning_rate = config["optimizer"]["learning_rate"]
dropout_rate = config["vein_grower"]["dropout_rate"]
num_convs = config["vein_grower"]["num_convs"]

save_name = f"vein_grower_{loss_type}_{window_size}"

# initialize model
reload(BuildCNN)
cnn = BuildCNN.CNN(
    window_size=window_size,
    layers=layers,
    output_shape=output_shape,
    output_activation=output_activation,
    dropout_rate=dropout_rate,
    num_convs=num_convs,
).to(device)

opt = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Focal loss
if loss_type == "fl":
    gamma = config["loss"]["focal_loss"]["gamma"]
    alpha = config["loss"]["focal_loss"]["alpha"]

    def FocalLoss(pred, target):
        pred = pred.clamp(min=1e-7, max=1.0 - 1e-7)
        pt_1 = torch.where(target == 1, pred, torch.ones_like(pred))
        pt_0 = torch.where(target == 0, pred, torch.zeros_like(pred))
        out = -torch.mean(alpha * ((1.0 - pt_1) ** gamma) * torch.log(pt_1))
        out = out - torch.mean((1.0 - alpha) * (pt_0**gamma) * torch.log(1.0 - pt_0))
        return out

    loss_fn = FocalLoss
else:
    loss_fn = torch.nn.BCELoss()

# Learning rate scheduler (from yaml config)
scheduler_config = config["scheduler"]
scheduler_before_Plateau = {
    "mode": scheduler_config["mode"],
    "factor": scheduler_config["factor"],
    "patience": scheduler_config["patience"],
    "threshold": scheduler_config["threshold"],
}
if (
    "verbose"
    in torch.optim.lr_scheduler.ReduceLROnPlateau.__init__.__code__.co_varnames
):
    scheduler_before_Plateau["verbose"] = scheduler_config.get("verbose", True)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **scheduler_before_Plateau)

# wrap model
reload(MW)
model = MW.ModelWrapper(
    model=cnn,
    optimizer=opt,
    loss=loss_fn,
    scheduler=scheduler,
    save_name=f"{config['data']['weights_path']}{save_name}",
    log_name=f"{config['data']['logs_path']}{save_name}.txt",
    device=device,
    config_dict=config,
)

# model summary
torchinfo.summary(cnn, input_size=(1, 3, window_size, window_size), device=device)
model.use_amp = config["training"].get("use_amp", False)

#### run model ####

# Load config parameters values
epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
workers = config["training"]["workers"]
early_stopping = config["training"]["early_stopping"]

model.fit(
    train_dataset=train_dataset,
    validation_dataset=val_dataset,
    batch_size=batch_size,
    epochs=epochs,
    verbose=config["training"]["verbose"],
    early_stopping=early_stopping,
    workers=workers,
)

#### Plot ####

rel_save_thresh = 0.0

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
