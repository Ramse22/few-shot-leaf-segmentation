import os, sys, glob, pdb, random, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import measure
import torch
import torchinfo
from importlib import reload
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
from scipy.signal import find_peaks

os.chdir(os.path.dirname(os.path.realpath(__file__)))

sys.path.append("../")
import utils.ImageLoader as ImageLoader
import utils.UNetTileGenerator as UNetTileGenerator
from utils.GetLowestGPU import GetLowestGPU
import utils.ModelWrapperGenerator as MW
import models.BuildUNet as BuildUNet

if "device" not in locals():
    device = torch.device(GetLowestGPU([0], verbose=2))

#### Load Images ####
# options
image_path = "../data/images/"
mask_path = "../data/vein_masks/"
roi_path = "../data_marion/leaf_unet_preds/"
image_extension = ".jpeg"
mask_extension = ".png"
roi_extension = ".png"
window_size = 128
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
    plt.tight_layout(pad=0.5)
    plt.show()

#### Make data loader ####

# options
val_img_idx = [file_names.index(l) for l in ["C_1_14_18_bot.png", "C_1_8_1_bot.png"]]
dilate = 50
plot = True
verbose = True

# instantiate data loaders
reload(UNetTileGenerator)
train_dataset = UNetTileGenerator.UNetTileGenerator(
    images=[images[i] for i in range(len(images)) if i not in val_img_idx],
    masks=[masks[i] for i in range(len(masks)) if i not in val_img_idx],
    rois=[rois[i] for i in range(len(rois)) if i not in val_img_idx],
    window_size=window_size,
    n_samples=None,  # 7_000_000 250_000
    augment=True,
    dilate=dilate,
)
val_dataset = UNetTileGenerator.UNetTileGenerator(
    images=[images[i] for i in range(len(images)) if i in val_img_idx],
    masks=[masks[i] for i in range(len(masks)) if i in val_img_idx],
    rois=[rois[i] for i in range(len(rois)) if i in val_img_idx],
    window_size=window_size,
    n_samples=None,  # 2_500_000 50_000
    augment=False,
    dilate=dilate,
)
print("Train: {0:,}, Val: {1:,}".format(len(train_dataset), len(val_dataset)))
print()

# plot example input/output tiles
if plot:
    print("Plotting training examples...")
    w = int(window_size / 2 / 2)
    fig = plt.figure(figsize=(15, 15))
    N = len(train_dataset)
    for i in range(64):
        rand_idx = np.random.choice(N)
        tile, mask = train_dataset[rand_idx]
        tile = train_dataset.image2numpy(tile)
        mask = train_dataset.mask2numpy(mask)
        tile[mask == 1] = [1, 0, 0]
        ax = fig.add_subplot(8, 8, i + 1)
        ax.imshow(tile, aspect="auto")
        plt.axis("off")
    plt.tight_layout(pad=1)
    plt.show()

#### Train Leaf tracing CNN ####

# options
layers = [32, 32, 32, 32, 64, 128]
loss = "bce"  # 'fl' 'bce'
save_name = f"vein_unet_{loss}_{window_size}"

# initialize model and optimizer
reload(BuildUNet)
unet = BuildUNet.BuildUNet(
    layers=layers,
    input_channels=3,
    output_channels=1,
    hidden_activation=torch.nn.LeakyReLU(),
    output_activation=torch.nn.Sigmoid(),
    dropout_rate=0.0,
    num_convs=3,
).to(device)
opt = torch.optim.Adam(unet.parameters(), lr=1e-3)

# focal loss
gamma, alpha = 2.0, 0.25


def FocalLoss(pred, target):
    pred = pred.clamp(min=1e-7, max=1.0 - 1e-7)
    pt_1 = torch.where(target == 1, pred, torch.ones_like(pred))
    pt_0 = torch.where(target == 0, pred, torch.zeros_like(pred))
    out = -torch.mean(alpha * ((1.0 - pt_1) ** gamma) * torch.log(pt_1))
    out = out - torch.mean((1.0 - alpha) * (pt_0**gamma) * torch.log(1.0 - pt_0))
    return out


# wrap model
reload(MW)
model = MW.ModelWrapper(
    model=unet,
    optimizer=opt,
    loss=FocalLoss if loss == "fl" else torch.nn.BCELoss(),
    save_name=f"../weights_marion/{save_name}",
    log_name=f"../logs_marion/{save_name}.txt",
    device=device,
)

# model summary
torchinfo.summary(unet, input_size=(1, 3, window_size, window_size), device=device)

# run model
# options
epochs = 1000
batch_size = 512
workers = 128
early_stopping = 20

# train
model.fit(
    train_dataset=train_dataset,
    validation_dataset=val_dataset,
    batch_size=batch_size,
    epochs=epochs,
    early_stopping=early_stopping,
    verbose=2,
    workers=workers,
)

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
for i in range(64):
    # predict on random validation tile
    rand_idx = np.random.choice(len(val_dataset))
    tile, true = val_dataset[rand_idx]
    pred = model.predict(tile[None].to(device))[0]
    tile = val_dataset.image2numpy(tile)
    true = val_dataset.mask2numpy(true)
    pred = val_dataset.mask2numpy(pred) > 0.5

    tile[pred] = [1, 0, 0]

    # plot tile, ground truth, and prediction
    ax = fig.add_subplot(8, 8, i + 1)
    plt.imshow(tile, aspect="auto")
    plt.axis("off")
    plt.xlim([0, window_size])
    plt.ylim([0, window_size])

plt.tight_layout(pad=0.2)
plt.show()

#### Evaluate trained model ####

# options
loss = "bce"  # 'bce', 'fl'
image_path = "../data/images/"
roi_path = "../data_marion/leaf_unet_preds/"
pred_path = f"../data_marion/vein_unet_{loss}_preds/"
prob_path = f"../data_marion/vein_unet_{loss}_probs/"
image_extension = "jpeg"
roi_extension = "png"
pred_extension = "png"
max_number = 10  # number of images to segment, set to None for all images
verbose = True
save = False
show = True
fig_size = 15

# book keeping
w = int(window_size / 2)

# get image paths
image_names = [
    os.path.basename(f) for f in glob.glob(image_path + "*" + image_extension)
]
image_names = [i for i in image_names if "_bot" in i]
image_names.sort()

# load model weights
model.save_name = f"../weights_marion/vein_unet_{loss}_{window_size}"
model.log_name = f"../logs_marion/vein_unet_{loss}_{window_size}.txt"
model.load_best_val(device=device)

# loop over all leaf images
for image_idx, image_name in enumerate(tqdm(image_names)):
    # don't exceed maximum
    if max_number is not None:
        if image_idx >= max_number:
            break

    # load image
    if verbose:
        print(f"Loading {image_name}...")
    image = np.array(Image.open(image_path + image_name)) / 255
    roi = np.array(Image.open(roi_path + image_name.replace("jpeg", "png")))
    roi = (roi[:, :, 0] / 255) > 0.5

    # start timer
    t0 = time.time()

    # pad image
    image = np.pad(image, ((w, w), (w, w), (0, 0)), "constant", constant_values=1)
    roi = np.pad(roi, ((w, w), (w, w)), "constant", constant_values=0)

    # initialize mask
    prob = np.zeros_like(image[:, :, 0])
    counts = np.zeros_like(image[:, :, 0])

    # progress bar
    if verbose:
        ii = len(range(w, image.shape[0] - w, w))
        jj = len(range(w, image.shape[1] - w, w))
        pbar = tqdm(total=ii * jj)

    # loop through image in window_size/2 steps
    for i in range(w, image.shape[0] - w, w):
        for j in range(w, image.shape[1] - w, w):
            # get window
            tile = image[i - w : i + w, j - w : j + w, :]
            tile = torch.from_numpy(tile).permute(2, 0, 1).float().to(device)

            # predict
            pred = model.predict(tile[None])[0]
            pred = pred.cpu().detach().numpy()[0] > 0.5
            pred = pred.astype(float)

            # add to mask
            prob[i - w : i + w, j - w : j + w] += pred
            counts[i - w : i + w, j - w : j + w] += 1

            # update progress bar
            if verbose:
                pbar.update(1)

    # average probabilities
    prob = prob / counts.clip(min=1)

    # threshold probabilities for venation mask
    if verbose:
        print("Computing optimal threshold...")
    try:
        thresholds = np.linspace(0.1, 0.9, 101)
        structure = ndimage.generate_binary_structure(2, 2)
        n_objects, sizes = np.array(
            [
                [ndimage.label(prob > t, structure=structure)[1], (prob > t).sum()]
                for t in thresholds
            ]
        ).T
        peaks, _ = find_peaks(
            -n_objects / sizes, prominence=100 / sizes.max(), distance=10
        )
        threshold = thresholds[peaks[0]]
    except:
        threshold = 0.5
    mask = 1.0 * np.array(prob > threshold)

    # remove background artifacts
    leaf, petiole = mask.copy(), mask.copy()
    leaf[~roi], petiole[roi] = 0, 0
    petiole = measure.label(petiole)
    petiole = petiole == np.argmax(np.bincount(petiole.flat)[1:]) + 1
    mask = leaf + petiole

    # remove padding
    image = image[w:-w, w:-w, :]
    mask = mask[w:-w, w:-w]
    prob = prob[w:-w, w:-w]

    # end timer
    t1 = time.time()
    if verbose:
        print(f"Finished in {t1 - t0:.2f} seconds.")

    # save results
    if save:
        if verbose:
            print("Saving mask...")
        save_mask = np.concatenate(
            [mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=-1
        )
        pil_mask = Image.fromarray(np.uint8(255 * save_mask))
        name = pred_path + image_name.replace(image_extension, pred_extension)
        pil_mask.save(name, quality=100, subsampling=0)

        if verbose:
            print("Saving prob...")
        save_prob = np.concatenate(
            [prob[:, :, None], prob[:, :, None], prob[:, :, None]], axis=-1
        )
        pil_prob = Image.fromarray(np.uint8(255 * save_prob))
        name = prob_path + image_name.replace(image_extension, pred_extension)
        pil_prob.save(name, quality=100, subsampling=0)

    # plot overlay
    if show:
        if max_number is None and image_idx % 100 != 0:
            continue
        if verbose:
            print("Plotting overlay...")
        fig = plt.figure(figsize=(image.shape[1] / image.shape[0] * fig_size, fig_size))
        plot_image = image.copy()
        plot_image[mask == 1] = [1, 0, 0]
        plt.imshow(plot_image)
        plt.show()

    if verbose:
        print()
