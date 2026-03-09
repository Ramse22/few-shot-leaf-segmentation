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

os.chdir(os.path.dirname(os.path.realpath(__file__)))

sys.path.append('../')
import utils.ImageLoader as ImageLoader
import utils.UNetTileGenerator as UNetTileGenerator
from utils.GetLowestGPU import GetLowestGPU
import utils.ModelWrapperGenerator as MW
import models.BuildUNet as BuildUNet

if 'device' not in locals():
    device = torch.device(GetLowestGPU(verbose=2))

#### Load images ####

# options
image_path = '../data/images/'
mask_path = '../data/leaf_masks/'
image_extension = '.jpeg'
mask_extension = '.png'
window_size = 256
pad = True
verbose=True
plot = True
figsize = 5

# initialize loader
reload(ImageLoader)
IL = ImageLoader.ImageLoader(
    image_path=image_path, 
    mask_path=mask_path,  
    image_ext=image_extension,
    mask_ext=mask_extension,
    window_size=window_size, 
    pad=pad,
    verbose=verbose)

# load data
print('Loading data...'); time.sleep(0.3)
images, masks = IL.load_data()
file_names = IL.file_names

# plot
if plot:
    print('Plotting examples...')
    size = [images[0].shape[0], images[0].shape[1]]
    n_show = min(len(IL), 8)
    n_rows = np.ceil(n_show // 2)
    fig = plt.figure(figsize=(4*size[1]/size[0]*figsize, n_rows*figsize))
    for i in range(n_show):
        ax = fig.add_subplot(n_rows, 4, 2*i+1)
        ax.imshow(images[i], aspect='auto')
        ax.set_title(file_names[i])
        ax = fig.add_subplot(n_rows, 4, 2*i+2)
        ax.imshow(masks[i], aspect='auto', cmap='gray')
        ax.set_title('index = {0}'.format(i))
        ax.axis('off')
    fig.tight_layout(pad=3)
    fig.show()

#### Make Dataloader ####

# options
plot = True
verbose = True

# split images into train/val sets
val_img_idx = [8, 7, 19, 13, 16, 41, 26, 21, 36, 12]
train_img_idx = [i for i in range(len(masks)) if i not in val_img_idx]
print('Val index: {0}'.format(val_img_idx))
print()

# instantiate data loaders
reload(UNetTileGenerator)
train_dataset = UNetTileGenerator.UNetTileGenerator(
    images=[images[i] for i in train_img_idx], 
    masks=[masks[i] for i in train_img_idx], 
    window_size=window_size, 
    n_samples=250_000,
    augment=True,
    dilate=window_size,
    verbose=verbose)
val_dataset = UNetTileGenerator.UNetTileGenerator(
    images=[images[i] for i in val_img_idx], 
    masks=[masks[i] for i in val_img_idx], 
    window_size=window_size, 
    n_samples=50_000, 
    augment=False, 
    dilate=window_size, 
    verbose=verbose)
print('Train: {0:,}, Val: {1:,}'.format(len(train_dataset), len(val_dataset)))
print()

# plot example input/output tiles
if plot:
    print('Plotting training examples...')
    w = int(window_size/2/2)
    fig = plt.figure(figsize=(15, 15))
    N = len(train_dataset)
    for i in range(64):
        rand_idx = np.random.choice(N)
        tile, mask = train_dataset[rand_idx]
        tile = train_dataset.image2numpy(tile)
        mask = train_dataset.mask2numpy(mask)
        mask[0], mask[-1], mask[:,0], mask[:,-1] = 0, 0, 0, 0
        ax = fig.add_subplot(8, 8, i+1)
        ax.imshow(tile, aspect='auto')
        if mask.sum() > 0:
            contour = measure.find_contours(mask, 0.5)[0]
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='r')
        ax.axis('off')
    fig.tight_layout(pad=1)
    fig.show()

#### Train Leaf segmentation U-Net ####

# options
layers = [32, 32, 32, 32, 32, 64, 128]
save_name = f'leaf_unet_{window_size}'

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

# wrap model
reload(MW)
model = MW.ModelWrapper(
    model=unet,
    optimizer=opt,
    loss=torch.nn.BCELoss(),
    save_name=f'../weights_marion/{save_name}',
    log_name=f'../logs_marion/{save_name}.txt',
    device=device)

# model summary
torchinfo.summary(
    unet, 
    input_size=(1, 3, window_size, window_size), 
    device=device)

#### Run model ####

# options
epochs = 1000
batch_size = 128
workers = 64
early_stopping = 20

# train 
model.fit(
    train_dataset=train_dataset,
    validation_dataset=val_dataset,
    batch_size=batch_size,
    epochs=epochs,
    early_stopping=early_stopping,
    verbose=2,
    workers=workers)

rel_save_thresh = 0.0

# load errors
total_train_losses, total_val_losses = [], []
with open(model.log_name, 'r') as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        line = line.split(',')
        total_train_losses.append(float(line[1]))
        total_val_losses.append(float(line[2]))

# find where errors decreased
train_idx, train_loss, val_idx, val_loss = [], [], [], []
best_train, best_val = 1e12, 1e12
for i in range(len(total_train_losses)):
    rel_diff = (best_train - total_train_losses[i])
    rel_diff /= best_train
    if rel_diff > rel_save_thresh:
        best_train = total_train_losses[i]
        train_idx.append(i)
        train_loss.append(best_train)
    rel_diff = (best_val - total_val_losses[i])
    rel_diff /= best_val
    if rel_diff > rel_save_thresh:
        best_val = total_val_losses[i]
        val_idx.append(i)
        val_loss.append(best_val)
idx = np.argmin(val_loss)

# plot errors and improvements
fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(1, 2, 1)
plt.plot(total_train_losses, 'b')
plt.plot(total_val_losses, 'r')
plt.plot(val_idx[idx], val_loss[idx], 'ko')
plt.legend([r'Train error', r'Val error', 'Best model'])
plt.xlabel(r'Epochs')
plt.ylabel(r'Total Loss')
plt.title(r'Convergence')
plt.grid()
ax = fig.add_subplot(1, 2, 2)
plt.plot(train_idx, train_loss, 'b.-')
plt.plot(val_idx, val_loss, 'r.-')
plt.legend([r'Train error', r'Val error'])
plt.xlabel('Epochs')
plt.ylabel(r'Total Loss')
plt.title(r'Improvements')
plt.grid()
plt.tight_layout(h_pad=2, w_pad=2)
plt.show()

# plot log-scaled errors and improvements
fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(1, 2, 1)
plt.semilogy(total_train_losses, 'b')
plt.semilogy(total_val_losses, 'r')
plt.semilogy(val_idx[idx], val_loss[idx], 'ko')
plt.legend([r'Train error', r'Val error', 'Best model'])
plt.xlabel(r'Epochs')
plt.ylabel(r'Total Loss')
plt.title(r'Log Convergence')
plt.grid()
ax = fig.add_subplot(1, 2, 2)
plt.semilogy(train_idx, train_loss, 'b.-')
plt.semilogy(val_idx, val_loss, 'r.-')
plt.legend([r'Train error', r'Val error'])
plt.xlabel('Epochs')
plt.ylabel(r'Total Loss')
plt.title(r'Log Improvements')
plt.grid()
plt.tight_layout(h_pad=2, w_pad=2)
plt.show()

# load model weights
model.load_best_val(device=device)

# plot example inputs/outputs/predictions
fig = plt.figure(figsize=(15,15))
for i in range(64):
    
    # predict on random validation tile
    rand_idx = np.random.choice(len(val_dataset))
    tile, true = val_dataset[rand_idx]
    pred = model.predict(tile[None].to(device))[0]
    tile = val_dataset.image2numpy(tile)
    true = val_dataset.mask2numpy(true)
    pred = val_dataset.mask2numpy(pred) > 0.5
    true[0], true[-1], true[:,0], true[:,-1] = 0, 0, 0, 0
    pred[0], pred[-1], pred[:,0], pred[:,-1] = 0, 0, 0, 0

    # plot tile, ground truth, and prediction
    ax = fig.add_subplot(8, 8, i+1)
    plt.imshow(tile, aspect='auto')
    if true.sum() > 0:
        contour = measure.find_contours(true, 0.5)[0]
        ax.plot(contour[:, 1], contour[:, 0], '-', linewidth=2, color='r')
    if pred.sum() > 0:
        contour = measure.find_contours(pred, 0.5)[0]
        ax.plot(contour[:, 1], contour[:, 0], '--', linewidth=2, color='g')
    plt.axis('off')
    plt.xlim([0, window_size])
    plt.ylim([0, window_size])
    
plt.tight_layout(pad=0.2)
plt.show()


#### Evaluate trained model ####

# options
image_path = '../data/images/'
pred_path = '../data_marion/leaf_unet_preds/'
image_extension = 'jpeg'
pred_extension = 'png'
max_number = 10 # number of images to segment, set to None for all images
verbose = True
save = True
show = True
fig_size = 15

# book keeping
w = int(window_size/2)

# get image paths
image_names = [os.path.basename(f) for f in glob.glob(image_path + '*' + image_extension)]
image_names.sort()

# load model weights
model.load_best_val(device=device)

# loop over all leaf images
for image_idx, image_name in enumerate(tqdm(image_names)):
    
    # don't exceed maximum
    if max_number is not None:
        if image_idx >= max_number:
            break
            
    # load image
    if verbose:
        print(f'Loading {image_name}...')
    image = np.array(Image.open(image_path + image_name)) / 255
    
    # start timer
    t0 = time.time()

    # pad image
    image = np.pad(image, ((w, w), (w, w), (0, 0)), 'constant', constant_values=1)

    # initialize mask
    mask = np.zeros_like(image[:, :, 0])
    counts = np.zeros_like(image[:, :, 0])

    # loop through image in window_size / 2 steps
    for i in range(w, image.shape[0] - w, w):
        for j in range(w, image.shape[1] - w, w):

            # get window
            tile = image[i-w:i+w, j-w:j+w, :]
            tile = torch.from_numpy(tile).permute(2, 0, 1).float().to(device)

            # predict
            pred = model.predict(tile[None])[0]
            pred = pred.cpu().detach().numpy()[0] > 0.5
            pred = pred.astype(float)

            # add to mask
            mask[i-w:i+w, j-w:j+w] += pred
            counts[i-w:i+w, j-w:j+w] += 1

    # average mask
    mask /= counts.clip(min=1)

    # dilate and erode mask to fill gaps
    mask = ndimage.binary_dilation(mask.astype(bool), iterations=1)
    mask = ndimage.binary_erosion(mask, iterations=1)

    # choose largest connected component
    mask = measure.label(mask)
    mask = mask == np.argmax(np.bincount(mask.flat)[1:]) + 1

    # end timer
    t1 = time.time()
    if verbose:
        print(f'Finished in {t1-t0:.2f} seconds.')

    # remove padding
    image = image[w:-w, w:-w, :]
    mask = mask[w:-w, w:-w]

    # save mask
    if save:
        if verbose: 
            print('Saving mask...')
        save_mask = np.concatenate([mask[:,:,None], mask[:,:,None], mask[:,:,None]], axis=-1)
        pil_mask = Image.fromarray(np.uint8(255*save_mask))
        name = pred_path + image_name.replace(image_extension, pred_extension)
        pil_mask.save(name, quality=100, subsampling=0)

    # plot overlay
    if show:
        if max_number is None and image_idx % 100 != 0:
            continue
        if verbose: 
            print('Plotting overlay...')
        contour = measure.find_contours(mask, 0.5)[0] # [N, 2]
        fig = plt.figure(figsize=(image.shape[1]/image.shape[0]*fig_size, fig_size))
        plt.imshow(image)
        plt.plot(contour[:,1], contour[:,0], 'r-')
        plt.show()
        
    if verbose: 
        print()