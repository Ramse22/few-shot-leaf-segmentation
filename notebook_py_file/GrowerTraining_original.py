# %%
import os, sys, glob, pdb, random, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchinfo
from importlib import reload
from datetime import datetime

os.chdir(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import utils.ImageLoader as ImageLoader
import utils.VeinGenerator as VeinGenerator
from utils.GetLowestGPU import GetLowestGPU
import utils.ModelWrapperGenerator as MW
import models.BuildCNN as BuildCNN

if 'device' not in locals():
    device = torch.device(GetLowestGPU(verbose=2))

# options
image_path = '../data/images/'
mask_path = '../data/vein_masks/'
roi_path = '../data/leaf_preds/'
image_extension = '.jpeg'
mask_extension = '.png'
roi_extension = '.png'
window_size = 128
verbose=True
plot = False  # disabled for timing test
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
    verbose=verbose)

# load data
print('Loading data...'); time.sleep(0.3)
images, masks, rois = IL.load_data()
file_names = IL.file_names

# add mask to roi to include petiole
rois = [(rois[i]+masks[i]).clip(0, 1) for i in range(len(rois))]

# plot (disabled for timing test)
if plot:
    print('Plotting examples...')
    size = [images[0].shape[0], images[0].shape[1]]
    fig = plt.figure(figsize=(6*size[1]/size[0]*figsize, np.ceil(len(IL)/2)*figsize))
    for i in range(len(IL)):
        ax = fig.add_subplot(int(np.ceil(len(IL)/2)), 6, 3*i+1)
        plt.imshow(images[i], aspect='auto')
        ax = fig.add_subplot(int(np.ceil(len(IL)/2)), 6, 3*i+2)
        plt.imshow(masks[i], aspect='auto', cmap='gray')
        plt.title(file_names[i] + ', index = {0}'.format(i))
        ax = fig.add_subplot(int(np.ceil(len(IL)/2)), 6, 3*i+3)
        plt.imshow(rois[i], aspect='auto', cmap='gray')
    plt.tight_layout(pad=0.5)
    plt.show()

# options
seed = 41
val_fraction = 0.2
val_count = int(np.ceil(len(file_names) * val_fraction))
val_count = max(1, min(val_count, max(1, len(file_names) - 1)))
rng = np.random.default_rng(seed)
perm = rng.permutation(len(file_names))
val_img_idx = sorted(perm[:val_count].tolist())
dilate = 50
plot = False  # disabled for timing test

# instantiate data loaders
reload(VeinGenerator)
train_dataset = VeinGenerator.VeinGenerator(
    images=[images[i] for i in range(len(images)) if i not in val_img_idx], 
    masks=[masks[i] for i in range(len(masks)) if i not in val_img_idx], 
    rois=[rois[i] for i in range(len(rois)) if i not in val_img_idx], 
    window_size=window_size, 
    augment=True,
    dilate=dilate)
val_dataset = VeinGenerator.VeinGenerator(
    images=[images[i] for i in range(len(images)) if i in val_img_idx], 
    masks=[masks[i] for i in range(len(masks)) if i in val_img_idx], 
    rois=[rois[i] for i in range(len(rois)) if i in val_img_idx], 
    window_size=window_size, 
    augment=False,
    dilate=dilate)
print('Train: {0:,}, Val: {1:,}'.format(len(train_dataset), len(val_dataset)))
print()

# plot example input/output tiles (disabled for timing test)
if plot:
    print('Plotting training examples:')
    fig = plt.figure(figsize=(15, 15))
    N = len(train_dataset)
    for i in range(32):
        rand_idx = np.random.choice(N)
        input, output = train_dataset[rand_idx]
        ax = fig.add_subplot(8, 8, 2*i+1)
        plt.imshow(train_dataset.image2numpy(input), vmin=0, vmax=1)
        plt.plot([62, 66, 66, 62, 62], [62, 62, 66, 66, 62], 'k-', linewidth=1)
        ax = fig.add_subplot(8, 8, 2*i+2)
        plt.imshow(train_dataset.mask2numpy(output), cmap='gray', vmin=0, vmax=1)
        plt.plot([0.5, 0.5], [-0.5, 2.5], c='gray')
        plt.plot([1.5, 1.5], [-0.5, 2.5], c='gray')
        plt.plot([-0.5, 2.5], [0.5, 0.5], c='gray')
        plt.plot([-0.5, 2.5], [1.5, 1.5], c='gray')
    plt.tight_layout(pad=0.5)
    plt.show()

# options
loss = 'fl' # 'fl' 'bce'
layers = [3, 32, 32, 32, 32, 64, 128]
output_shape = [2, 3, 3]
output_activation = torch.nn.Softmax2d()
save_name = f'vein_grower_{seed}_TIMING_TEST'

# initialize model and optimizer
reload(BuildCNN)
cnn = BuildCNN.CNN(
    window_size=window_size, 
    layers=layers,
    output_shape=output_shape,
    output_activation=output_activation).to(device)
opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)

# focal loss
gamma, alpha = 2.0, 0.25
def FocalLoss(pred, target):
    pred = pred.clamp(min=1e-7, max=1.0-1e-7)
    pt_1 = torch.where(target == 1, pred, torch.ones_like(pred))
    pt_0 = torch.where(target == 0, pred, torch.zeros_like(pred))
    out = -torch.mean(alpha*((1.0 - pt_1)**gamma)*torch.log(pt_1))
    out = out - torch.mean((1.0 - alpha)*(pt_0**gamma)*torch.log(1.0 - pt_0))
    return out

# wrap model
reload(MW)
model = MW.ModelWrapper(
    model=cnn,
    optimizer=opt,
    loss=FocalLoss,
    save_name=f'../weights_marion/{save_name}',
    log_name=f'../logs_marion/{save_name}.txt',
    device=device)

# --- TIMING TEST OVERRIDE ---
# Checkpoint resuming disabled on purpose: we want a clean 1-epoch run from
# scratch every time, not a resume from a previous partial run.
checkpoint_path = f'../weights_marion/{save_name}_checkpoint.pt'
initial_epoch = 0
best_val_loss = None
RESUME_FROM_CHECKPOINT = False
if RESUME_FROM_CHECKPOINT and os.path.exists(checkpoint_path):
    print(f'Resuming from checkpoint...')
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cnn.load_state_dict(ckpt['model_state_dict'])
    opt.load_state_dict(ckpt['optimizer_state_dict'])
    initial_epoch = ckpt['epoch'] + 1
    best_val_loss = ckpt['best_val_loss']
    print(f'Resuming from epoch {initial_epoch}')
# -----------------------------

# model summary
torchinfo.summary(
    cnn, 
    input_size=(1, 3, window_size, window_size), 
    device=device)

epochs = 1  # TIMING TEST: forced to 1 (was 1000)
batch_size = 512 #originally 1024
workers = 16
early_stopping = 10

class CheckpointCallback:
    on_train_begin = False
    on_train_end = False
    on_epoch_begin = False
    on_batch_begin = False
    on_batch_end = False
    on_epoch_end = True

    def __call__(self, wrapper):
        best_val = min(wrapper.val_loss_list) if wrapper.val_loss_list else 1e12
        torch.save({
            'epoch': initial_epoch + len(wrapper.train_loss_list) - 1,
            'model_state_dict': wrapper.model.state_dict(),
            'optimizer_state_dict': wrapper.optimizer.state_dict(),
            'best_val_loss': best_val,
        }, checkpoint_path)

#### run model (TIMED) ####

if device.type == 'cuda':
    torch.cuda.synchronize(device)
t0 = time.perf_counter()

model.fit(
    train_dataset=train_dataset,
    validation_dataset=val_dataset,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
    early_stopping=early_stopping,
    workers=workers,
    initial_epoch=initial_epoch,
    best_val_loss=best_val_loss,
    callbacks=[CheckpointCallback()])

if device.type == 'cuda':
    torch.cuda.synchronize(device)
t1 = time.perf_counter()

elapsed = t1 - t0
print("=" * 60)
print(f"[TIMING][ORIGINAL SCRIPT] {epochs} epoch(s) elapsed time: {elapsed:.3f} s")
print(f"[TIMING][ORIGINAL SCRIPT] batch_size={batch_size}, workers={workers}, window_size={window_size}, device={device}")
print("=" * 60)

with open("timing_results.txt", "a") as f:
    f.write(
        f"ORIGINAL,{datetime.now().isoformat()},{elapsed:.4f},epochs={epochs},"
        f"batch_size={batch_size},workers={workers},window_size={window_size},device={device}\n"
    )

# Plotting/analysis sections skipped for timing test.
