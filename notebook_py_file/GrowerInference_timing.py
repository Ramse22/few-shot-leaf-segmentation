import os, sys, glob, time
import numpy as np
import torch
from PIL import Image
from importlib import reload
from datetime import datetime

sys.path.append('../')
import models.BuildCNN as BuildCNN
import models.VeinGrower as VeinGrower
from utils.GetLowestGPU import GetLowestGPU

if 'device' not in locals():
    device = torch.device(GetLowestGPU(verbose=2))

# options
window_size = 128
weights_path = '../weights_marion/vein_grower_43_best_val_model.save'
layers = [3, 32, 32, 32, 32, 64, 128]
output_shape = [2, 3, 3]
output_activation = torch.nn.Softmax2d()

# load model
print('loading cnn model...')
reload(BuildCNN)
model = BuildCNN.CNN(
    window_size=window_size,
    layers=layers,
    output_shape=output_shape,
    output_activation=output_activation).to(device)
weights = torch.load(weights_path, map_location=device)
model.load_state_dict(weights)
model.eval()

# initialize vein grower
print('initializing vein grower...')
reload(VeinGrower)
grower = VeinGrower.VeinGrower(
    window_size=window_size,
    model=model,
    device=device,
    verbose=True)

# options
image_path = '../data/images/'
roi_path = '../data/leaf_preds/'
image_extension = 'jpeg'
roi_extension = 'png'
n_locs = 10000
batch_size = 2048
threshold = None
post_process = True

# use first image only for timing
with open('../bot_images.txt') as f:
    image_names = [line.strip() for line in f.readlines()]
image_names.sort()
image_name = image_names[0]

print(f'Timing inference on: {image_name}')
image = np.array(Image.open(image_path + image_name), dtype=np.float32)/255
roi = np.array(Image.open(
    roi_path + image_name.replace(image_extension, roi_extension)), dtype=np.float32)/255
roi = roi[:,:,0] > 0.5

# time the inference
t0 = time.time()
prob, mask = grower.grow(
    image=image,
    roi=roi,
    start_locs=None,
    n_locs=n_locs,
    batch_size=batch_size,
    threshold=threshold,
    post_process=post_process)
elapsed = time.time() - t0

print(f"[TIMING][ORIGINAL INFERENCE] elapsed time: {elapsed:.3f} s")
print(f"[TIMING] n_locs={n_locs}, batch_size={batch_size}, window_size={window_size}, device={device}")

with open("timing_results.txt", "a") as f:
    f.write(
        f"ORIGINAL_INFERENCE,{datetime.now().isoformat()},{elapsed:.4f},"
        f"n_locs={n_locs},batch_size={batch_size},window_size={window_size},device={device}\n"
    )
