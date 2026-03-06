import os, sys, glob, pdb, random, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from PIL import Image
from importlib import reload
from skimage import measure

sys.path.append('../')
import models.BuildCNN as BuildCNN
import models.VeinGrower as VeinGrower
from utils.GetLowestGPU import GetLowestGPU

if 'device' not in locals():
    device = torch.device(GetLowestGPU(verbose=2))

#### initialize grower ####

# options
window_size = 128
loss = 'fl' # 'fl' 'bce'
weights_path = f'../weights/vein_grower_{loss}_{window_size}_best_val_model.save'
layers = layers = [3, 32, 32, 32, 32, 64, 128]
output_shape = [2, 3, 3]
output_activation = torch.nn.Softmax2d()

# load CNN model
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


#### Grower inference ####

# options
image_path = '../data/images/'
roi_path = '../data_marion/leaf_preds/'
pred_path = f'../data_marion/vein_{loss}_preds/'
prob_path = f'../data_marion/vein_{loss}_probs/'
image_extension = 'jpeg'
roi_extension = 'png'
pred_extension = 'png'
prob_extension = 'png'
n_locs = 10000 # number of seed pixels
batch_size = 2048
threshold = None
post_process = True
max_number = 10 # number of images to segment, set to None for all images
verbose = True
save = True
show = True
fig_size = 15

# get image paths
image_names = [os.path.basename(f) for f in glob.glob(image_path+'*'+image_extension) if '_bot' in f]
image_names.sort()

# loop over all leaf images
for image_idx, image_name in enumerate(image_names):
    
    # don't exceed maximum
    if max_number is not None:
        if image_idx >= max_number:
            break
            
    # load image
    if verbose:
        print(f'Loading {image_name}...')
    image = np.array(Image.open(image_path + image_name), dtype=np.float32)/255
    if roi_path is not None:
        roi = np.array(Image.open(
            roi_path + image_name.replace(image_extension, roi_extension)), dtype=np.float32)/255
        roi = roi[:,:,0] > 0.5
    else:
        roi = None
    
    # segment the venation
    t0 = time.time()
    prob, mask = grower.grow(
        image=image, 
        roi=roi, 
        start_locs=None, 
        n_locs=n_locs, 
        batch_size=batch_size, 
        threshold=threshold,
        post_process=post_process)
    t1 = time.time()
    if verbose:
        print('Iteration completed in {0:1.2f} seconds'.format(t1-t0))
        
    # get positive class
    prob = prob[0]

    # save mask
    if save:
        if verbose: 
            print('Saving mask...')
        save_mask = np.concatenate([mask[:,:,None], mask[:,:,None], mask[:,:,None]], axis=-1)
        pil_mask = Image.fromarray(np.uint8(255*save_mask))
        name = pred_path + image_name.replace(image_extension, pred_extension)
        pil_mask.save(name, quality=100, subsampling=0)

    # save prob
    if save:
        if verbose: 
            print('Saving prob...')
        prob = prob[0] if len(prob.shape) == 3 else prob
        save_prob = np.concatenate([prob[:,:,None], prob[:,:,None], prob[:,:,None]], axis=-1)
        pil_prob = Image.fromarray(np.uint8(255*save_prob))
        name = prob_path + image_name.replace(image_extension, prob_extension)
        pil_prob.save(name, quality=100, subsampling=0)

    # plot overlay
    if show:
        if verbose: 
            print('Plotting overlay...')
        image[mask] = [1, 0, 0]
        fig = plt.figure(figsize=(image.shape[1]/image.shape[0]*fig_size, fig_size))
        plt.imshow(image)
        plt.show()
        
    if verbose: 
        print()

