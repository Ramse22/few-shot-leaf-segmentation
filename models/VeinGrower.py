import os, sys, pdb
import torch
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from torch.utils.data import Dataset, DataLoader


class TileGenerator(Dataset):
    def __init__(self, image, indices, w):

        super().__init__()
        self.image = image
        self.indices = indices
        self.w = w

        # ensure indices are in correct format
        if len(self.indices) == 2:
            try:  # if this works, do nothing
                idx = self.indices[0]
                i, j = idx[0], idx[1]
            except:  # if not, restructure
                i = self.indices[0][0]
                j = self.indices[1][0]
                self.indices = np.array([[i, j]])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = torch.tensor(self.indices[index], dtype=torch.long)
        i, j = idx[0], idx[1]

        # Clamp indices to image bounds
        i_start = max(0, i - self.w)
        i_end = min(self.image.shape[1], i + self.w)
        j_start = max(0, j - self.w)
        j_end = min(self.image.shape[2], j + self.w)

        tile = self.image[:, i_start:i_end, j_start:j_end]

        # Pad if too small
        expected_size = 2 * self.w
        if tile.shape[1] < expected_size or tile.shape[2] < expected_size:
            pad_h = expected_size - tile.shape[1]
            pad_w = expected_size - tile.shape[2]
            tile = np.pad(tile, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")

        x = torch.tensor(tile, dtype=torch.float)
        return idx, x


class VeinGrower:
    """
    Grows region on an image given starting location(s). Stops when no more
    pixels are added to region.

    Args:
        model    (callable): initialized CNN in eval mode
        window_size   (int): height and width of input tile
        device     (device): torch device
        verbose      (bool): whether to print progress updates

    Inputs:
        image       (array): image with shape (3, H, W)
        roi         (array): optional
        start_locs  (array): seed pixels with shape (N, 2)
        n_locs        (int): number of seed pixels to sample
        batch_size    (int): model prediciton batch size
        threshold   (float): mask probability threshold (default None)
        post_process (bool): whether to clean up the segmentation mask

    Returns:
        probs      (tensor): probability map (H, W)
        veins      (tensor): thresholded probabilities (H, W)
    """

    def __init__(self, model, window_size, device=None, verbose=False):

        super().__init__()
        self.window_size = window_size
        self.model = model
        self.device = device
        self.verbose = verbose

    # swap image axes [H, W, C] -> [C, H, W]
    def channels_first(self, image):
        return np.swapaxes(np.swapaxes(image, 0, 2), 1, 2)

    # swap image axes [C, H, W] -> [H, W, C]
    def channels_last(self, image):
        return np.swapaxes(np.swapaxes(image, 0, 2), 0, 1)

    def grow(
        self,
        image,
        roi=None,
        start_locs=None,
        n_locs=10000,
        batch_size=1024,
        threshold=None,
        post_process=True,
    ):

        # pad image and mask with zeros
        w = int(self.window_size / 2)
        image = np.pad(
            image, [[w, w], [w, w], [0, 0]], "constant", constant_values=1.0
        ).copy()
        if roi is not None:
            roi = np.pad(roi, [[w, w], [w, w]], "constant", constant_values=0.0).copy()

        # reorder image channels
        image = self.channels_first(image)

        # initialize prediction accumulator
        mask = np.zeros((2, image.shape[-2], image.shape[-1]))

        # all non-padding pixel locations — single pass replaces iterative growing
        valid = np.zeros((image.shape[-2], image.shape[-1]))
        valid[w:-w, w:-w] = 1
        all_locs = np.argwhere(valid == 1)

        # single inference pass over all pixels
        tile_generator = TileGenerator(image, all_locs, w)
        tile_batch_loader = DataLoader(
            tile_generator, batch_size=batch_size, shuffle=False, num_workers=0
        )

        n_batches = len(tile_batch_loader)
        for b_idx, (idx_batch, tile_batch) in enumerate(tile_batch_loader):
            idx_batch = idx_batch.detach().cpu().numpy()
            tile_batch = tile_batch.to(self.device)
            with torch.no_grad():
                pred_batch = self.model(tile_batch).detach().cpu().numpy()
            for idx, pred in zip(idx_batch, pred_batch):
                i, j = idx[0], idx[1]
                mask[:, i - 1 : i + 2, j - 1 : j + 2] += pred

            if self.verbose:
                p = "\rBatch {0}/{1}".format(b_idx + 1, n_batches)
                sys.stdout.write(p)

        if self.verbose:
            print()

        # normalize mask probabilities
        probs = mask / ((mask[0:1] + mask[1:2]).clip(1.0, np.inf))

        # threshold probabilities for venation mask
        if self.verbose:
            print("Computing optimal threshold...")
        if threshold is None:
            thresholds = np.linspace(0.1, 0.9, 101)
            structure = ndimage.generate_binary_structure(2, 2)
            n_objects, sizes = np.array(
                [
                    [
                        ndimage.label(probs[0] > t, structure=structure)[1],
                        (probs[0] > t).sum(),
                    ]
                    for t in thresholds
                ]
            ).T
            peaks, _ = find_peaks(
                -n_objects / sizes, prominence=100 / sizes.max(), distance=10
            )
            threshold = thresholds[peaks[0]]
        veins = 1.0 * np.array(probs[0] > threshold)

        # keep largest object (e.g., petiole) outside of the ROI
        if post_process and roi is not None:
            if self.verbose:
                print("Post processing...")

            # separate vein/petiole
            venation = roi * veins
            petiole = ~roi * veins

            # format petiole
            labeled_array, num_features = ndimage.label(petiole)
            if num_features >= 2:
                sizes = np.zeros(num_features + 1)
                unique_labels = np.unique(labeled_array)
                for i, u in enumerate(unique_labels):
                    size = (labeled_array == u).sum()
                    sizes[i] = size
                largest = unique_labels[np.argsort(sizes)[-2]]
                petiole = labeled_array == largest

            # add vein + petiole
            veins = venation + petiole

        # remove padding
        probs = probs[:, w:-w, w:-w]
        veins = veins[w:-w, w:-w] > 0.5

        return probs, veins
