import os, sys, glob
import numpy as np
import torch
import yaml
from PIL import Image
from importlib import reload

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../")
import models.BuildCNN as BuildCNN
import models.VeinGrower as VeinGrower
from utils.GetLowestGPU import GetLowestGPU

device = torch.device(GetLowestGPU(verbose=0))

GT_IMAGES = [
    "C_1_14_18_bot", "C_1_21_27_bot", "C_1_29_24_bot", "C_1_4_19_bot",
    "C_1_7_2_bot", "C_1_8_12_bot", "C_1_8_1_bot", "C_2_26_51_bot"
]

OUR_RUNS = [
    "20260701-191150-693819",
    "20260702-181007-264761",
    "20260705-143619-164979"
]

image_path   = "../data/images/"
roi_path     = "../data_marion/leaf_preds_jlag_sam3/"
results_path = "../results/"

layers           = [3, 32, 32, 32, 32, 64, 128]
output_shape     = [2, 3, 3]
output_activation = torch.nn.Softmax2d()
n_locs     = 10000
batch_size = 2048
threshold  = None

for run_name in OUR_RUNS:
    run_dir      = results_path + run_name + "/"
    weights_file = run_dir + "vein_grower_best_val_model.save"
    config_file  = run_dir + "config.yaml"
    pred_dir     = run_dir + "vein_fl_preds/"

    if not os.path.exists(weights_file):
        print(f"Skipping {run_name} - no weights found")
        continue

    missing = [f for f in GT_IMAGES if not os.path.exists(pred_dir + f + ".png")]
    if not missing:
        print(f"Skipping {run_name} - all GT images already done")
        continue

    with open(config_file) as f:
        config = yaml.safe_load(f)
    window_size = config["data"]["window_size"]

    print(f"\nProcessing {run_name} (window_size={window_size}) - {len(missing)} missing GT images")

    reload(BuildCNN)
    model = BuildCNN.CNN(
        window_size=window_size, layers=layers,
        output_shape=output_shape, output_activation=output_activation
    ).to(device)
    weights = torch.load(weights_file, map_location=device)
    model.load_state_dict(weights)
    model.eval()

    reload(VeinGrower)
    grower = VeinGrower.VeinGrower(window_size=window_size, model=model, device=device, verbose=True)

    os.makedirs(pred_dir, exist_ok=True)

    for base_name in missing:
        image_file = image_path + base_name + ".jpeg"
        roi_file   = roi_path + base_name + ".png"

        if not os.path.exists(image_file):
            print(f"  Image not found: {image_file}")
            continue

        print(f"  Inferring {base_name}...")
        image = np.array(Image.open(image_file), dtype=np.float32) / 255
        roi   = None
        if os.path.exists(roi_file):
            roi_arr = np.array(Image.open(roi_file), dtype=np.float32)
            roi = (roi_arr[:, :, 0] if roi_arr.ndim == 3 else roi_arr) / 255 > 0.5

        prob, mask = grower.grow(
            image=image, roi=roi, start_locs=None,
            n_locs=n_locs, batch_size=batch_size,
            threshold=threshold, post_process=True
        )

        save_mask = np.stack([mask, mask, mask], axis=-1)
        Image.fromarray(np.uint8(255 * save_mask)).save(pred_dir + base_name + ".png")
        print(f"  Saved {base_name}.png")
