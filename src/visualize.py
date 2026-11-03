"""Save prediction images.

NAME_prob.png    probability map
NAME_pred.png    binary vessel mask
NAME_overlay.png vessels in green on the photo (if rgb is given)
NAME_errors.png  white = correct, red = false vessel, blue = missed (if gt is given)
"""
import os

import numpy as np
from PIL import Image

from config import THRESHOLD


def save_prediction_images(out_dir, name, prob, fov=None, rgb=None, gt=None,
                           threshold=THRESHOLD):
    os.makedirs(out_dir, exist_ok=True)
    if fov is not None:
        prob = prob * (fov > 0.5)
    pred = prob > threshold

    def save(arr, suffix):
        Image.fromarray(arr).save(os.path.join(out_dir, f"{name}_{suffix}.png"))

    save((np.clip(prob, 0, 1) * 255).astype(np.uint8), "prob")
    save(pred.astype(np.uint8) * 255, "pred")

    if rgb is not None:
        overlay = np.clip(rgb, 0, 255).astype(np.float32)
        overlay[pred] = 0.4 * overlay[pred] + 0.6 * np.array([0, 255, 0])
        save(overlay.astype(np.uint8), "overlay")

    if gt is not None:
        gt = gt > 0.5
        err = np.zeros(pred.shape + (3,), np.uint8)
        err[pred & gt] = (255, 255, 255)
        err[pred & ~gt] = (255, 0, 0)
        err[~pred & gt] = (0, 128, 255)
        save(err, "errors")
