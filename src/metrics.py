"""Segmentation metrics, computed in two separate ways:

"fov"        - only pixels inside the field of view (FOV). This is the main result.
"full_image" - all pixels, with outside-FOV pixels set to background. This is what
               the reference code reported. The extra pixels are easy true
               negatives, so accuracy, specificity and AUC come out higher.

Scores are computed per image and then averaged.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from config import THRESHOLD

METRIC_NAMES = ("acc", "se", "sp", "auc", "dice", "iou_vessel", "iou_bg")
PROTOCOLS = ("fov", "full_image")


def _div(a, b):
    return float(a) / float(b) if b > 0 else float("nan")


def binary_metrics(prob, gt, threshold=THRESHOLD):
    """Metrics for 1-D arrays of probabilities and 0/1 labels."""
    gt = np.asarray(gt) > 0.5
    prob = np.asarray(prob, dtype=np.float64)
    pred = prob > threshold
    tp = np.sum(pred & gt)
    tn = np.sum(~pred & ~gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    # AUC needs both classes in the label
    auc = float(roc_auc_score(gt, prob)) if 0 < gt.sum() < gt.size else float("nan")
    return {
        "acc": _div(tp + tn, tp + tn + fp + fn),
        "se": _div(tp, tp + fn),                  # sensitivity
        "sp": _div(tn, tn + fp),                  # specificity
        "auc": auc,
        "dice": _div(2 * tp, 2 * tp + fp + fn),   # same as F1
        "iou_vessel": _div(tp, tp + fp + fn),
        "iou_bg": _div(tn, tn + fp + fn),
    }


def evaluate_prediction(prob, vessel_gt, fov_mask, threshold=THRESHOLD):
    """Score one image. Inputs are (H, W) arrays cropped to the original size.
    Returns {"fov": {...}, "full_image": {...}}."""
    if not (prob.shape == vessel_gt.shape == fov_mask.shape):
        raise ValueError(f"Shape mismatch: prob {prob.shape}, label {vessel_gt.shape}, "
                         f"FOV mask {fov_mask.shape}")
    inside = fov_mask > 0.5
    return {
        "fov": binary_metrics(prob[inside], vessel_gt[inside], threshold),
        "full_image": binary_metrics((prob * inside).ravel(),
                                     (vessel_gt * inside).ravel(), threshold),
    }


def mean_metrics(per_image):
    return {p: {k: float(np.nanmean([m[p][k] for m in per_image])) for k in METRIC_NAMES}
            for p in PROTOCOLS}


def format_metrics(mean):
    header = f"{'':<14}" + "".join(f"{k:>11}" for k in METRIC_NAMES)
    rows = [header]
    for p, label in (("fov", "inside FOV"), ("full_image", "full image*")):
        rows.append(f"{label:<14}" + "".join(f"{mean[p][k]:>11.4f}" for k in METRIC_NAMES))
    rows.append("* reference-code method; use 'inside FOV' as the main result.")
    return "\n".join(rows)
