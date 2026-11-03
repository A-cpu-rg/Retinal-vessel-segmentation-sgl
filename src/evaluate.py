"""Evaluate a checkpoint on the DRIVE test set.

Run this once, on the checkpoint you already chose. The test set is never used
to choose a checkpoint. train.py uses evaluate_dataset() for validation.
"""
import argparse
import json
import os

import torch

from config import CHECKPOINT_DIR, DATA_ROOT, OUTPUT_DIR, THRESHOLD
from dataset import DriveDataset
from metrics import evaluate_prediction, format_metrics, mean_metrics
from model import build_model
from utils import DEVICE_CHOICES, get_device, load_model_weights, predict
from visualize import save_prediction_images


@torch.no_grad()
def evaluate_dataset(model, ds, device, self_ensemble=True, save_dir=None):
    """Score every image of a mode="full" DriveDataset. Returns (mean, per_image)."""
    model.to(device).eval()
    per_image = []
    for i in range(len(ds)):
        s = ds[i]
        h, w = s["h"], s["w"]
        prob = predict(model, s["hr"][None].to(device), self_ensemble)[0, 0, :h, :w].cpu().numpy()
        gt = s["ve"][0, :h, :w].numpy()
        fov = s["ma"][0, :h, :w].numpy()
        m = evaluate_prediction(prob, gt, fov)
        m["id"] = s["id"]
        per_image.append(m)
        if save_dir:
            rgb = s["hr"][:, :h, :w].permute(1, 2, 0).numpy()
            save_prediction_images(save_dir, s["id"], prob, fov=fov, rgb=rgb, gt=gt)
    return mean_metrics(per_image), per_image


def evaluate_model(model, root, device, split="test", indices=None,
                   self_ensemble=True, save_dir=None):
    ds = DriveDataset(root, split=split, indices=indices, mode="full")
    return evaluate_dataset(model, ds, device, self_ensemble, save_dir)


def run_test_evaluation(ckpt_path, root, device, self_ensemble=True,
                        save_dir=None, results_path=None):
    """Load a checkpoint, evaluate it on the test set, print and save the results."""
    model = build_model()
    ckpt = load_model_weights(model, ckpt_path)
    info = {"checkpoint": str(ckpt_path)}
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        info.update(checkpoint_epoch=ckpt["epoch"], config=ckpt.get("config"),
                    best=ckpt.get("best"))
    print(f"[eval] checkpoint {ckpt_path}"
          + (f" (epoch {info['checkpoint_epoch']})" if "checkpoint_epoch" in info else ""))

    mean, per_image = evaluate_model(model, root, device, split="test",
                                     self_ensemble=self_ensemble, save_dir=save_dir)
    print(f"\n=== DRIVE test set ({len(per_image)} images, "
          f"self-ensemble={'on' if self_ensemble else 'off'}) ===")
    print(format_metrics(mean))
    if save_dir:
        print(f"[eval] prediction images written to {save_dir}")

    if results_path:
        os.makedirs(os.path.dirname(str(results_path)) or ".", exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({**info, "self_ensemble": self_ensemble, "threshold": THRESHOLD,
                       "protocols": {
                           "fov": "pixels inside the field of view only (main result)",
                           "full_image": "whole image, outside-FOV pixels counted as "
                                         "background (reference-code method)"},
                       "mean": mean, "per_image": per_image}, f, indent=2)
        print(f"[eval] metrics written to {results_path}")
    return mean, per_image


def main():
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on the DRIVE test set.")
    ap.add_argument("--root", default=str(DATA_ROOT), help="DRIVE folder")
    ap.add_argument("--ckpt", default=str(CHECKPOINT_DIR / "final" / "last.pt"))
    ap.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    ap.add_argument("--no-self-ensemble", action="store_true",
                    help="one prediction instead of the average of 8 flips")
    ap.add_argument("--save-dir", default=str(OUTPUT_DIR / "test_predictions"),
                    help="where to save prediction images ('' to skip)")
    ap.add_argument("--results", default=str(OUTPUT_DIR / "test_results.json"))
    args = ap.parse_args()

    run_test_evaluation(args.ckpt, args.root, get_device(args.device),
                        self_ensemble=not args.no_self_ensemble,
                        save_dir=args.save_dir or None, results_path=args.results)


if __name__ == "__main__":
    main()
