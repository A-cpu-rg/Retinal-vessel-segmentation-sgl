"""Predict vessels on your own retinal images (no labels needed).

Example:
    python src/infer.py --ckpt checkpoints/final/last.pt \
        --images data/DRIVE/test/images/01_test.tif \
        --masks  data/DRIVE/test/mask/01_test_mask.gif
"""
import argparse
from pathlib import Path

import torch

from config import CHECKPOINT_DIR, OUTPUT_DIR
from dataset import load_gray, load_rgb, pad_to_square, padded_side
from model import build_model
from utils import DEVICE_CHOICES, get_device, load_model_weights, predict
from visualize import save_prediction_images


def main():
    ap = argparse.ArgumentParser(description="Segment vessels in retinal images.")
    ap.add_argument("--ckpt", default=str(CHECKPOINT_DIR / "final" / "last.pt"))
    ap.add_argument("--images", nargs="+", required=True, help="RGB image files")
    ap.add_argument("--masks", nargs="*", default=[],
                    help="optional FOV masks, one per image, same order")
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR / "inference"))
    ap.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    ap.add_argument("--no-self-ensemble", action="store_true")
    args = ap.parse_args()

    if args.masks and len(args.masks) != len(args.images):
        ap.error(f"got {len(args.images)} images but {len(args.masks)} masks")
    for p in args.images + args.masks:
        if not Path(p).is_file():
            ap.error(f"file not found: {p}")

    device = get_device(args.device)
    model = build_model()
    load_model_weights(model, args.ckpt)
    model.to(device)

    for i, path in enumerate(args.images):
        rgb = load_rgb(path)
        h, w = rgb.shape[:2]
        x = torch.from_numpy(pad_to_square(rgb, padded_side(h, w)).transpose(2, 0, 1))[None]
        prob = predict(model, x.to(device), not args.no_self_ensemble)[0, 0, :h, :w].cpu().numpy()
        fov = None
        if args.masks:
            fov = (load_gray(args.masks[i]) > 125).astype("float32")
            if fov.shape != (h, w):
                raise ValueError(f"mask {args.masks[i]} is {fov.shape}, image is {(h, w)}")
        name = Path(path).stem
        save_prediction_images(args.out_dir, name, prob, fov=fov, rgb=rgb)
        print(f"[infer] {path} -> {args.out_dir}/{name}_*.png")


if __name__ == "__main__":
    main()
