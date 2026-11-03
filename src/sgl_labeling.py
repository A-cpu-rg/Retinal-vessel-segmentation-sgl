"""Study Group Learning step 1: make pseudo-labels with K study-group models.

The training images are split randomly into K groups (as in the paper). For each
group, a new model is trained on the other K-1 groups and then predicts the
images of that group, which it has never seen. These predictions become the
pseudo-labels used to train the final model.

Pseudo-labels use 8-way self-ensemble (as in the reference code) and are set to
0 outside the field of view. The fold split is saved to folds.json.
"""
import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import CHECKPOINT_DIR, TEACHER_DIR
from dataset import DriveDataset, list_ids, teacher_filename
from model import build_model
from train import add_common_args, config_from_args, train_model
from utils import free_device_memory, get_device, load_model_weights, predict


def make_folds(ids, k, seed=0):
    """Split ids randomly into k groups whose sizes differ by at most one."""
    if not 2 <= k <= len(ids):
        raise ValueError(f"k must be between 2 and the number of images ({len(ids)}), got {k}")
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    return [sorted(shuffled[i::k]) for i in range(k)]


def _check_or_write_folds(path, folds_info):
    # Stop if this folder already has pseudo-labels from a different split.
    if path.exists():
        if json.loads(path.read_text()) != folds_info:
            raise ValueError(
                f"{path} was made with different folds (k/seed/images).\n"
                "Use a different --teacher-dir (and checkpoint folder) for a new split.")
    else:
        path.write_text(json.dumps(folds_info, indent=2))


@torch.no_grad()
def write_pseudo_labels(model, root, ids, out_dir, device, self_ensemble=True):
    ds = DriveDataset(root, split="training", indices=ids, mode="full")
    for i in range(len(ds)):
        s = ds[i]
        h, w = s["h"], s["w"]
        prob = predict(model, s["hr"][None].to(device), self_ensemble)[0, 0, :h, :w].cpu().numpy()
        prob = prob * s["ma"][0, :h, :w].numpy()  # no vessels outside the FOV
        img = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(Path(out_dir) / teacher_filename(s["id"]))
        print(f"  [sgl] pseudo-label saved: {s['id']}")


def generate_pseudo_labels(root, out_dir, cfg, k=8, ckpt_dir=CHECKPOINT_DIR / "folds",
                           device=None, resume=False, self_ensemble=True):
    device = device or get_device("auto")
    out_dir, ckpt_dir = Path(out_dir), Path(ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = list_ids(root, "training")
    folds = make_folds(ids, k, cfg.seed)
    _check_or_write_folds(out_dir / "folds.json", {"k": k, "seed": cfg.seed, "folds": folds})
    print(f"[sgl] {len(ids)} training images, {k} study groups: "
          + " | ".join(" ".join(f) for f in folds))

    for fi, holdout in enumerate(folds, start=1):
        train_ids = [i for i in ids if i not in holdout]
        assert not set(train_ids) & set(holdout)  # held-out images must not be trained on

        if resume and all((out_dir / teacher_filename(i)).exists() for i in holdout):
            print(f"\n[sgl] fold {fi}/{k}: pseudo-labels already exist, skipping")
            continue
        print(f"\n[sgl] fold {fi}/{k}  held out={holdout}  train on {len(train_ids)} images")
        fold_cfg = replace(cfg, seed=cfg.seed + fi, val_ids=[])
        ckpt = train_model(root, fold_cfg, ckpt_dir / f"fold{fi}", train_ids=train_ids,
                           teacher_dir=None, device=device, resume=resume)
        model = build_model()
        load_model_weights(model, ckpt)
        model.to(device)
        write_pseudo_labels(model, root, holdout, out_dir, device, self_ensemble)
        del model
        free_device_memory(device)
    print(f"\n[sgl] all pseudo-labels are in {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Study Group Learning: make pseudo-labels.")
    add_common_args(ap)
    ap.add_argument("--k", type=int, default=8, help="number of study groups (paper: 8)")
    ap.add_argument("--epochs", type=int, default=15, help="epochs per fold model")
    ap.add_argument("--out-dir", default=str(TEACHER_DIR), help="where to save pseudo-labels")
    ap.add_argument("--ckpt-dir", default=str(CHECKPOINT_DIR / "folds"))
    ap.add_argument("--no-self-ensemble", action="store_true")
    args = ap.parse_args()
    generate_pseudo_labels(args.root, args.out_dir, config_from_args(args, args.epochs),
                           k=args.k, ckpt_dir=args.ckpt_dir, device=get_device(args.device),
                           resume=args.resume, self_ensemble=not args.no_self_ensemble)


if __name__ == "__main__":
    main()
