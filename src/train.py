"""Train one CON model: a study-group fold model (manual labels only) or the
final SGL model (manual labels + pseudo-labels).

The test set is never used here. The checkpoint to use is either the last
epoch (last.pt) or, with --val-ids, the best epoch on those held-out training
images (best.pt). last.pt is saved every epoch so a run can be resumed.
"""
import argparse
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, DATA_ROOT
from dataset import DriveDataset, list_ids
from evaluate import evaluate_dataset
from losses import SGLoss
from model import build_model
from utils import (DEVICE_CHOICES, get_device, load_checkpoint, save_checkpoint,
                   seed_worker, set_seed)

SELECT_METRICS = ("dice", "auc")


@dataclass
class TrainConfig:
    """All training settings. Saved inside every checkpoint."""
    epochs: int = 20
    batch_size: int = 4
    patch_size: int = 128       # the paper uses 256
    lr: float = 1e-4
    repeat: int = 40            # random patches per image per epoch
    elastic: bool = True
    seed: int = 0
    workers: int = 0            # DataLoader processes
    val_ids: list = field(default_factory=list)  # training ids held out for validation
    eval_every: int = 5
    select_metric: str = "dice"  # "dice" or "auc", inside the FOV
    val_self_ensemble: bool = False


# Settings that may change when resuming (they don't affect what was learned).
_MAY_CHANGE_ON_RESUME = {"epochs", "workers", "eval_every"}


def _checkpoint(model, optimizer, epoch, cfg, run_info, history, best):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "config": asdict(cfg),
        **run_info,
        "history": history,
        "best": best,
        "torch_version": str(torch.__version__),  # str, so weights_only loading accepts it
    }


def _check_resume_compatible(ckpt, cfg, run_info, path):
    saved, now = ckpt.get("config", {}), asdict(cfg)
    diffs = [f"{k}: checkpoint={saved.get(k)!r}, now={now[k]!r}"
             for k in now if k not in _MAY_CHANGE_ON_RESUME and saved.get(k) != now[k]]
    diffs += [f"{k}: checkpoint={ckpt.get(k)!r}, now={v!r}"
              for k, v in run_info.items() if ckpt.get(k) != v]
    if diffs:
        raise ValueError(f"Cannot resume from {path}: these settings differ from the "
                         "saved run:\n  " + "\n  ".join(diffs)
                         + "\nUse the same settings, or a different output folder.")


def train_model(root, cfg, out_dir, train_ids=None, teacher_dir=None, device=None,
                resume=False, log_every=20):
    """Train one model. Returns the checkpoint to use: best.pt with validation
    images, otherwise last.pt."""
    if cfg.select_metric not in SELECT_METRICS:
        raise ValueError(f"select_metric must be one of {SELECT_METRICS}")
    if cfg.eval_every < 1:
        raise ValueError("eval_every must be at least 1")
    device = device or get_device("auto")
    out_dir = Path(out_dir)
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"

    all_ids = list_ids(root, "training")
    val_ids = sorted(cfg.val_ids)
    unknown = sorted(set(val_ids) - set(all_ids))
    if unknown:
        raise ValueError(f"--val-ids {unknown} are not training images. Training ids: {all_ids}")
    train_ids = sorted(train_ids) if train_ids is not None else all_ids
    train_ids = [i for i in train_ids if i not in set(val_ids)]
    if not train_ids:
        raise ValueError("No training images left after removing the validation images.")
    run_info = {"train_ids": train_ids,
                "teacher_dir": str(teacher_dir) if teacher_dir else None}

    set_seed(cfg.seed)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    start_epoch, history, best = 1, [], {"metric": cfg.select_metric, "score": None, "epoch": None}

    if last_path.exists():
        if not resume:
            raise FileExistsError(
                f"A checkpoint already exists at {last_path}.\n"
                "Use --resume to continue that run, or choose a different output folder.")
        ckpt = load_checkpoint(last_path)
        _check_resume_compatible(ckpt, cfg, run_info, last_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, history, best = ckpt["epoch"] + 1, ckpt["history"], ckpt["best"]
        print(f"[train] resuming {last_path} after epoch {ckpt['epoch']}")
    elif resume:
        print(f"[train] --resume: no checkpoint in {out_dir} yet, starting from scratch")

    print(f"[train] device={device} epochs={cfg.epochs} batch={cfg.batch_size} "
          f"patch={cfg.patch_size} lr={cfg.lr} teacher={'yes' if teacher_dir else 'no'}\n"
          f"        train images ({len(train_ids)}): {' '.join(train_ids)}\n"
          f"        validation images: {' '.join(val_ids) if val_ids else 'none (fixed epochs)'}")
    if start_epoch > cfg.epochs:
        print(f"[train] already trained for {cfg.epochs} epochs, nothing to do")

    ds =DriveDataset(root, split="training", indices=train_ids, teacher_dir=teacher_dir,
                      mode="patch", patch_size=cfg.patch_size, repeat=cfg.repeat,
                      elastic=cfg.elastic)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.workers, worker_init_fn=seed_worker,
                        pin_memory=device.type == "cuda")
    if len(loader) == 0:
        raise ValueError(f"batch_size={cfg.batch_size} is larger than the {len(ds)} "
                         "patches per epoch; lower --batch-size or raise --repeat.")
    val_ds = (DriveDataset(root, split="training", indices=val_ids, mode="full")
              if val_ids else None)
    criterion = SGLoss()

    for epoch in range(start_epoch, cfg.epochs + 1):
        # Re-seed every epoch so a resumed run gets the same random patches.
        set_seed(cfg.seed + epoch)
        model.train()
        running, n, t0 = 0.0, 0, time.time()
        for it, batch in enumerate(loader):
            hr = batch["hr"].to(device)
            ve = batch["ve"].to(device)
            te = batch["te"].to(device)
            optimizer.zero_grad()
            _, seg = model(hr)
            loss = criterion(seg, ve, te)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n += 1
            if (it + 1) % log_every == 0:
                print(f"  epoch {epoch:3d} it {it+1:4d}/{len(loader)} "
                      f"loss {running/n:.4f}", flush=True)
        entry = {"epoch": epoch, "loss": running / max(n, 1), "seconds": time.time() - t0}
        print(f"[epoch {epoch:3d}/{cfg.epochs}] mean_loss={entry['loss']:.4f} "
              f"({entry['seconds']:.1f}s)", flush=True)

        if val_ds is not None and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs):
            mean, _ = evaluate_dataset(model, val_ds, device, cfg.val_self_ensemble)
            entry["val"] = mean
            score = mean["fov"][cfg.select_metric]
            v = mean["fov"]
            print(f"  [val @ep{epoch}] inside FOV: dice={v['dice']:.4f} auc={v['auc']:.4f} "
                  f"acc={v['acc']:.4f} se={v['se']:.4f} sp={v['sp']:.4f}", flush=True)
            if not math.isnan(score) and (best["score"] is None or score > best["score"]):
                best = {"metric": cfg.select_metric, "score": score, "epoch": epoch}
                save_checkpoint(best_path, _checkpoint(model, None, epoch, cfg, run_info,
                                                       history + [entry], best))
                print(f"  [val] new best {cfg.select_metric}={score:.4f} -> {best_path}",
                      flush=True)

        history.append(entry)
        save_checkpoint(last_path, _checkpoint(model, optimizer, epoch, cfg, run_info,
                                               history, best))

    if val_ids and best_path.exists():
        print(f"[train] best epoch {best['epoch']} ({best['metric']}={best['score']:.4f}) "
              f"-> {best_path}")
        return best_path
    print(f"[train] final checkpoint -> {last_path}")
    return last_path


def add_common_args(ap):
    """Arguments shared by train.py, sgl_labeling.py and run_sgl.py."""
    d = TrainConfig()
    ap.add_argument("--root", default=str(DATA_ROOT), help="DRIVE folder (has training/ and test/)")
    ap.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--patch-size", type=int, default=d.patch_size)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--repeat", type=int, default=d.repeat,
                    help="random patches per image per epoch")
    ap.add_argument("--no-elastic", action="store_true", help="turn off elastic deformation")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--workers", type=int, default=d.workers,
                    help="DataLoader worker processes (try 2 on Colab)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from existing last.pt checkpoints")


def add_validation_args(ap):
    d = TrainConfig()
    ap.add_argument("--val-ids", nargs="*", default=[],
                    help="training ids held out to pick the best epoch, e.g. 37 38 39 40")
    ap.add_argument("--eval-every", type=int, default=d.eval_every)
    ap.add_argument("--select-metric", choices=SELECT_METRICS, default=d.select_metric)


def config_from_args(args, epochs):
    return TrainConfig(
        epochs=epochs, batch_size=args.batch_size, patch_size=args.patch_size,
        lr=args.lr, repeat=args.repeat, elastic=not args.no_elastic, seed=args.seed,
        workers=args.workers, val_ids=sorted(getattr(args, "val_ids", []) or []),
        eval_every=getattr(args, "eval_every", TrainConfig.eval_every),
        select_metric=getattr(args, "select_metric", TrainConfig.select_metric))


def main():
    ap = argparse.ArgumentParser(description="Train one CON model.")
    add_common_args(ap)
    add_validation_args(ap)
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--teacher-dir", default=None,
                    help="pseudo-label folder (omit to train on manual labels only)")
    ap.add_argument("--out-dir", default=str(CHECKPOINT_DIR / "single"),
                    help="folder for last.pt / best.pt")
    args = ap.parse_args()

    train_model(args.root, config_from_args(args, args.epochs), args.out_dir,
                teacher_dir=args.teacher_dir, device=get_device(args.device),
                resume=args.resume)


if __name__ == "__main__":
    main()
