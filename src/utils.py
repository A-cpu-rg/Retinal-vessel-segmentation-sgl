"""Device selection, random seeds, checkpoints and 8-way self-ensemble."""
import itertools
import os
import random
from pathlib import Path

import numpy as np
import torch

DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def get_device(prefer="auto"):
    """With "auto", picks CUDA, then Apple MPS, then CPU. Asking for a device
    that is not available raises an error instead of silently using the CPU."""
    if prefer not in DEVICE_CHOICES:
        raise ValueError(f"Unknown device {prefer!r}; choose one of {DEVICE_CHOICES}")
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if prefer == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but no CUDA GPU is available. "
                           "On Colab: Runtime > Change runtime type > GPU.")
    if prefer == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps was requested but Apple MPS is not available.")
    return torch.device(prefer)


def free_device_memory(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def set_seed(seed):
    # Augmentation uses all three random generators, so seed all three.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id):
    """DataLoader worker_init_fn: give NumPy and `random` the worker's torch seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(path, contents):
    # Write to a temp file, then rename, so a crash never leaves a broken checkpoint.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(contents, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model_weights(model, path):
    """Load weights into `model` and return the checkpoint.
    Works with full checkpoints ({"model": ...}) and plain state_dicts."""
    ckpt = load_checkpoint(path)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return ckpt


def _transform(x, hflip, vflip, transpose):
    if hflip:
        x = torch.flip(x, dims=[-1])
    if vflip:
        x = torch.flip(x, dims=[-2])
    if transpose:
        x = x.transpose(-1, -2)
    return x


def _inverse(x, hflip, vflip, transpose):
    # undo the steps in reverse order
    if transpose:
        x = x.transpose(-1, -2)
    if vflip:
        x = torch.flip(x, dims=[-2])
    if hflip:
        x = torch.flip(x, dims=[-1])
    return x


@torch.no_grad()
def predict(model, hr, self_ensemble=True):
    """Vessel probability (B, 1, H, W) for a padded (B, 3, H, W) image tensor.

    With self_ensemble, the 8 flip/transpose versions are predicted and averaged.
    Puts the model in eval mode.
    """
    model.eval()
    combos = list(itertools.product([False, True], repeat=3)) if self_ensemble \
        else [(False, False, False)]
    acc = None
    for hflip, vflip, transpose in combos:
        xt = _transform(hr, hflip, vflip, transpose)
        _, seg = model(xt)
        seg = _inverse(seg, hflip, vflip, transpose)
        acc = seg if acc is None else acc + seg
    return acc / len(combos)
