"""DRIVE loading, padding, training patches and augmentation.

Uses the original DRIVE file names (21_training.tif, 21_training_mask.gif,
01_test.tif, ...). The reference code's renamed files (21_mask.gif, ...) also work.

Each sample has: hr (RGB 0..255), ve (vessel label 0/1), ma (FOV mask 0/1),
te (pseudo-label 0..1, or a copy of ve when there is none), id, h, w.

mode="patch": random crops with flips/transpose and elastic warping (training).
mode="full":  whole image, zero-padded to a square (validation, test, pseudo-labels).
              Crop predictions back with [:h, :w].
"""
import glob
import math
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import Dataset

from config import PAD_MULTIPLE, SIZE_MULTIPLE

SPLITS = ("training", "test")


def teacher_filename(idx):
    """File name of the pseudo-label for training image `idx`."""
    return f"{idx}_training_x1_teacher.png"


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32)


def load_gray(path):
    # convert("L") gives plain 0..255 values, also for palette GIFs
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32)


def _find_file(folder, candidates, what):
    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    tried = ", ".join(os.path.join(folder, c) for c in candidates)
    raise FileNotFoundError(f"Could not find the {what}. Looked for: {tried}")


def list_ids(root, split):
    """Sorted image ids in a split, e.g. ["21", "22", ..., "40"]."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    img_dir = os.path.join(root, split, "images")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(
            f"DRIVE image folder not found: {img_dir}\n"
            f"Put DRIVE in {root} (with 'training/' and 'test/' inside) or pass "
            "--root. See the Dataset section of README.md.")
    ids = sorted({os.path.basename(p).split("_")[0]
                  for p in glob.glob(os.path.join(img_dir, "*.tif"))})
    if not ids:
        raise FileNotFoundError(f"No .tif images found in {img_dir}")
    return ids


def padded_side(h, w, multiple=PAD_MULTIPLE):
    """Side of the square an h x w image is padded to (608 for DRIVE)."""
    return int(math.ceil(max(h, w) / multiple) * multiple)


def pad_to_square(arr, side):
    """Zero-pad an (H, W) or (H, W, C) array on the bottom/right to side x side."""
    out = np.zeros((side, side) + arr.shape[2:], dtype=np.float32)
    out[:arr.shape[0], :arr.shape[1]] = arr
    return out


def elastic_transform(image, alpha, sigma, alpha_affine, rng=np.random, field_channels=5):
    """Random affine + elastic warp of an (H, W, C) array.

    All channels (image and labels) are moved by one shared displacement field,
    so they stay aligned. The field is built like the reference code built it
    (5 noise channels blurred together), which keeps the same warp strength.
    """
    shape = image.shape
    shape_size = shape[:2]
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([
        center_square + square_size,
        [center_square[0] + square_size, center_square[1] - square_size],
        center_square - square_size,
    ])
    pts2 = pts1 + rng.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    noise_shape = shape_size + (field_channels,)
    dx = gaussian_filter(rng.rand(*noise_shape) * 2 - 1, sigma).mean(axis=2) * alpha
    dy = gaussian_filter(rng.rand(*noise_shape) * 2 - 1, sigma).mean(axis=2) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    coords = [(y + dy).ravel(), (x + dx).ravel()]
    channels = [map_coordinates(image[..., c], coords, order=1, mode="reflect").reshape(shape_size)
                for c in range(shape[2])]
    return np.stack(channels, axis=2)


class DriveDataset(Dataset):
    """DRIVE images with vessel labels, FOV masks and optional pseudo-labels.

    indices: image ids to use (None = all). teacher_dir: folder with pseudo-labels.
    repeat: random patches per image per epoch (patch mode only).
    """

    def __init__(self, root, split="training", indices=None, teacher_dir=None,
                 mode="patch", patch_size=128, repeat=40, augment=True, elastic=True):
        if mode not in ("patch", "full"):
            raise ValueError(f"mode must be 'patch' or 'full', got {mode!r}")
        self.root = str(root)
        self.split = split
        self.mode = mode
        self.patch_size = patch_size
        self.repeat = repeat if mode == "patch" else 1
        self.augment = augment and mode == "patch"
        self.elastic = elastic and mode == "patch"
        self.teacher_dir = str(teacher_dir) if teacher_dir is not None else None

        available = list_ids(self.root, split)
        if indices is None:
            self.ids = available
        else:
            missing = sorted(set(indices) - set(available))
            if missing:
                raise ValueError(f"Image ids {missing} are not in {self.root}/{split}/images. "
                                 f"Available ids: {available}")
            self.ids = sorted(indices)
        if not self.ids:
            raise ValueError("No images selected.")

        self.samples = [self._load_one(i) for i in self.ids]

        if mode == "patch":
            smallest = min(min(s["hr"].shape[:2]) for s in self.samples)
            if not patch_size or patch_size > smallest or patch_size % SIZE_MULTIPLE:
                raise ValueError(f"patch_size must be a multiple of {SIZE_MULTIPLE} and at "
                                 f"most {smallest} (the smallest image side), got {patch_size}")

    def _load_one(self, idx):
        base = os.path.join(self.root, self.split)
        s = self.split
        hr = load_rgb(_find_file(os.path.join(base, "images"),
                                 [f"{idx}_{s}.tif", f"{idx}_training.tif"],
                                 f"image for id {idx}"))
        try:
            ve_path = _find_file(os.path.join(base, "1st_manual"),
                                 [f"{idx}_manual1.gif"], f"manual vessel label for id {idx}")
        except FileNotFoundError as e:
            hint = ("\nSome copies of DRIVE do not include the manual labels of the test "
                    "set; they are needed for evaluation.") if s == "test" else ""
            raise FileNotFoundError(f"{e}{hint}") from None
        ma_path = _find_file(os.path.join(base, "mask"),
                             [f"{idx}_{s}_mask.gif", f"{idx}_mask.gif"],
                             f"FOV mask for id {idx}")
        ve = (load_gray(ve_path) > 200).astype(np.float32)
        ma = (load_gray(ma_path) > 125).astype(np.float32)
        if ve.sum() == 0:
            raise ValueError(f"The vessel label {ve_path} has no vessel pixels. "
                             "Vessels should be white (255) on black.")

        if self.teacher_dir is not None:
            tp = os.path.join(self.teacher_dir, teacher_filename(idx))
            if not os.path.isfile(tp):
                raise FileNotFoundError(
                    f"Pseudo-label not found: {tp}\nRun the study-group step first "
                    "(python src/sgl_labeling.py) or check --teacher-dir.")
            te = load_gray(tp) / 255.0
        else:
            te = ve.copy()  # no pseudo-labels: the loss uses the manual label twice

        for name, arr in (("vessel label", ve), ("FOV mask", ma), ("pseudo-label", te)):
            if arr.shape != hr.shape[:2]:
                raise ValueError(f"Size mismatch for id {idx}: image is {hr.shape[:2]} "
                                 f"but its {name} is {arr.shape}")
        return {"id": idx, "hr": hr, "ve": ve, "ma": ma, "te": te}

    def __len__(self):
        return len(self.samples) * self.repeat

    def _augment(self, hr, ve, ma, te):
        if random.random() < 0.5:  # hflip
            hr, ve, ma, te = hr[:, ::-1], ve[:, ::-1], ma[:, ::-1], te[:, ::-1]
        if random.random() < 0.5:  # vflip
            hr, ve, ma, te = hr[::-1], ve[::-1], ma[::-1], te[::-1]
        if random.random() < 0.5:  # transpose (with the flips: all 8 rotations/flips)
            hr = hr.transpose(1, 0, 2)
            ve, ma, te = ve.T, ma.T, te.T
        return hr, ve, ma, te

    def _crop(self, hr, ve, ma, te):
        ih, iw = hr.shape[:2]
        ip = self.patch_size
        iy = random.randrange(0, ih - ip + 1)
        ix = random.randrange(0, iw - ip + 1)
        sl = (slice(iy, iy + ip), slice(ix, ix + ip))
        return hr[sl], ve[sl], ma[sl], te[sl]

    def _getitem_patch(self, s):
        hr, ve, ma, te = s["hr"], s["ve"], s["ma"], s["te"]
        if self.augment:
            hr, ve, ma, te = self._augment(hr, ve, ma, te)
        hr, ve, ma, te = self._crop(np.ascontiguousarray(hr),
                                    np.ascontiguousarray(ve),
                                    np.ascontiguousarray(ma),
                                    np.ascontiguousarray(te))
        if self.elastic and random.random() < 0.5:
            # warp image and all labels together so they stay aligned
            merged = np.concatenate([hr, ve[..., None], ma[..., None], te[..., None]], axis=2)
            merged = elastic_transform(merged, merged.shape[1] * 3,
                                       merged.shape[1] * 0.07, merged.shape[1] * 0.09)
            hr = merged[..., 0:3]
            ve = (merged[..., 3] > 0.5).astype(np.float32)
            ma = (merged[..., 4] > 0.5).astype(np.float32)
            te = np.clip(merged[..., 5], 0, 1)
        return hr, ve, ma, te

    def _getitem_full(self, s):
        h, w = s["hr"].shape[:2]
        side = padded_side(h, w)
        return tuple(pad_to_square(a, side) for a in (s["hr"], s["ve"], s["ma"], s["te"]))

    def __getitem__(self, i):
        s = self.samples[i % len(self.samples)]
        if self.mode == "patch":
            hr, ve, ma, te = self._getitem_patch(s)
            h = w = self.patch_size
        else:
            hr, ve, ma, te = self._getitem_full(s)
            h, w = s["hr"].shape[:2]
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
        return {"hr": to_t(hr.transpose(2, 0, 1)), "ve": to_t(ve)[None],
                "ma": to_t(ma)[None], "te": to_t(te)[None],
                "id": s["id"], "h": h, "w": w}
