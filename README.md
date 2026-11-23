# Retinal Vessel Segmentation with Study Group Learning

A PyTorch reproduction of Study Group Learning (SGL) for retinal vessel segmentation on
DRIVE, using the CON architecture: an EnhanceNet U-Net followed by a SegNet U-Net.

**Authors:** Prerak Arya (230039) and Abhishek Meena (230137)

**Project status:** the baseline and SGL pipelines are implemented and pass unit and synthetic
end-to-end tests. A real DRIVE training run has **not** been completed because the dataset is not
included in this repository or available in the audited environment. No project performance
numbers are claimed yet.

![SGL and inference architecture](assets/architecture.svg)

## Method

The 20 DRIVE training images are split into `K` groups. For each group, a fresh model is trained
on the other groups and predicts the held-out images. These out-of-fold probabilities become
pseudo-labels. The final objectives are:

```text
baseline/fold model: BCE(prediction, manual label)
final SGL model:     BCE(prediction, manual label) + BCE(prediction, pseudo-label)
```

When validation IDs are supplied, they are excluded from both final training and all fold-teacher
training. This prevents validation information from leaking into pseudo-labels. The fold manifest
also records the split, training configuration, and self-ensemble setting so incompatible labels
cannot be silently reused.

## Results

The table deliberately contains no invented values. Run the full comparison command below to
populate `outputs/test_results.json`, `outputs/baseline_test_results.json`, and
`outputs/comparison.json`.

| Metric (inside DRIVE FOV) | Baseline | SGL | SGL - baseline | Paper SGL |
|---|---:|---:|---:|---:|
| Accuracy | pending | pending | pending | 0.9705 |
| Sensitivity | pending | pending | pending | 0.8380 |
| Specificity | pending | pending | pending | 0.9834 |
| AUC | pending | pending | pending | 0.9886 |
| Dice / F1 | pending | pending | pending | 0.8316 |
| Vessel IoU | pending | pending | pending | not reported |

The last column is from Zhou et al. (2021), not from this implementation. This project reports
mean per-image metrics inside the field of view as its primary protocol; whole-image metrics are
saved separately for comparison with the reference code.

### Verified implementation results

The following results were produced during the repository audit on 23 November 2026. They verify
that the implementation runs correctly, but they are not DRIVE performance measurements.

| Check | Result |
|---|---:|
| Core unit tests | **8/8 passed** |
| Python source compilation | **passed** |
| Model forward pass (`1 x 3 x 32 x 48`) | **passed** |
| DRIVE-format image, label, mask, and teacher loading | **passed** |
| Fold partition and eight-way transform inversion checks | **passed** |
| Synthetic fold training and pseudo-label generation | **passed** |
| Synthetic final SGL and manual-label baseline training | **passed** |
| Synthetic evaluation and comparison JSON generation | **passed** |

The synthetic integration run used three generated 32 x 32 training images, two generated test
images, two study groups, one epoch, CPU execution, and no self-ensemble. Its segmentation scores
are intentionally omitted because random generated images do not measure retinal-vessel quality.
The CON implementation contains **15,533,320 trainable parameters**.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download DRIVE from <https://drive.grand-challenge.org/>. Evaluation requires the test labels,
which some copies omit. Keep the original names:

```text
data/DRIVE/
  training/
    images/21_training.tif ... 40_training.tif
    1st_manual/21_manual1.gif ... 40_manual1.gif
    mask/21_training_mask.gif ... 40_training_mask.gif
  test/
    images/01_test.tif ... 20_test.tif
    1st_manual/01_manual1.gif ... 20_manual1.gif
    mask/01_test_mask.gif ... 20_test_mask.gif
```

## Run the reproducible comparison

This command generates out-of-fold labels, trains SGL and baseline models with identical final
training settings, evaluates each selected checkpoint once, and writes signed metric deltas:

```bash
python src/run_sgl.py \
  --k 8 --fold-epochs 15 --final-epochs 25 \
  --patch-size 256 --val-ids 37 38 39 40 \
  --with-baseline
```

The defaults use 128-pixel patches for lower memory use. The paper uses 256-pixel patches. If no
validation IDs are provided, the last fixed epoch is evaluated; with validation IDs, `best.pt` is
selected by validation Dice (or `--select-metric auc`). The test set is never used for checkpoint
selection.

Important options:

| Option | Purpose |
|---|---|
| `--root PATH` | DRIVE folder; default `data/DRIVE` |
| `--device auto,cuda,mps,cpu` | device selection; auto prefers CUDA, then MPS |
| `--resume` | continue compatible checkpoints and reuse compatible pseudo-labels |
| `--skip-labeling` | reuse a complete, compatible pseudo-label directory |
| `--no-self-ensemble` | disable eight-way flip/transpose averaging |
| `--no-elastic` | disable affine/elastic training augmentation |
| `--workers N` | data-loader processes |

Main outputs:

```text
checkpoints/folds/foldN/{last,best}.pt
checkpoints/final/{last,best}.pt
checkpoints/baseline/{last,best}.pt
outputs/pseudo_labels/*_teacher.png
outputs/test_results.json
outputs/baseline_test_results.json
outputs/comparison.json
outputs/test_predictions/
outputs/baseline_test_predictions/
```

Individual stages remain available:

```bash
# Pseudo-labels only
python src/sgl_labeling.py --k 8 --epochs 15

# Baseline only (single manual-label BCE)
python src/train.py --epochs 25 --out-dir checkpoints/baseline

# Final SGL only
python src/train.py --teacher-dir outputs/pseudo_labels \
  --epochs 25 --out-dir checkpoints/final

# Evaluate an already selected checkpoint
python src/evaluate.py --ckpt checkpoints/final/last.pt

# Segment an image without a ground-truth label
python src/infer.py --ckpt checkpoints/final/last.pt \
  --images data/DRIVE/test/images/01_test.tif \
  --masks data/DRIVE/test/mask/01_test_mask.gif
```

## Verification

Run the fast unit suite:

```bash
python -m unittest discover -s tests -v
```

Run the synthetic integration smoke test:

```bash
python tests/smoke_pipeline.py
```

The smoke test creates a temporary DRIVE-shaped dataset and executes fold training, pseudo-label
generation, final SGL training, baseline training, evaluation, and comparison. Its one-epoch scores
are sanity checks only and must not be presented as research results.

The repository was audited on Python 3.12 with PyTorch 2.9.1 on CPU. Unit tests and the synthetic
pipeline passed. Full DRIVE training, GPU performance, and elastic augmentation were not exercised
in that environment because DRIVE and OpenCV were unavailable; OpenCV is included in
`requirements.txt` for a normal installation.

## Code map

| File | Responsibility |
|---|---|
| `src/model.py` | CON, EnhanceNet, and SegNet |
| `src/dataset.py` | DRIVE loading, crops, padding, and augmentation |
| `src/losses.py` | baseline BCE and dual-target SGL loss |
| `src/sgl_labeling.py` | study groups and out-of-fold pseudo-labels |
| `src/train.py` | deterministic/resumable training and validation selection |
| `src/run_sgl.py` | complete SGL experiment and optional baseline comparison |
| `src/evaluate.py`, `src/metrics.py` | FOV and whole-image evaluation |
| `src/comparison.py` | baseline/SGL metric deltas and JSON output |
| `src/infer.py`, `src/visualize.py` | unlabeled inference and saved visualizations |
| `tests/` | unit checks and synthetic pipeline smoke test |

## Known limitations

- Real DRIVE metrics remain pending until the dataset and sufficient training hardware are
  provided. CPU execution is supported but a full `K=8` experiment is expensive.
- The paper's intentionally corrupted/noisy-label ablations are not implemented; its exact label
  corruption procedure is not fully specified here.
- Learning rate, epoch count, batch size, and the `100/255` threshold follow the reference-code
  conventions rather than a complete hyperparameter specification in the paper.

## References

- Y. Zhou, H. Yu, H. Shi. *Study Group Learning: Improving Retinal Vessel Segmentation Trained
  with Noisy Labels.* MICCAI 2021. [arXiv:2103.03451](https://arxiv.org/abs/2103.03451)
- [Official SGL implementation](https://github.com/SHI-Labs/SGL-Retinal-Vessel-Segmentation)
- J. Staal et al. *Ridge-based vessel segmentation in color images of the retina.* IEEE TMI,
  2004. [DRIVE](https://drive.grand-challenge.org/)
