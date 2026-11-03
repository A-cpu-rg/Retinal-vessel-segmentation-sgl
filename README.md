# Retinal Vessel Segmentation with Study Group Learning (SGL)

This project finds blood vessels in retina photos (DRIVE dataset) using the
method from:

> Yuqian Zhou, Hanchao Yu, Humphrey Shi. **"Study Group Learning: Improving
> Retinal Vessel Segmentation Trained with Noisy Labels."** MICCAI 2021.
> [arXiv:2103.03451](https://arxiv.org/abs/2103.03451)

**Status:** the code runs. The real DRIVE experiment has not been run yet, so
there are no results of my own in this README.

## Credits

- **Method:** the paper above.
- **Official code:** [SHI-Labs/SGL-Retinal-Vessel-Segmentation](https://github.com/SHI-Labs/SGL-Retinal-Vessel-Segmentation)
- **Starting code for this project:** [Tejmul/Retinal--Vessel-Segmentation](https://github.com/Tejmul/Retinal--Vessel-Segmentation)
- **Dataset:** DRIVE, Staal et al., IEEE TMI 2004 ([drive.grand-challenge.org](https://drive.grand-challenge.org/))

## How it works

**Model (CON):** two U-Nets in a row. EnhanceNet turns the photo into an
"enhanced" image, and SegNet turns that into a vessel probability for every
pixel. Only the final output is trained against labels. The model has about
15.5M parameters.

**Study Group Learning:**
1. Split the 20 training images randomly into K groups (the paper uses K = 8).
2. For each group, train a model on the other groups and let it predict that
   group. Every training image gets a *pseudo-label* from a model that never
   saw it.
3. Train the final model with `loss = BCE(pred, manual label) + BCE(pred, pseudo-label)`.

## Dataset

DRIVE is not included. Download it and place it like this (original DRIVE
file names):

```
data/DRIVE/
  training/images/21_training.tif   training/1st_manual/21_manual1.gif   training/mask/21_training_mask.gif
  test/images/01_test.tif           test/1st_manual/01_manual1.gif       test/mask/01_test_mask.gif
```

Evaluation needs the test labels (`test/1st_manual/`); some DRIVE copies
leave them out. To keep the data somewhere else, use `--root /path/to/DRIVE`.

## Setup (Mac)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The device is chosen automatically: CUDA, then Apple MPS, then CPU. You can
force one with `--device cuda|mps|cpu`.

## Usage

Run these from the project folder.

```bash
# Full pipeline: pseudo-labels -> final model -> one test evaluation
python src/run_sgl.py --k 8 --fold-epochs 15 --final-epochs 25

# Single steps
python src/sgl_labeling.py --k 8 --epochs 15                  # pseudo-labels only
python src/train.py --teacher-dir outputs/pseudo_labels --epochs 25 --out-dir checkpoints/final
python src/train.py --epochs 25 --out-dir checkpoints/baseline   # baseline, manual labels only

# Test-set evaluation of a saved checkpoint
python src/evaluate.py --ckpt checkpoints/final/last.pt

# Vessel maps for any image
python src/infer.py --ckpt checkpoints/final/last.pt \
    --images data/DRIVE/test/images/01_test.tif --masks data/DRIVE/test/mask/01_test_mask.gif
```

| Option | Meaning |
|---|---|
| `--patch-size` | training crop size (default 128; the paper uses 256) |
| `--batch-size`, `--lr`, `--repeat`, `--seed` | batch size, learning rate, patches per image per epoch, random seed |
| `--val-ids 37 38 39 40` | hold out these training images and keep the best epoch as `best.pt` |
| `--resume` | continue a stopped run from `last.pt` |
| `--workers` | data-loading processes (0 on a Mac, 2 on Colab) |

Run any script with `--help` to see all options.

**Outputs:**
- pseudo-labels: `outputs/pseudo_labels/`
- checkpoints: `checkpoints/folds/foldN/` and `checkpoints/final/`
- test metrics: `outputs/test_results.json`
- prediction images: `outputs/test_predictions/`

**Choosing the checkpoint:** the test set is used only once, at the end. By
default, the last epoch is used. With `--val-ids`, the best epoch on those
held-out training images is used.

**Metrics:** accuracy, sensitivity, specificity, AUC, Dice and IoU,
calculated **inside the field of view (FOV)**. The reference code's
whole-image numbers are shown on a separate line for comparison only. They
look higher because the black area outside the eye counts as correct.

## Google Colab

Choose **Runtime > Change runtime type > GPU**, upload DRIVE to Google Drive,
then run:

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/A-cpu-rg/Retinal-vessel-segmentation-sgl.git
%cd Retinal-vessel-segmentation-sgl
!pip install -r requirements.txt
!python src/run_sgl.py --device cuda --workers 2 --patch-size 256 --resume \
    --root /content/drive/MyDrive/DRIVE \
    --ckpt-dir /content/drive/MyDrive/sgl_run/checkpoints \
    --teacher-dir /content/drive/MyDrive/sgl_run/pseudo_labels \
    --results /content/drive/MyDrive/sgl_run/test_results.json \
    --save-dir /content/drive/MyDrive/sgl_run/test_predictions
```

Everything is saved to Google Drive. If the session disconnects, run the same
command again and it continues where it stopped.

## Results

Pending. I have not trained or evaluated on DRIVE yet.

For reference, the **paper** reports on DRIVE (K = 8): Se 0.8380, Sp 0.9834,
Acc 0.9705, AUC 0.9886, F1 0.8316. These are the authors' numbers, not mine.

The files already in `checkpoints/` and `outputs/` came with the copied
starting code. They are not my results, and Git ignores them.

## Changes from the starting code

- The checkpoint is no longer picked using the test set.
- Metrics are calculated inside the FOV.
- Works with the original DRIVE file names and gives clear errors for missing files.
- Study groups are split randomly, with K = 8 as in the paper.
- Pseudo-labels are 0 outside the FOV.
- Elastic warping now moves the image and its labels by exactly the same amount.
- Random seeds now cover all random numbers.
- Checkpoints can be resumed.
- Added `infer.py` and prediction images.

## Limitations

- The paper's noisy-label experiments (erasing thin vessels from labels) are
  not included. The paper does not describe them in enough detail to copy them
  exactly.
- The paper does not give the learning rate, epochs, batch size or threshold.
  The values here come from the reference code.

## Files

```
src/config.py        paths and constants
src/model.py         CON model (EnhanceNet + SegNet)
src/dataset.py       DRIVE loading and augmentation
src/losses.py        SGL loss
src/metrics.py       evaluation metrics
src/utils.py         device, seeds, checkpoints, self-ensemble
src/train.py         train one model
src/sgl_labeling.py  study-group pseudo-labels
src/evaluate.py      test-set evaluation
src/infer.py         predict on your own images
src/visualize.py     save prediction images
src/run_sgl.py       full pipeline
```
