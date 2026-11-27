# Milestone 2 · Baseline reproduction

[Project overview](../README.md)

The baseline trains CON with binary cross-entropy against the first manual
vessel annotation. Both experiment arms use the same model architecture and
DRIVE partitions. The complete comparison applies the same final training and
evaluation configuration to both arms.

## Local setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download DRIVE from the [official challenge](https://drive.grand-challenge.org/Download/), which
requires registration. Evaluation requires the test labels, which some copies omit. The Colab
notebook uses a [complete third-party mirror](https://github.com/wfdubowen/Retina-Unet/tree/master/DRIVE)
containing both manual test annotations; cite DRIVE itself rather than the mirror. Keep the original
names:

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

Training uses random patches with flips, transposition, and optional
affine and elastic augmentation. Evaluation pads full images to a square
with a side divisible by 32 and restores the original image dimensions
before scoring. [DRIVE loader](../src/dataset.py)

## Train and evaluate the baseline

Run from the repository root. This fixed-epoch example uses the final
checkpoint and saves baseline metrics and predictions at the same paths as
the full comparison command.

```bash
python src/train.py --epochs 25 --patch-size 256 --out-dir checkpoints/baseline
python src/evaluate.py --ckpt checkpoints/baseline/last.pt \
  --save-dir outputs/baseline_test_predictions \
  --results outputs/baseline_test_results.json
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

The smoke test uses three generated 32 × 32 training images, two generated
test images, two study groups, one epoch, CPU execution, and a single
prediction per image. The previous audit also checked a rectangular model
input, fold partitioning, eight-way transform inversion, and all 40 DRIVE
samples. CON has 15,533,320 trainable parameters. The smoke-test scores
measure execution on generated images. Full DRIVE scores remain pending.
