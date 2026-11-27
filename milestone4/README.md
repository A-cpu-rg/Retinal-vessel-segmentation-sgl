# Milestone 4 · SGL experiment and evaluation

[Project overview](../README.md)

## Generate pseudo-labels and train

SGL divides the eligible DRIVE training images into eight groups by default.
A separate CON model trains on the other groups and predicts the images in
its held-out group. The final SGL model minimizes binary cross-entropy against
the manual labels plus binary cross-entropy against the resulting soft
pseudo-labels. The baseline uses the manual-label loss alone.
[Fold training](../src/sgl_labeling.py) · [Losses](../src/losses.py)

## Run in Colab or locally

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A-cpu-rg/Retinal-vessel-segmentation-sgl/blob/main/notebooks/full_experiment_colab.ipynb)

Select a GPU runtime and run the notebook cells in order. The notebook
downloads DRIVE, validates all 40 samples, mounts Google Drive for resumable
checkpoints, and prints the comparison table. Outputs persist under
`MyDrive/retinal_sgl_full_run/`. [Notebook source](../notebooks/full_experiment_colab.ipynb)

For a local run, complete the [DRIVE setup](../milestone2/README.md#local-setup)
and execute from the repository root.

```bash
python src/run_sgl.py --k 8 --fold-epochs 15 --final-epochs 25 \
  --patch-size 256 --with-baseline
```

| Option | Purpose |
|---|---|
| `--root PATH` | Override the default DRIVE directory, `data/DRIVE` |
| `--device auto` | Select CUDA, then MPS, then CPU. Choose `cuda`, `mps`, or `cpu` explicitly when needed |
| `--patch-size N` | Set training patch size. The CLI default is 128 and the paper uses 256 |
| `--val-ids ID [ID ...]` | Hold out training images for checkpoint selection. The default uses the last fixed epoch |
| `--select-metric dice` | Select by validation Dice when validation IDs are supplied. Use `auc` for ROC-AUC |
| `--resume` | Continue compatible checkpoints and reuse compatible generated pseudo-labels |
| `--skip-labeling` | Use a complete existing set of pseudo-labels for the selected training IDs |
| `--no-self-ensemble` | Use a single prediction instead of averaging eight flip and transpose variants |
| `--no-elastic` | Disable affine and elastic training augmentation |
| `--workers N` | Set the number of data-loader processes |

## Review generated outputs

| Local path | Contents |
|---|---|
| `checkpoints/folds/foldN/` | Fold-teacher checkpoints |
| `checkpoints/final/`, `checkpoints/baseline/` | Final training checkpoints and saved histories |
| `outputs/pseudo_labels/` | Soft labels and the `folds.json` split and training record |
| `outputs/test_results.json`, `outputs/baseline_test_results.json` | Per-image and mean test metrics for each model |
| `outputs/comparison.json` | Baseline and SGL means plus signed SGL-minus-baseline deltas |
| `outputs/test_predictions/`, `outputs/baseline_test_predictions/` | Probability maps, binary masks, overlays, and error maps |

In Colab, the SGL test metrics are named `sgl_test_results.json` and these
outputs are written directly under the Google Drive run directory. The
evaluation records both field-of-view and whole-image metrics. Whole-image
scoring counts pixels outside the field of view as background, following
the reference implementation. Thresholded metrics use a vessel probability
greater than `100/255`. [Evaluation code](../src/evaluate.py) · [Metric definitions](../src/metrics.py)

## Run individual stages

```bash
python src/sgl_labeling.py --k 8 --epochs 15 --patch-size 256

python src/train.py --teacher-dir outputs/pseudo_labels \
  --epochs 25 --patch-size 256 --out-dir checkpoints/final

python src/evaluate.py --ckpt checkpoints/final/last.pt

python src/infer.py --ckpt checkpoints/final/last.pt \
  --images data/DRIVE/test/images/01_test.tif \
  --masks data/DRIVE/test/mask/01_test_mask.gif
```

Use the checkpoint selected during training if validation was enabled. The
inference command only needs the input image and checkpoint. The mask is optional.
