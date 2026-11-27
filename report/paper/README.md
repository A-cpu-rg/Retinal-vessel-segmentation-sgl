# Retinal vessel segmentation report

[Read the IEEE-format PDF](retinal_sgl_ieee.pdf) by Prerak Arya (230039) and
Abhsihek Meena (230137).

The paper reports a matched two-group CPU pilot on DRIVE. Both models were
evaluated on the 20 official test images. The SGL arm increased mean ROC-AUC
from 0.9103 to 0.9178. At the fixed threshold, both arms classified nearly
all in-field pixels as vessel. The planned eight-group GPU benchmark is
documented separately. [Full experiment](../../milestone4/README.md)

## Review the evidence

| File | Contents |
|---|---|
| [Pilot summary](evidence/summary.json) | Split statistics, mean metrics, paired bootstrap intervals, and pooled error counts |
| [Per-image data](evidence/per_image.csv) | Audit of 40 images and paired model results for the 20 test images |
| [Fold manifest](evidence/folds.json) | Teacher split and pilot settings recorded by the training pipeline |
| [Figures](figures/) | Workflow, data audit, paired comparison, and score distributions with a shared error map |
| [LaTeX source](retinal_sgl_ieee.tex) | IEEEtran conference manuscript |
| [Analysis script](analyze.py) | Figure and evidence generation from the saved pilot outputs |

The evaluated pipeline was [revision 7b413c6](https://github.com/A-cpu-rg/Retinal-vessel-segmentation-sgl/commit/7b413c6)
on macOS arm64 with Python 3.12.2, PyTorch 2.9.1, NumPy 2.2.6, Pillow 12.0.0,
scikit-learn 1.7.2 and Matplotlib 3.10.7. CPU execution used four intra-op
threads and one inter-op thread.

## Reproduce the figures

Set up DRIVE using the [project instructions](../../milestone2/README.md#local-setup).
The analysis also needs the test split's second manual annotations in
test/2nd_manual/ with filenames 01_manual2.gif through 20_manual2.gif. Install
the repository requirements and Matplotlib. From the repository root, run:

~~~bash
run_dir="$(mktemp -d)"

OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
python -c '
import runpy, sys, torch

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
sys.path.insert(0, "src")
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
' src/run_sgl.py \
  --device cpu --k 2 --fold-epochs 3 --final-epochs 5 \
  --patch-size 96 --repeat 8 --batch-size 4 --lr 0.0001 \
  --seed 0 --workers 0 --no-elastic --no-self-ensemble --with-baseline \
  --ckpt-dir "$run_dir/checkpoints" --teacher-dir "$run_dir/pseudo_labels" \
  --save-dir "$run_dir/test_predictions" --results "$run_dir/test_results.json" \
  --baseline-results "$run_dir/baseline_test_results.json" \
  --comparison-results "$run_dir/comparison.json"

python report/paper/analyze.py --drive data/DRIVE --run "$run_dir"
~~~

The analysis writes the figures and tabular evidence used in the paper. The
score plots use the saved 8-bit probability maps. The AUC values in the
metric table come from the pipeline's unquantized predictions.

To compile the manuscript, run Tectonic with the IEEEtran bundle from the
report/paper directory:

~~~bash
tectonic retinal_sgl_ieee.tex
~~~
