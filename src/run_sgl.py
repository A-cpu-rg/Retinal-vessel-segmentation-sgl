"""Full SGL pipeline on DRIVE:
1. study-group pseudo-labels (sgl_labeling.py)
2. final model trained on manual labels + pseudo-labels
3. one evaluation of the chosen checkpoint on the test set
"""
import argparse
from pathlib import Path

from config import CHECKPOINT_DIR, OUTPUT_DIR, TEACHER_DIR
from evaluate import run_test_evaluation
from sgl_labeling import generate_pseudo_labels
from train import add_common_args, add_validation_args, config_from_args, train_model
from utils import get_device


def main():
    ap = argparse.ArgumentParser(description="Full SGL pipeline: pseudo-labels -> "
                                             "final model -> test evaluation.")
    add_common_args(ap)
    add_validation_args(ap)
    ap.add_argument("--k", type=int, default=8, help="number of study groups (paper: 8)")
    ap.add_argument("--fold-epochs", type=int, default=15)
    ap.add_argument("--final-epochs", type=int, default=25)
    ap.add_argument("--teacher-dir", default=str(TEACHER_DIR))
    ap.add_argument("--ckpt-dir", default=str(CHECKPOINT_DIR),
                    help="saves to <ckpt-dir>/folds/foldN and <ckpt-dir>/final")
    ap.add_argument("--skip-labeling", action="store_true",
                    help="reuse existing pseudo-labels in --teacher-dir")
    ap.add_argument("--no-self-ensemble", action="store_true")
    ap.add_argument("--save-dir", default=str(OUTPUT_DIR / "test_predictions"))
    ap.add_argument("--results", default=str(OUTPUT_DIR / "test_results.json"))
    args = ap.parse_args()

    device = get_device(args.device)
    ckpt_dir = Path(args.ckpt_dir)
    self_ensemble = not args.no_self_ensemble
    print(f"=== SGL pipeline on {device} ===")

    if args.skip_labeling:
        print(f"[run] skipping labeling, using pseudo-labels in {args.teacher_dir}")
    else:
        generate_pseudo_labels(args.root, args.teacher_dir,
                               config_from_args(args, args.fold_epochs), k=args.k,
                               ckpt_dir=ckpt_dir / "folds", device=device,
                               resume=args.resume, self_ensemble=self_ensemble)

    print("\n=== Final SGL training (manual labels + pseudo-labels) ===")
    final_ckpt = train_model(args.root, config_from_args(args, args.final_epochs),
                             ckpt_dir / "final", teacher_dir=args.teacher_dir,
                             device=device, resume=args.resume)

    run_test_evaluation(final_ckpt, args.root, device, self_ensemble=self_ensemble,
                        save_dir=args.save_dir or None, results_path=args.results)


if __name__ == "__main__":
    main()
