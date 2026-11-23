import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from comparison import compare_metrics
from dataset import DriveDataset, teacher_filename
from losses import BCELoss, SGLoss
from metrics import METRIC_NAMES, PROTOCOLS, evaluate_prediction
from model import build_model
from sgl_labeling import make_folds, validate_pseudo_labels
from train import TrainConfig, _validate_config
from utils import _inverse, _transform


def _save(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


class CoreTests(unittest.TestCase):
    def test_manual_and_sgl_losses_have_expected_targets(self):
        pred = torch.tensor([[[[0.25, 0.75]]]])
        manual = torch.tensor([[[[0.0, 1.0]]]])
        teacher = torch.tensor([[[[1.0, 0.0]]]])
        bce = BCELoss()(pred, manual)
        self.assertTrue(torch.allclose(SGLoss()(pred, manual, teacher),
                                       bce + BCELoss()(pred, teacher)))
        self.assertFalse(torch.allclose(bce, SGLoss()(pred, manual, manual)))

    def test_folds_are_disjoint_complete_and_deterministic(self):
        ids = [str(i) for i in range(21, 41)]
        folds = make_folds(ids, 8, seed=7)
        self.assertEqual(folds, make_folds(ids, 8, seed=7))
        self.assertEqual(sorted(sum(folds, [])), ids)
        self.assertEqual(len(set(sum(folds, []))), len(ids))
        self.assertLessEqual(max(map(len, folds)) - min(map(len, folds)), 1)

    def test_transforms_round_trip(self):
        x = torch.arange(2 * 3 * 4).reshape(1, 2, 3, 4)
        for hflip in (False, True):
            for vflip in (False, True):
                for transpose in (False, True):
                    transformed = _transform(x, hflip, vflip, transpose)
                    self.assertTrue(torch.equal(
                        _inverse(transformed, hflip, vflip, transpose), x))

    def test_metrics_respect_fov(self):
        prob = np.array([[0.9, 0.1], [0.9, 0.9]], dtype=np.float32)
        gt = np.array([[1, 0], [0, 1]], dtype=np.float32)
        fov = np.array([[1, 1], [0, 1]], dtype=np.float32)
        result = evaluate_prediction(prob, gt, fov, threshold=0.5)
        self.assertEqual(result["fov"]["acc"], 1.0)
        self.assertEqual(result["fov"]["dice"], 1.0)
        self.assertEqual(result["full_image"]["acc"], 1.0)

    def test_comparison_uses_signed_sgl_minus_baseline_delta(self):
        baseline = {p: {m: 0.25 for m in METRIC_NAMES} for p in PROTOCOLS}
        sgl = {p: {m: 0.5 for m in METRIC_NAMES} for p in PROTOCOLS}
        result = compare_metrics(baseline, sgl)
        self.assertEqual(result["delta_sgl_minus_baseline"]["fov"]["dice"], 0.25)

    def test_model_forward_shape_and_range(self):
        model = build_model().eval()
        with torch.no_grad():
            enhance, seg = model(torch.zeros(1, 3, 32, 48))
        self.assertEqual(enhance.shape, (1, 3, 32, 48))
        self.assertEqual(seg.shape, (1, 1, 32, 48))
        self.assertTrue(torch.all((seg >= 0) & (seg <= 1)))

    def test_dataset_loads_original_drive_names_and_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "DRIVE"
            rgb = np.zeros((32, 48, 3), dtype=np.uint8)
            vessel = np.zeros((32, 48), dtype=np.uint8)
            vessel[15, 8:40] = 255
            fov = np.full((32, 48), 255, dtype=np.uint8)
            _save(root / "training/images/21_training.tif", rgb)
            _save(root / "training/1st_manual/21_manual1.gif", vessel)
            _save(root / "training/mask/21_training_mask.gif", fov)
            teachers = Path(tmp) / "teachers"
            _save(teachers / teacher_filename("21"), vessel)
            (teachers / "folds.json").write_text(json.dumps(
                {"k": 1, "seed": 0, "folds": [["21"]]}))

            ds = DriveDataset(root, indices=["21"], teacher_dir=teachers,
                              mode="patch", patch_size=32, repeat=1, elastic=False)
            sample = ds[0]
            self.assertEqual(sample["hr"].shape, (3, 32, 32))
            self.assertEqual(sample["te"].shape, (1, 32, 32))
            validate_pseudo_labels(teachers, ["21"])
            _save(teachers / teacher_filename("22"), vessel)
            with self.assertRaises(ValueError):
                validate_pseudo_labels(teachers, ["21", "22"])

    def test_invalid_training_settings_fail_early(self):
        with self.assertRaises(ValueError):
            _validate_config(TrainConfig(epochs=0))
        with self.assertRaises(ValueError):
            _validate_config(TrainConfig(workers=-1))


if __name__ == "__main__":
    unittest.main()
