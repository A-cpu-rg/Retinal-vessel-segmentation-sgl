"""Small synthetic end-to-end smoke run; its scores are not research results."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def save(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def make_sample(root, split, image_id):
    rng = np.random.default_rng(int(image_id))
    rgb = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    vessel = np.zeros((32, 32), dtype=np.uint8)
    vessel[8 + int(image_id) % 8, 4:28] = 255
    vessel[4:28, 8 + int(image_id) % 12] = 255
    fov = np.full((32, 32), 255, dtype=np.uint8)
    suffix = "training" if split == "training" else "test"
    save(root / split / "images" / f"{image_id}_{suffix}.tif", rgb)
    save(root / split / "1st_manual" / f"{image_id}_manual1.gif", vessel)
    save(root / split / "mask" / f"{image_id}_{suffix}_mask.gif", fov)


def main():
    project = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="sgl-smoke-") as tmp:
        work = Path(tmp)
        root = work / "DRIVE"
        for image_id in ("21", "22", "23"):
            make_sample(root, "training", image_id)
        for image_id in ("01", "02"):
            make_sample(root, "test", image_id)

        command = [
            sys.executable, str(project / "src/run_sgl.py"),
            "--root", str(root), "--device", "cpu", "--k", "2",
            "--fold-epochs", "1", "--final-epochs", "1", "--batch-size", "1",
            "--patch-size", "32", "--repeat", "1", "--no-elastic",
            "--no-self-ensemble", "--with-baseline",
            "--ckpt-dir", str(work / "checkpoints"),
            "--teacher-dir", str(work / "pseudo_labels"),
            "--save-dir", "", "--results", str(work / "sgl.json"),
            "--baseline-results", str(work / "baseline.json"),
            "--comparison-results", str(work / "comparison.json"),
        ]
        subprocess.run(command, cwd=project, check=True)
        result = json.loads((work / "comparison.json").read_text())
        assert "delta_sgl_minus_baseline" in result
        assert len(list((work / "pseudo_labels").glob("*_teacher.png"))) == 3
        print("[smoke] complete: synthetic pipeline produced all expected artifacts")


if __name__ == "__main__":
    main()
