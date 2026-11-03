"""Default paths and shared constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "DRIVE"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TEACHER_DIR = OUTPUT_DIR / "pseudo_labels"

# A pixel is "vessel" if its probability is above this. Value from the
# reference code; the paper does not give one.
THRESHOLD = 100 / 255.0

# The network halves the image 4 times, so height and width must be multiples of 16.
SIZE_MULTIPLE = 16

# Full images are zero-padded to a square whose side is a multiple of 32
# (608 x 608 for DRIVE, the same size the reference code used).
PAD_MULTIPLE = 32
