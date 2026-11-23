"""Create a transparent baseline-versus-SGL metric comparison."""
import json
from pathlib import Path

from metrics import METRIC_NAMES, PROTOCOLS


def compare_metrics(baseline, sgl):
    """Return both metric sets and signed SGL-minus-baseline deltas."""
    delta = {
        protocol: {
            metric: float(sgl[protocol][metric] - baseline[protocol][metric])
            for metric in METRIC_NAMES
        }
        for protocol in PROTOCOLS
    }
    return {"baseline": baseline, "sgl": sgl, "delta_sgl_minus_baseline": delta}


def format_comparison(comparison, protocol="fov"):
    rows = [
        f"{'metric':<14}{'baseline':>12}{'SGL':>12}{'delta':>12}",
    ]
    for metric in METRIC_NAMES:
        rows.append(
            f"{metric:<14}"
            f"{comparison['baseline'][protocol][metric]:>12.4f}"
            f"{comparison['sgl'][protocol][metric]:>12.4f}"
            f"{comparison['delta_sgl_minus_baseline'][protocol][metric]:>+12.4f}"
        )
    return "\n".join(rows)


def save_comparison(path, baseline, sgl, **metadata):
    result = {**metadata, **compare_metrics(baseline, sgl)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result
