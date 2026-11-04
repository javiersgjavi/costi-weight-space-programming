"""Describe measured GELU paths on calibration; never fit or propose weights."""

import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoints import sha256


def summarize(mask, residual, windows, nodes):
    count = int(mask.sum())
    if not count:
        return {"n": 0}
    r = residual[mask]
    return {
        "n": count,
        "patients": int(len(np.unique(windows[mask]))),
        "median_normalized_residual": float(np.median(r)),
        "mean_normalized_residual": float(np.mean(r, dtype=np.float64)),
        "positive_residual_fraction": float(np.mean(r > 0)),
        "physiological_fraction": float(np.mean(nodes[mask] < 34)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    metadata = json.loads(args.diagnostic.with_suffix(".json").read_text())
    assert metadata["split"] == "calibration"
    with np.load(args.diagnostic) as data:
        hidden, z, coefficients = data["hidden"], data["preactivation"], data["effective"]
        residual, windows, nodes = data["residual"], data["window"], data["node"]
    importance = np.mean(np.abs(hidden * coefficients[None, :]), axis=0)
    bounds = [-float("inf"), -3, -2, -1.5, -1, -.5, 0, .5, 1, 2, 4, float("inf")]
    rows = []
    for unit in np.argsort(-importance):
        unit = int(unit)
        rows.append({
            "unit": unit,
            "coefficient": float(coefficients[unit]),
            "mean_absolute_output_contribution": float(importance[unit]),
            "z_quantiles": np.quantile(z[:, unit], [0, .01, .1, .5, .9, .99, 1]).tolist(),
            "positive_preactivations": int(np.sum(z[:, unit] > 0)),
            "bins": [
                {"lower": None if not np.isfinite(lo) else lo,
                 "upper": None if not np.isfinite(hi) else hi,
                 **summarize((z[:, unit] >= lo) & (z[:, unit] < hi), residual, windows, nodes)}
                for lo, hi in zip(bounds[:-1], bounds[1:])
            ],
        })
    report = {
        "diagnostic": str(args.diagnostic), "diagnostic_sha256": sha256(args.diagnostic),
        "recipe": metadata["edit"]["name"], "split": "calibration",
        "patients": int(len(np.unique(windows))), "scoring_entries": len(residual),
        "evaluated_patients": metadata.get("evaluated_patients"),
        "patients_field_scope": "Patients contributing at least one scored entry; zero-entry patients are excluded from hidden/residual arrays.",
        "samples": metadata["samples"],
        "interpretation": "Residual is prediction minus truth, normalized. Conditional associations are diagnostic, not evidence of a causal effect. No derivatives, fitting, coefficient sweep or prospective-loss optimization.",
        "overall": summarize(np.ones(len(residual), dtype=bool), residual, windows, nodes),
        "physiological": summarize(nodes < 34, residual, windows, nodes),
        "static_admin_time": summarize(nodes >= 34, residual, windows, nodes),
        "units": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "overall": report["overall"],
                      "leading_units": [r["unit"] for r in rows[:20]]}, indent=2))


if __name__ == "__main__":
    main()
