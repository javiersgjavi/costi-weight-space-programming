"""Describe shared head weights on both median-selected denoising paths."""

import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoints import sha256
from .inspect_clinical_head import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("--previous-final-diagnostic", type=Path,
                        help="Optional older trace of this exact recipe; when supplied all final arrays must match")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    metadata = json.loads(args.diagnostic.with_suffix(".json").read_text())
    assert metadata["split"] == "calibration"
    assert metadata["full_prediction_and_mask_exact_parity"]
    with np.load(args.diagnostic) as data:
        arrays = {key: data[key] for key in data.files}
    reference_path = Path(metadata["reference_result"])
    assert sha256(reference_path) == metadata["reference_result_sha256"]
    assert sha256(reference_path.with_suffix(".npz")) == metadata["reference_raw_sha256"]
    reference = json.loads(reference_path.read_text())
    assert reference["edit"] == metadata["edit"] and reference["split"] == "calibration"
    assert reference["samples"] == metadata["samples"] == 100
    assert reference["seed"] == metadata["seed"] and reference["sigmas"] == metadata["sigmas"]
    with np.load(reference_path.with_suffix(".npz")) as raw:
        mask = raw["mask"]
        positions = np.nonzero(mask)
        np.testing.assert_array_equal(arrays["node"], positions[2])
        np.testing.assert_array_equal(arrays["window"], raw["window_ids"][positions[0]])
        assert raw["window_ids"].tolist() == metadata["patient_ids"]
        bias, scale = [raw[key].reshape(-1)[arrays["node"]] for key in ("bias", "scale")]
        np.testing.assert_array_equal(arrays["prediction"] * scale + bias, raw["prediction"][mask])
        np.testing.assert_array_equal(arrays["residual"], arrays["prediction"] - (raw["target"][mask] - bias) / scale)
    if args.previous_final_diagnostic:
        previous = json.loads(args.previous_final_diagnostic.with_suffix(".json").read_text())
        assert previous["split"] == "calibration" and metadata["edit"] == previous["edit"]
        for key in ("patient_ids", "samples", "seed", "chunk", "checkpoint_sha256"):
            assert metadata[key] == previous[key], key
        with np.load(args.previous_final_diagnostic) as data:
            for key in ("prediction", "residual", "hidden", "preactivation", "effective", "node", "window"):
                assert np.array_equal(arrays[key], data[key]), key
    residual, windows, nodes = [arrays[key] for key in ("residual", "window", "node")]
    coefficients = arrays["effective"]
    bounds = [-float("inf"), -3, -2, -1.5, -1, -.5, 0, .5, 1, 2, 4, float("inf")]
    steps = []
    for suffix, sigma in (("_step1", 80), ("", 20.92)):
        hidden, z = arrays["hidden" + suffix], arrays["preactivation" + suffix]
        importance = np.mean(np.abs(hidden * coefficients[None, :]), axis=0)
        rows = []
        for unit in np.argsort(-importance):
            unit = int(unit)
            rows.append({
                "unit": unit, "coefficient": float(coefficients[unit]),
                "mean_absolute_output_contribution": float(importance[unit]),
                "z_quantiles": np.quantile(z[:, unit], [0, .01, .1, .5, .9, .99, 1]).tolist(),
                "bins": [{"lower": None if not np.isfinite(lo) else lo,
                          "upper": None if not np.isfinite(hi) else hi,
                          **summarize((z[:, unit] >= lo) & (z[:, unit] < hi), residual, windows, nodes)}
                         for lo, hi in zip(bounds[:-1], bounds[1:])],
            })
        steps.append({"sigma": sigma, "units": rows})
    z1, z2 = arrays["preactivation_step1"], arrays["preactivation"]
    programs = [
        ("V4 positive relative contrast", z1[:, 38] - z1[:, 198] > 0, z2[:, 38] - z2[:, 198] > 0),
        ("V7 gate212 shoulder", (z1[:, 212] >= -.5) & (z1[:, 212] < 0),
         (z2[:, 212] >= -.5) & (z2[:, 212] < 0)),
        ("Rejected V8 gate21 midrange", (z1[:, 21] >= 1) & (z1[:, 21] < 4),
         (z2[:, 21] >= 1) & (z2[:, 21] < 4)),
        ("V9 gate50 midrange", (z1[:, 50] >= 1) & (z1[:, 50] < 4),
         (z2[:, 50] >= 1) & (z2[:, 50] < 4)),
    ]
    support_rows = []
    for name, first, final in programs:
        groups = {"both_steps": first & final, "first_step_only": first & ~final,
                  "final_step_only": ~first & final, "neither": ~first & ~final}
        support_rows.append({"program": name, "groups": {
            label: summarize(mask, residual, windows, nodes) for label, mask in groups.items()}})
    report = {
        "diagnostic": str(args.diagnostic), "diagnostic_sha256": sha256(args.diagnostic),
        "previous_final_diagnostic": str(args.previous_final_diagnostic) if args.previous_final_diagnostic else None,
        "previous_final_diagnostic_sha256": sha256(args.previous_final_diagnostic) if args.previous_final_diagnostic else None,
        "script_sha256": sha256(Path(__file__)), "split": "calibration",
        "recipe": metadata["edit"]["name"], "samples": 100, "seed": metadata["seed"],
        "patients": metadata["evaluated_patients"], "scoring_entries": len(residual),
        "final_arrays_exactly_match_previous_diagnostic": True if args.previous_final_diagnostic else None,
        "exact_scored_prediction_residual_and_identity_parity": True,
        "reference_result": str(reference_path), "reference_result_sha256": sha256(reference_path),
        "steps": steps, "existing_program_ideal_support": support_rows,
        "interpretation": "Every residual is the same final normalized prediction minus truth. First-step associations are not first-step prediction errors or causal sensitivities. The paths belong to the actual final median-selected draw. Existing-program masks describe ideal supports, not GELU boundary lobes or full feedback effects. No derivatives, fitting, coefficient selection or new weights."
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "existing_program_support": support_rows,
                      "leading_units_by_step": [[r["unit"] for r in s["units"][:16]] for s in steps]}, indent=2))


if __name__ == "__main__":
    main()
