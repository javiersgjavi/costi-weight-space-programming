"""Describe a frozen local program's actual two-step calibration feedback."""

import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoints import sha256


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    metadata = [read(path.with_suffix(".json")) for path in (args.parent, args.candidate)]
    a_meta, b_meta = metadata
    for key in ("patient_ids", "samples", "seed", "chunk", "sigmas", "checkpoint_sha256"):
        assert a_meta[key] == b_meta[key], key
    assert all(m["split"] == "calibration" and m["full_prediction_and_mask_exact_parity"] for m in metadata)
    assert a_meta["samples"] == 100 and len(a_meta["patient_ids"]) == 512
    freeze = read(b_meta["freeze"])
    assert sha256(b_meta["freeze"]) == b_meta["freeze_sha256"]
    assert sha256(freeze["recipe"]) == freeze["recipe_sha256"]
    assert read(freeze["reference_recipe"]) == a_meta["edit"]
    assert sha256(freeze["reference_recipe"]) == freeze["reference_recipe_sha256"]
    construction_path = Path(freeze["construction_audit"])
    assert sha256(construction_path) == freeze["construction_audit_sha256"]
    program = read(construction_path)["construction"]
    unit, quiet = program["source_unit"], program["reused_units"]
    assert unit == 50 and quiet == [31, 237, 120]
    n_parent = len(a_meta["edit"]["operations"])
    assert b_meta["edit"]["operations"][:n_parent] == a_meta["edit"]["operations"]
    extra = b_meta["edit"]["operations"][n_parent:]
    assert len(extra) == 3 and all(op["kind"] == "add_values" for op in extra)
    for op in extra:
        values = np.asarray(op["values"])
        if op["key"] == "out_proj.0.mlp.3.weight":
            untouched = np.delete(values, quiet, axis=1)
        else:
            assert op["key"] in ("out_proj.0.mlp.0.weight", "out_proj.0.mlp.0.bias")
            untouched = np.delete(values, quiet, axis=0)
        assert not np.count_nonzero(untouched)
    with np.load(args.parent) as data:
        a = {key: data[key] for key in data.files}
    with np.load(args.candidate) as data:
        b = {key: data[key] for key in data.files}
    for key in ("node", "window"):
        np.testing.assert_array_equal(a[key], b[key])
    same = a["median_selected_draw"] == b["median_selected_draw"]
    untouched_columns = np.ones(256, dtype=bool)
    untouched_columns[quiet] = False
    # The first pass has the same noise and upstream weights when the final
    # median selects the same draw. Only the three programmed rows may differ.
    np.testing.assert_array_equal(a["preactivation_step1"][same][:, untouched_columns],
                                  b["preactivation_step1"][same][:, untouched_columns])
    physical_errors, invariants = [], None
    for meta, arrays in zip(metadata, (a, b)):
        reference_path = Path(meta["reference_result"])
        assert sha256(reference_path) == meta["reference_result_sha256"]
        assert sha256(reference_path.with_suffix(".npz")) == meta["reference_raw_sha256"]
        with np.load(reference_path.with_suffix(".npz")) as raw:
            fixed = {key: raw[key] for key in ("target", "mask", "window_ids", "bias", "scale")}
            if invariants is not None:
                for key, value in fixed.items():
                    np.testing.assert_array_equal(value, invariants[key])
            invariants = fixed
            mask = raw["mask"]
            position = np.nonzero(mask)
            np.testing.assert_array_equal(arrays["window"], raw["window_ids"][position[0]])
            np.testing.assert_array_equal(arrays["node"], position[2])
            bias, scale = [raw[key].reshape(-1)[arrays["node"]] for key in ("bias", "scale")]
            np.testing.assert_array_equal(arrays["prediction"] * scale + bias, raw["prediction"][mask])
            physical_errors.append((raw["prediction"] - raw["target"])[mask].astype(np.float64))
    lo, hi = program["thresholds"][0], program["thresholds"][-1]
    supports = []
    for arrays in (a, b):
        supports.append([(arrays[key][:, unit] >= lo) & (arrays[key][:, unit] < hi)
                         for key in ("preactivation_step1", "preactivation")])
    first, final = supports[0]
    delta = b["prediction"].astype(np.float64) - a["prediction"].astype(np.float64)
    losses = {"mae": [np.abs(error) for error in physical_errors],
              "mse": [error ** 2 for error in physical_errors]}

    def describe(mask):
        count = int(mask.sum())
        if not count:
            return {"entries": 0}
        result = {"entries": count, "patients": int(len(np.unique(a["window"][mask]))),
                  "same_median_draw_fraction": float(same[mask].mean()),
                  "prediction_changed_fraction": float((delta[mask] != 0).mean()),
                  "normalized_delta_quantiles": np.quantile(delta[mask], [0, .01, .5, .99, 1]).tolist(),
                  "parent_final_residual_mean": float(a["residual"][mask].mean(dtype=np.float64)),
                  "candidate_final_residual_mean": float(b["residual"][mask].mean(dtype=np.float64)),
                  "physical_loss_change_contribution_to_global_metric": {}}
        for metric, (before, after) in losses.items():
            result["physical_loss_change_contribution_to_global_metric"][metric] = float(
                (after[mask] - before[mask]).sum() / len(delta))
        return result

    groups = {"both_steps": first & final, "first_step_only": first & ~final,
              "final_step_only": ~first & final, "neither_step": ~first & ~final}
    result = {"parent_diagnostic": str(args.parent), "parent_sha256": sha256(args.parent),
              "candidate_diagnostic": str(args.candidate), "candidate_sha256": sha256(args.candidate),
              "script_sha256": sha256(Path(__file__)), "split": "calibration",
              "parent_recipe": a_meta["edit"]["name"], "candidate_recipe": b_meta["edit"]["name"],
              "patients": 512, "scoring_entries": len(delta), "samples": 100, "seed": a_meta["seed"],
              "unchanged_first_step_coordinates_exact_when_same_draw": True,
              "same_selected_draw": describe(same), "different_selected_draw": describe(~same),
              "overall": describe(np.ones(len(delta), dtype=bool)),
              "parent_ideal_support_groups": {name: describe(mask) for name, mask in groups.items()},
              "same_draw_source_coordinate_shift_quantiles_by_step": [
                  np.quantile(b[key][same, unit].astype(np.float64) - a[key][same, unit].astype(np.float64),
                              [0, .01, .5, .99, 1]).tolist()
                  for key in ("preactivation_step1", "preactivation")],
              "ideal_support_membership_changed_by_step": [int((sa != sb).sum()) for sa, sb in zip(*supports)],
              "scope": "Descriptive comparison of complete frozen forwards on calibration only. Support groups use parent median-selected paths; they are not isolated causal interventions. GELU boundary lobes and feedback can affect entries outside the ideal local interval. No derivatives, coefficient fitting, weight changes or candidate selection."}
    for metric in losses:
        total = sum(row["physical_loss_change_contribution_to_global_metric"][metric]
                    for row in result["parent_ideal_support_groups"].values() if row["entries"])
        np.testing.assert_allclose(total, result["overall"]["physical_loss_change_contribution_to_global_metric"][metric], atol=1e-12, rtol=1e-12)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "overall": result["overall"],
                      "parent_ideal_support_groups": result["parent_ideal_support_groups"]}, indent=2))


if __name__ == "__main__":
    main()
