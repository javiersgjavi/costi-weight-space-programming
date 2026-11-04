"""Bound an overextended negative head path using two dormant existing neurons."""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoints import sha256
from .run import Inference


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    parent_path = Path("weight_space/edits/clinical_v0_composed_gate255_translate.json")
    diagnostic_path = Path("artifacts/audit/clinical_gate255_head_cal512_median100_v0.npz")
    steps_path = Path("artifacts/audit/clinical_gate167_tail_steps_cal_enriched.json")
    output = Path("weight_space/edits/clinical_v0_gate167_tail.json")
    audit_path = Path("artifacts/audit/clinical_gate167_tail_construction.json")
    plan_path = Path("artifacts/plans/clinical_gate167_tail_calibration.json")
    if any(p.exists() for p in (output, audit_path, plan_path)):
        raise FileExistsError("Preserve previously constructed hypotheses")
    parent = json.loads(parent_path.read_text())
    metadata = json.loads(diagnostic_path.with_suffix(".json").read_text())
    steps = json.loads(steps_path.read_text())
    assert metadata["edit"] == parent and metadata["split"] == "calibration"
    assert steps["recipe_sha256"] == sha256(parent_path)
    assert steps["contrast"] == "preactivation167 - 2"
    assert steps["targets_or_gradients_used"] is False
    with np.load(diagnostic_path) as d:
        z, residual = d["preactivation"][:, 167], d["residual"]
        windows, nodes, hidden = d["window"], d["node"], torch.from_numpy(d["hidden"])
    tail = z >= 2
    evidence = {"entries": int(tail.sum()), "patients": len(np.unique(windows[tail])),
                "median_normalized_residual": float(np.median(residual[tail])),
                "mean_normalized_residual": float(np.mean(residual[tail], dtype=np.float64)),
                "positive_residual_fraction": float(np.mean(residual[tail] > 0)),
                "variable_counts": {str(int(node)): int(np.sum(tail & (nodes == node))) for node in np.unique(nodes[tail])},
                "patients_with_negative_median": sum(bool(np.median(residual[tail & (windows == patient)]) < 0) for patient in np.unique(windows[tail]))}
    assert evidence["patients"] >= 8
    assert evidence["median_normalized_residual"] < 0 and evidence["mean_normalized_residual"] < 0
    source = next(Path("../weights/ctimba_mimic-iii-challenge_point/version_0").glob("*.ckpt"))
    before = Inference(source, 0, parent)
    state = before.model.state_dict()
    prefix = "out_proj.0.mlp."
    readout = state["out_proj.1.weight"].flatten()
    effective = readout @ state[prefix + "3.weight"]
    assert effective[167] < 0
    # Fixed design: 5% of the existing negative path, ramp over coarse diagnostic
    # bins [2,4], then constant. No residual magnitudes enter the coefficient.
    relative_slope, sharpness = .05, 16.
    amplitude = relative_slope * float(-effective[167])
    destinations = [(181, 2., 1.), (150, 4., -1.)]
    deltas = {key: torch.zeros_like(state[key]) for key in
              (prefix + "0.weight", prefix + "0.bias", prefix + "3.weight")}
    removed = {}
    for unit, threshold, sign in destinations:
        removed[str(unit)] = float((hidden[:, unit] * effective[unit]).abs().max())
        assert removed[str(unit)] < 1e-6
        deltas[prefix + "0.weight"][unit] = sharpness * state[prefix + "0.weight"][167] - state[prefix + "0.weight"][unit]
        deltas[prefix + "0.bias"][unit] = sharpness * (state[prefix + "0.bias"][167] - threshold) - state[prefix + "0.bias"][unit]
        desired = sign * amplitude / sharpness
        deltas[prefix + "3.weight"][:, unit] = readout * ((desired - effective[unit]) / readout.square().sum())
    recipe = copy.deepcopy(parent)
    recipe["name"] = "clinical_v0_gate167_tail"
    recipe["operations"] += [{"key": key, "kind": "add_values", "values": delta.tolist()} for key, delta in deltas.items()]
    recipe["parent_recipe_sha256"] = sha256(parent_path)
    recipe["rationale"] += (
        " Unit167 has a negative effective output coefficient and is usually inactive. Its rare "
        "positive activation above 2 mostly denotes the hospital-admission-time variable in calibration. "
        "The original coarse bins show negative mean and median residuals in this tail on both the "
        "128-patient composed-head diagnostic and the expanded 512-patient gate255 diagnostic; a "
        "minority of physiological cases have the opposite sign. Program two dormant units to add "
        "A*(GELU(16*(z167-2))-GELU(16*(z167-4)))/16, A=0.05*abs(coefficient167). "
        "This reduces 5% of the negative slope only between the fixed coarse bin boundaries 2 and 4, "
        "then caps the response. The existing low and negative activations are almost unaffected. "
        "The coefficient is taken directly from the weight, not fitted to residual magnitudes. "
        "Both denoising steps were probed using calibration inputs without reading targets. "
        "No gradients, optimizer, training, parameter sweep, or test-derived edit is used. "
        "Full forwards must establish whether this head-level interpretation improves accuracy. "
        "This is a separate child of gate255, not a composition with the relative-tail candidate."
    )
    candidate = Inference(source, 0, recipe)
    old_head, new_head = [copy.deepcopy(runner.model.out_proj).double() for runner in (before, candidate)]
    features = torch.randn(512, 64, generator=torch.Generator().manual_seed(167)).double()
    old_z = old_head[0].mlp[0](features)
    x = old_z[:, 167]
    programmed = amplitude * (F.gelu(sharpness * (x - 2)) - F.gelu(sharpness * (x - 4))) / sharpness
    old_effective = old_head[1].weight.flatten() @ old_head[0].mlp[3].weight
    removed_function = sum(old_effective[unit] * F.gelu(old_z[:, unit]) for unit, _, _ in destinations)
    discrepancy = float(((new_head(features) - old_head(features)).flatten() - programmed + removed_function).abs().max())
    assert discrepancy < 5e-6, discrepancy
    grid = torch.linspace(-10, 10, 20001, dtype=torch.float64)
    response = amplitude * (F.gelu(sharpness * (grid - 2)) - F.gelu(sharpness * (grid - 4))) / sharpness
    original = Inference(source, 0).model.state_dict()
    changes = {key: int((value != original[key]).sum()) for key, value in candidate.model.state_dict().items()
               if not torch.equal(value, original[key])}
    audit = {"time_unix": time.time(), "source_sha256": sha256(source), "parent_sha256": sha256(parent_path),
             "diagnostic_sha256": sha256(diagnostic_path), "step_probe_sha256": sha256(steps_path),
             "calibration_tail_evidence": evidence,
             "construction": {"source_unit": 167, "reused_units": [181, 150], "thresholds": [2, 4],
                              "sharpness": sharpness, "relative_slope_reduction": relative_slope,
                              "nominal_head_output_cap": 2 * amplitude},
             "removed_path_maximum_calibration_contributions": removed,
             "formula_error_on_stored_float32_weights_evaluated_in_float64": discrepancy,
             "head_response_range_on_minus10_plus10": [float(response.min()), float(response.max())],
             "changes_from_original": changes, "changed_values_from_original": sum(changes.values()),
             "gradients_training_or_fitting_used": False, "test_targets_or_residuals_used": False,
             "limitation": "Sparse calibration association, primarily an administrative variable; no uniform physiological benefit claimed. Bounds describe direct head response, not two-step feedback."}
    output.write_text(json.dumps(recipe, indent=2) + "\n")
    audit.update(recipe=str(output), recipe_sha256=sha256(output))
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    common = dict(dataset="mimic-iii-challenge_point", replicate=0, split="calibration", samples=100,
                  seed=20261103, chunk=20, batch=1, device="cuda", backend="triton", edit=str(output))
    jobs = [dict(common, limit=512, output="artifacts/results/clinical_point_v0_gate167_tail_cal512x100_seed1_chunk20"),
            dict(common, limit=128, output="artifacts/results/clinical_point_v0_gate167_tail_cal128x100_seed1_chunk20")]
    plan_path.write_text(json.dumps({"created_unix": time.time(), "recipe_sha256": sha256(output),
                                    "selection": "Fixed hypothesis; primary screen pairs the 512 patients with existing gate255 control. Independent 128 run preserves progress-plot comparability. No automatic promotion or tuning.", "jobs": jobs}, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
