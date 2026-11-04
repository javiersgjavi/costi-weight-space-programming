"""Compile one fixed local correction to a positive learned output path."""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoints import sha256
from .finalize_clinical import state_hash
from .run import Inference


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    parent_path = Path("weight_space/edits/clinical_v0_relative_tail.json")
    parent_freeze_path = Path("artifacts/clinical_candidate_freeze_v4_relative_tail.json")
    diagnostic = Path("artifacts/audit/clinical_relative_tail_head_cal512_median100_v0.npz")
    evidence_path = Path("artifacts/audit/clinical_positive_midrange_candidates_v4_cal512.json")
    output = Path("weight_space/edits/clinical_v0_gate21_midrange.json")
    audit_path = Path("artifacts/audit/clinical_gate21_midrange_construction.json")
    freeze_path = Path("artifacts/clinical_candidate_freeze_v8_gate21.json")
    plan_path = Path("artifacts/plans/clinical_gate21_midrange_calibration512.json")
    if any(p.exists() for p in (output, audit_path, freeze_path, plan_path)):
        raise FileExistsError("Preserve previous hypotheses and measurements")
    parent, frozen, evidence = [json.loads(p.read_text()) for p in
                                (parent_path, parent_freeze_path, evidence_path)]
    assert sha256(parent_path) == frozen["recipe_sha256"]
    assert evidence["diagnostic_sha256"] == sha256(diagnostic)
    assert evidence["recipe_sha256"] == sha256(parent_path)
    assert evidence["exact_scored_predictions_and_normalized_residuals_verified"]
    metadata = json.loads(diagnostic.with_suffix(".json").read_text())
    assert metadata["edit"] == parent and metadata["split"] == "calibration"
    row = next(r for r in evidence["rows"] if r["unit"] == 21)
    for key in ("median_normalized_residual", "mean_normalized_residual"):
        assert row["midrange_1_to_4"][key] > 0
        assert row["lower_positive_0_to_1"][key] < 0
        assert row["upper_tail_at_least4"][key] < 0
    assert row["midrange_1_to_4"]["positive_patient_median_fraction"] > .5
    source = Path(frozen["source_checkpoint"])
    assert sha256(source) == frozen["source_sha256"]
    for path, digest in frozen["source_sha256_files"].items():
        assert sha256(path) == digest
    before = Inference(source, 0, parent)
    assert state_hash(before.model) == frozen["edited_state_sha256"]
    state, prefix = before.model.state_dict(), "out_proj.0.mlp."
    readout = state["out_proj.1.weight"].flatten()
    effective = readout @ state[prefix + "3.weight"]
    assert float(effective[21]) > 0

    # The coarse sign-reversal interval supplies the two boundaries. Its
    # midpoint and a fixed fraction of the learned coefficient supply shape
    # and scale; no candidate losses or residual magnitudes set parameters.
    unit, thresholds, multipliers = 21, [1., 2.5, 4.], [1., -2., 1.]
    destinations, fraction, sharpness = [4, 133, 144], .05, 64.
    amplitude = -fraction * float(effective[unit])
    direction = state[prefix + "0.weight"][unit]
    bias = state[prefix + "0.bias"][unit]
    deltas = {key: torch.zeros_like(state[key]) for key in
              (prefix + "0.weight", prefix + "0.bias", prefix + "3.weight")}
    with np.load(diagnostic) as d:
        hidden = torch.from_numpy(d["hidden"])
        calibration_z = torch.from_numpy(d["preactivation"][:, unit])
    removed = {}
    for destination, threshold, multiplier in zip(destinations, thresholds, multipliers):
        removed[str(destination)] = float((hidden[:, destination] * effective[destination]).abs().max())
        assert removed[str(destination)] < 1e-6
        assert abs(float(effective[destination])) < 1e-5
        deltas[prefix + "0.weight"][destination] = sharpness * direction - state[prefix + "0.weight"][destination]
        deltas[prefix + "0.bias"][destination] = sharpness * (bias - threshold) - state[prefix + "0.bias"][destination]
        desired = multiplier * amplitude / sharpness
        deltas[prefix + "3.weight"][:, destination] = readout * ((desired - effective[destination]) / readout.square().sum())

    recipe = copy.deepcopy(parent)
    recipe["name"] = "clinical_v0_gate21_midrange"
    recipe["parent_recipe_sha256"] = sha256(parent_path)
    recipe["operations"] += [{"key": key, "kind": "add_values", "values": value.tolist()}
                              for key, value in deltas.items()]
    recipe["rationale"] += (
        " A separate local hypothesis addresses positive learned path21. In the unchanged V4 "
        "calibration512 diagnostic, 231 median-selected scored entries across84 patients have "
        "z21 in[1,4). Both normalized residual mean and median are positive; the corresponding "
        "statistics are negative in[0,1) and atz>=4. All231 entries are physiological, mostly "
        "EtCO2; only53.57% of patient medians are positive, so the association is mixed. "
        "Reuse nearly disconnected units4,133,144 for A*(GELU(64*(z-1))-2*GELU(64*(z-2.5)) "
        "+GELU(64*(z-4)))/64, with A=-0.05*effective_coefficient21. The ideal triangular "
        "subtraction vanishes below1 and above4; smoothing introduces small boundary lobes. "
        "The coarse inspected interval, its midpoint, sharpness64 and fixed5% weight fraction "
        "are declared before any candidate accuracy evaluation. No residual-amplitude fit, "
        "gradients, training, optimizer, parameter sweep, confirmation or test residuals. "
        "This child ofV4 contains neither gate212, gate12 nor the rejected unit167 program."
    )
    after = Inference(source, 0, recipe)
    h0, h1 = [copy.deepcopy(r.model.out_proj).double() for r in (before, after)]
    features = torch.randn(1024, 64, generator=torch.Generator().manual_seed(758)).double()
    z = h0[0].mlp[0](features)

    def program(values):
        return amplitude * sum(m * F.gelu(sharpness * (values - t))
                               for t, m in zip(thresholds, multipliers)) / sharpness

    old_effective = h0[1].weight.flatten() @ h0[0].mlp[3].weight
    removed_function = sum(old_effective[d] * F.gelu(z[:, d]) for d in destinations)
    actual = (h1(features) - h0(features)).flatten()
    discrepancy = float((actual - program(z[:, unit]) + removed_function).abs().max())
    assert discrepancy < 1e-5, discrepancy
    grid = torch.linspace(-10, 10, 20001, dtype=torch.float64)
    response, calibration_response = program(grid), program(calibration_z.double())
    original = Inference(source, 0).model.state_dict()
    changes = {key: int((value != original[key]).sum()) for key, value in after.model.state_dict().items()
               if not torch.equal(value, original[key])}
    parent_changes = {key: int((value != state[key]).sum()) for key, value in after.model.state_dict().items()
                      if not torch.equal(value, state[key])}
    assert set(parent_changes) == set(deltas)
    assert all(not p.requires_grad and p.grad is None for p in after.model.parameters())
    output.write_text(json.dumps(recipe, indent=2) + "\n")
    audit = {
        "time_unix": time.time(), "source_sha256": sha256(source), "parent_sha256": sha256(parent_path),
        "recipe": str(output), "recipe_sha256": sha256(output),
        "builder_sha256": sha256(Path(__file__)), "diagnostic_sha256": sha256(diagnostic),
        "evidence": str(evidence_path), "evidence_sha256": sha256(evidence_path),
        "construction": {"source_unit": unit, "reused_units": destinations, "thresholds": thresholds,
                         "multipliers": multipliers, "sharpness": sharpness,
                         "fixed_relative_fraction": fraction, "amplitude": amplitude,
                         "ideal_piecewise_linear_peak": amplitude * 1.5},
        "removed_path_maximum_calibration_contributions": removed,
        "formula_error_on_stored_float32_weights_evaluated_in_float64": discrepancy,
        "head_response_range_on_minus10_plus10": [float(response.min()), float(response.max())],
        "calibration_fixed_feature_response_quantiles": torch.quantile(
            calibration_response, torch.tensor([0., .01, .5, .99, 1.], dtype=torch.float64)).tolist(),
        "changes_from_parent": parent_changes, "changes_from_original": changes,
        "changed_values_from_original": sum(changes.values()),
        "gradients_training_or_fitting_used": False, "test_targets_or_residuals_used": False,
        "scope": "Head formula verification only. All100 draws and both denoising steps need full forward validation; the global prediction change is not bounded by the local head formula."
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    freeze = {
        "time_unix": time.time(), "candidate": recipe["name"], "source_checkpoint": str(source),
        "source_sha256": sha256(source), "recipe": str(output), "recipe_sha256": sha256(output),
        "edited_state_sha256": state_hash(after.model), "source_sha256_files": frozen["source_sha256_files"],
        "reference_recipe": str(parent_path), "reference_recipe_sha256": sha256(parent_path),
        "construction_audit": str(audit_path), "construction_audit_sha256": sha256(audit_path),
        "selection": "Frozen before first full candidate accuracy forward. One fixed hypothesis, paired V4 calibration512 control. No automatic selection or retuning.",
        "test_history": "V1 complete test already opened; unchanged V4 complete test currently running. This edit uses only V4 calibration evidence. Any future candidate test is benchmark reuse."
    }
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    plan = {
        "created_unix": time.time(), "recipe_sha256": sha256(output),
        "selection": "One fixed unit21 local program. Compare both metrics versus V4 on identical512 calibration patients. Failure is recorded without coefficient retuning.",
        "jobs": [{"dataset": "mimic-iii-challenge_point", "replicate": 0, "split": "calibration", "limit": 512,
                  "samples": 100, "seed": 20261103, "chunk": 20, "batch": 1, "device": "cuda", "backend": "triton",
                  "edit": str(output), "output": "artifacts/results/clinical_point_v0_gate21_midrange_cal512x100_seed1_chunk20"}]
    }
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
