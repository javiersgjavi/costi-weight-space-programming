"""Program one local, smooth triangular correction using three quiet head paths."""

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
    parent_path = Path("weight_space/edits/physionet/clinical_v0_composed_gate255_translate.json")
    parent_freeze_path = Path("artifacts/clinical_candidate_freeze_v3_gate255.json")
    diagnostic_path = Path("artifacts/audit/clinical_gate255_head_cal512_median100_v0.npz")
    evidence_path = Path("artifacts/audit/clinical_gate212_near_zero_cal512.json")
    output = Path("weight_space/edits/physionet/clinical_v0_gate212_shoulder.json")
    audit_path = Path("artifacts/audit/clinical_gate212_shoulder_construction.json")
    freeze_path = Path("artifacts/clinical_candidate_freeze_v7_gate212.json")
    plan_path = Path("artifacts/plans/clinical_gate212_shoulder_calibration512.json")
    if any(path.exists() for path in (output, audit_path, freeze_path, plan_path)):
        raise FileExistsError("Preserve previous hypotheses")
    parent, parent_freeze, evidence = [json.loads(path.read_text()) for path in
                                      (parent_path, parent_freeze_path, evidence_path)]
    assert sha256(parent_path) == parent_freeze["recipe_sha256"]
    for path, digest in parent_freeze["source_sha256_files"].items():
        assert sha256(path) == digest
    metadata = json.loads(diagnostic_path.with_suffix(".json").read_text())
    assert metadata["edit"] == parent and metadata["split"] == "calibration"
    assert evidence["diagnostic_sha256"] == sha256(diagnostic_path)
    assert evidence["unit"] == 212 and evidence["region"] == [-.5, 0.]
    for group in ("all", "physiological"):
        assert evidence[group]["median_normalized_residual"] > 0
        assert evidence[group]["mean_normalized_residual"] > 0
        assert evidence[group]["positive_patient_median_fraction"] > .5
    source = Path(parent_freeze["source_checkpoint"])
    assert sha256(source) == parent_freeze["source_sha256"]
    before = Inference(source, 0, parent)
    assert state_hash(before.model) == parent_freeze["edited_state_sha256"]
    state, prefix = before.model.state_dict(), "out_proj.0.mlp."
    readout = state["out_proj.1.weight"].flatten()
    effective = readout @ state[prefix + "3.weight"]
    assert float(effective[212]) < 0
    # Fixed design: a triangular shoulder on [-0.5, 0], centered at -0.25.
    # Its scale is a fraction of a learned coefficient, never a residual fit.
    relative_fraction, sharpness = .05, 64.
    amplitude = -relative_fraction * abs(float(effective[212]))
    destinations = [(4, -.5, 1.), (133, -.25, -2.), (144, 0., 1.)]
    direction, bias = state[prefix + "0.weight"][212], state[prefix + "0.bias"][212]
    deltas = {key: torch.zeros_like(state[key]) for key in
              (prefix + "0.weight", prefix + "0.bias", prefix + "3.weight")}
    with np.load(diagnostic_path) as data:
        hidden, calibration_z = torch.from_numpy(data["hidden"]), torch.from_numpy(data["preactivation"][:, 212])
    removed = {}
    for unit, threshold, multiplier in destinations:
        removed[str(unit)] = float((hidden[:, unit] * effective[unit]).abs().max())
        assert removed[str(unit)] < 1e-6
        assert abs(float(effective[unit])) < 1e-5
        deltas[prefix + "0.weight"][unit] = sharpness * direction - state[prefix + "0.weight"][unit]
        deltas[prefix + "0.bias"][unit] = sharpness * (bias - threshold) - state[prefix + "0.bias"][unit]
        desired = multiplier * amplitude / sharpness
        deltas[prefix + "3.weight"][:, unit] = readout * ((desired - effective[unit]) / readout.square().sum())
    recipe = copy.deepcopy(parent)
    recipe["name"] = "clinical_v0_gate212_shoulder"
    recipe["parent_recipe_sha256"] = sha256(parent_path)
    recipe["operations"] += [{"key": key, "kind": "add_values", "values": value.tolist()} for key, value in deltas.items()]
    recipe["rationale"] += (
        " A separate local hypothesis targets the near-zero shoulder of negative-coefficient unit212. "
        "In the unchanged V3 calibration512 diagnostic, z212 in [-0.5,0) occurs at456 scoring entries "
        "across211 patients. Both median and mean normalized residuals are positive overall and in "
        "the physiological subgroup; the association is mixed by variable and is not causal proof. "
        "Reuse nearly disconnected units4,133,144 to implement a smooth triangular subtraction: "
        "A*(GELU(64*(z+0.5))-2*GELU(64*(z+0.25))+GELU(64*z))/64, "
        "where A=-0.05*abs(effective_coefficient212). The ideal triangular response vanishes on "
        "both sides of the selected interval instead of extending a new constant correction above zero. "
        "GELU smoothing creates small boundary lobes; stored float32 weights are verified separately. "
        "Thresholds, midpoint, sharpness and5% coefficient fraction are fixed before candidate evaluation; "
        "none are fitted to residual amplitudes. This starts from V3 and includes neither V4's "
        "relative-tail program nor gate12 nor the rejected unit167 program. No training, gradients, "
        "optimizer, coefficient sweep, confirmation residuals or test residuals are used."
    )
    after = Inference(source, 0, recipe)
    h0, h1 = [copy.deepcopy(runner.model.out_proj).double() for runner in (before, after)]
    features = torch.randn(1024, 64, generator=torch.Generator().manual_seed(757)).double()
    z = h0[0].mlp[0](features)

    def program(values):
        return amplitude * (F.gelu(sharpness * (values + .5)) -
                            2 * F.gelu(sharpness * (values + .25)) +
                            F.gelu(sharpness * values)) / sharpness

    old_effective = h0[1].weight.flatten() @ h0[0].mlp[3].weight
    removed_function = sum(old_effective[unit] * F.gelu(z[:, unit]) for unit, _, _ in destinations)
    actual = (h1(features) - h0(features)).flatten()
    discrepancy = float((actual - program(z[:, 212]) + removed_function).abs().max())
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
    audit = {"time_unix": time.time(), "source_sha256": sha256(source), "parent_sha256": sha256(parent_path),
             "recipe": str(output), "recipe_sha256": sha256(output),
             "diagnostic_sha256": sha256(diagnostic_path), "evidence": str(evidence_path),
             "evidence_sha256": sha256(evidence_path),
             "construction": {"source_unit": 212, "reused_units": [4, 133, 144], "thresholds": [-.5, -.25, 0.],
                              "multipliers": [1., -2., 1.], "sharpness": sharpness,
                              "fixed_relative_fraction": relative_fraction, "amplitude": amplitude,
                              "ideal_piecewise_linear_peak": amplitude * .25},
             "removed_path_maximum_calibration_contributions": removed,
             "formula_error_on_stored_float32_weights_evaluated_in_float64": discrepancy,
             "head_response_range_on_minus10_plus10": [float(response.min()), float(response.max())],
             "calibration_fixed_feature_response_quantiles": torch.quantile(calibration_response, torch.tensor([0., .01, .5, .99, 1.], dtype=torch.float64)).tolist(),
             "changes_from_parent": parent_changes, "changes_from_original": changes,
             "changed_values_from_original": sum(changes.values()),
             "gradients_training_or_fitting_used": False, "test_targets_or_residuals_used": False,
             "scope": "A local intended head response; smoothing, float32 rounding, median selection and two-step feedback require complete forward evaluation. V1 test is already opened; any future test reuses the benchmark."}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    freeze = {"time_unix": time.time(), "candidate": recipe["name"], "source_checkpoint": str(source),
              "source_sha256": sha256(source), "recipe": str(output), "recipe_sha256": sha256(output),
              "edited_state_sha256": state_hash(after.model), "source_sha256_files": parent_freeze["source_sha256_files"],
              "reference_recipe": str(parent_path), "reference_recipe_sha256": sha256(parent_path),
              "construction_audit": str(audit_path), "construction_audit_sha256": sha256(audit_path),
              "selection": "Frozen before first candidate forward. Exact V3 calibration512 control. No automatic search, composition or promotion.",
              "test_history": parent_freeze["test_history"]}
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    plan = {"created_unix": time.time(), "recipe_sha256": sha256(output),
            "selection": "Single fixed gate212 shoulder program. Compare both metrics against V3 on identical512 calibration patients; do not retune if it fails.",
            "jobs": [{"dataset": "mimic-iii-challenge_point", "replicate": 0, "split": "calibration", "limit": 512,
                      "samples": 100, "seed": 20261103, "chunk": 20, "batch": 1, "device": "cuda", "backend": "triton",
                      "edit": str(output), "output": "artifacts/results/clinical_point_v0_gate212_shoulder_cal512x100_seed1_chunk20"}]}
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
