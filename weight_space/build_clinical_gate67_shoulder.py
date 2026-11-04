"""Compile one fixed subtraction on a learned negative-GELU lobe."""

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


def read(path):
    return json.loads(Path(path).read_text())


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    parent_path = Path("weight_space/edits/clinical_v0_relative_tail.json")
    parent_freeze = read("artifacts/clinical_candidate_freeze_v4_relative_tail.json")
    diagnostic = Path("artifacts/audit/clinical_relative_tail_head_both_steps_cal512.npz")
    evidence_path = Path("artifacts/audit/clinical_gate67_gate90_regions_cal512.json")
    output = Path("weight_space/edits/clinical_v0_gate67_shoulder.json")
    audit_path = Path("artifacts/audit/clinical_gate67_shoulder_construction.json")
    freeze_path = Path("artifacts/clinical_candidate_freeze_v12_gate67.json")
    plan_path = Path("artifacts/plans/clinical_gate67_shoulder_calibration512.json")
    for path in (output, audit_path, freeze_path, plan_path):
        assert not path.exists(), path
    parent, evidence = read(parent_path), read(evidence_path)
    assert sha256(parent_path) == parent_freeze["recipe_sha256"]
    assert evidence["split"] == "calibration"
    row = next(r for r in evidence["rows"] if r["model"] == "V4" and r["unit"] == 67)
    assert row["diagnostic_sha256"] == sha256(diagnostic) and row["region"] == [-1., -.5]
    for group in ("first", "final", "both"):
        values = row["groups"][group]
        assert values["mean_normalized_residual"] > 0 and values["median_normalized_residual"] > 0
        assert values["positive_patient_median_fraction"] > .6 and values["mean_physical_residual"] > 0
    assert row["groups"]["first_only"]["mean_physical_residual"] < 0
    assert row["groups"]["final_only"]["mean_normalized_residual"] < 0
    deferred = next(r for r in evidence["rows"] if r["model"] == "V4" and r["unit"] == 90)
    assert deferred["groups"]["first_only"]["mean_normalized_residual"] < 0
    metadata = read(diagnostic.with_suffix(".json"))
    assert metadata["edit"] == parent and metadata["full_prediction_and_mask_exact_parity"]
    source = Path(parent_freeze["source_checkpoint"])
    assert sha256(source) == parent_freeze["source_sha256"]
    for path, digest in parent_freeze["source_sha256_files"].items():
        assert sha256(path) == digest
    before = Inference(source, 0, parent)
    assert state_hash(before.model) == parent_freeze["edited_state_sha256"]
    state, prefix = before.model.state_dict(), "out_proj.0.mlp."
    readout = state["out_proj.1.weight"].flatten()
    effective = readout @ state[prefix + "3.weight"]
    assert float(effective[67]) < 0
    # The previously inspected half-unit bin supplies the support. Midpoint,
    # sharpness and relative scale are fixed, with no residual-amplitude fit.
    unit, thresholds, multipliers = 67, [-1., -.75, -.5], [1., -2., 1.]
    quiet, fraction, sharpness = [4, 133, 144], .05, 64.
    amplitude = -fraction * abs(float(effective[unit]))
    direction, bias = state[prefix + "0.weight"][unit], state[prefix + "0.bias"][unit]
    deltas = {key: torch.zeros_like(state[key]) for key in
              (prefix + "0.weight", prefix + "0.bias", prefix + "3.weight")}
    with np.load(diagnostic) as data:
        hidden = [torch.from_numpy(data[key]) for key in ("hidden_step1", "hidden")]
        coordinates = [torch.from_numpy(data[key][:, unit]) for key in ("preactivation_step1", "preactivation")]
    removed = {}
    for destination, threshold, multiplier in zip(quiet, thresholds, multipliers):
        removed[str(destination)] = [float((h[:, destination] * effective[destination]).abs().max()) for h in hidden]
        assert max(removed[str(destination)]) < 1e-6 and abs(float(effective[destination])) < 1e-5
        deltas[prefix + "0.weight"][destination] = sharpness * direction - state[prefix + "0.weight"][destination]
        deltas[prefix + "0.bias"][destination] = sharpness * (bias - threshold) - state[prefix + "0.bias"][destination]
        desired = multiplier * amplitude / sharpness
        deltas[prefix + "3.weight"][:, destination] = readout * ((desired - effective[destination]) / readout.square().sum())
    recipe = copy.deepcopy(parent)
    recipe["name"] = "clinical_v0_gate67_shoulder"
    recipe["parent_recipe_sha256"] = sha256(parent_path)
    recipe["operations"] += [{"key": key, "kind": "add_values", "values": value.tolist()} for key, value in deltas.items()]
    recipe["rationale"] += (
        " A separate hypothesis addresses the negative-GELU lobe of learned path67. "
        "Its negative readout coefficient makes this lobe a positive output contribution. "
        "The inspected calibration interval[-1,-0.5) has positive final residual means "
        "and medians at both actual noise-step paths; the final support covers1349 entries "
        "across343 patients, about90% physiological, with67.35% positive patient medians. "
        "The pattern remains after V9, but this candidate starts from verified V4 and "
        "contains no V9, V7, V8 or V10 operations. Reuse quiet units4,133,144 for "
        "A*(GELU(64*(z+1))-2*GELU(64*(z+0.75))+GELU(64*(z+0.5)))/64, "
        "A=-0.05*abs(effective_coefficient67). The ideal subtraction is confined to the "
        "inspected half-unit bin; small GELU boundary lobes remain. The first-only group "
        "has a negative mean physical residual and the final-only group has a slightly "
        "negative normalized mean, so this is not a uniform-error or MSE guarantee. "
        "Path90 was deferred because its first-only underprediction was much stronger. "
        "Support, midpoint, sharpness64 and5% learned-weight fraction are fixed before "
        "accuracy evaluation. No derivative, fitting, optimizer, parameter sweep, training, "
        "confirmation residuals or test residuals are used."
    )
    after = Inference(source, 0, recipe)
    h0, h1 = [copy.deepcopy(r.model.out_proj).double() for r in (before, after)]
    features = torch.randn(1024, 64, generator=torch.Generator().manual_seed(763)).double()
    z = h0[0].mlp[0](features)

    def program(values):
        return amplitude * sum(m * F.gelu(sharpness * (values - t))
                               for t, m in zip(thresholds, multipliers)) / sharpness

    old_effective = h0[1].weight.flatten() @ h0[0].mlp[3].weight
    removed_function = sum(old_effective[d] * F.gelu(z[:, d]) for d in quiet)
    error = float(((h1(features) - h0(features)).flatten() - program(z[:, unit]) + removed_function).abs().max())
    assert error < 1e-5, error
    response = program(torch.linspace(-10, 10, 20001, dtype=torch.float64))
    original = Inference(source, 0).model.state_dict()
    changes = {k: int((v != original[k]).sum()) for k, v in after.model.state_dict().items()
               if not torch.equal(v, original[k])}
    parent_changes = {k: int((v != state[k]).sum()) for k, v in after.model.state_dict().items()
                      if not torch.equal(v, state[k])}
    assert set(parent_changes) == set(deltas)
    assert all(not p.requires_grad and p.grad is None for p in after.model.parameters())
    output.write_text(json.dumps(recipe, indent=2) + "\n")
    audit = {"time_unix": time.time(), "recipe": str(output), "recipe_sha256": sha256(output),
             "builder_sha256": sha256(Path(__file__)), "parent_sha256": sha256(parent_path),
             "diagnostic_sha256": sha256(diagnostic), "evidence": str(evidence_path),
             "evidence_sha256": sha256(evidence_path),
             "construction": {"source_unit": unit, "reused_units": quiet, "thresholds": thresholds,
                              "multipliers": multipliers, "sharpness": sharpness, "fixed_relative_fraction": fraction,
                              "amplitude": amplitude, "ideal_piecewise_linear_peak": amplitude * .25},
             "removed_path_maximum_contributions_step1_step2": removed,
             "formula_error_on_stored_float32_weights_evaluated_in_float64": error,
             "head_response_range_on_minus10_plus10": [float(response.min()), float(response.max())],
             "calibration_fixed_feature_response_quantiles_by_step": [torch.quantile(program(q.double()), torch.tensor([0., .01, .5, .99, 1.], dtype=torch.float64)).tolist() for q in coordinates],
             "changes_from_original": changes, "changes_from_parent": parent_changes,
             "changed_values_from_original": sum(changes.values()),
             "gradients_training_or_fitting_used": False, "test_targets_or_residuals_used": False,
             "scope": "A fixed local head correction. Median selection and complete two-step feedback must be evaluated; the local response is not a global prediction bound."}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    freeze = {"time_unix": time.time(), "candidate": recipe["name"], "source_checkpoint": str(source),
              "source_sha256": sha256(source), "recipe": str(output), "recipe_sha256": sha256(output),
              "edited_state_sha256": state_hash(after.model), "source_sha256_files": parent_freeze["source_sha256_files"],
              "reference_recipe": str(parent_path), "reference_recipe_sha256": sha256(parent_path),
              "construction_audit": str(audit_path), "construction_audit_sha256": sha256(audit_path),
              "selection": "Frozen before first accuracy forward. One fixed gate67 lobe hypothesis versus exact V4 calibration512; no automatic retuning or promotion.",
              "test_history": "V1 and V4 tests already opened. No test targets used; all further test comparisons reuse the benchmark."}
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    plan = {"created_unix": time.time(), "recipe_sha256": sha256(output),
            "selection": "One fixed gate67 shoulder hypothesis. Require lower MAE and MSE vs V4 on512 calibration patients before confirmation; no retuning if either worsens.",
            "jobs": [{"dataset": "mimic-iii-challenge_point", "replicate": 0, "split": "calibration", "limit": 512,
                      "samples": 100, "seed": 20261103, "chunk": 20, "batch": 1, "device": "cuda", "backend": "triton",
                      "edit": str(output), "output": "artifacts/results/clinical_point_v0_gate67_shoulder_cal512x100_seed1_chunk20"}]}
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
