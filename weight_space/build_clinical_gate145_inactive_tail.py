"""Compile a fixed capped subtraction for a strongly inactive learned path."""

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
    diagnostic = Path("artifacts/audit/clinical_relative_tail_head_both_steps_cal512.npz")
    evidence_path = Path("artifacts/audit/clinical_negative_path_regions_both_steps_cal512.json")
    output = Path("weight_space/edits/clinical_v0_gate145_inactive_tail.json")
    audit_path = Path("artifacts/audit/clinical_gate145_inactive_tail_construction.json")
    freeze_path = Path("artifacts/clinical_candidate_freeze_v10_gate145.json")
    plan_path = Path("artifacts/plans/clinical_gate145_inactive_tail_calibration512.json")
    if any(p.exists() for p in (output, audit_path, freeze_path, plan_path)):
        raise FileExistsError("Preserve previous hypotheses")
    parent, frozen, evidence = [json.loads(p.read_text()) for p in
                                (parent_path, parent_freeze_path, evidence_path)]
    assert sha256(parent_path) == frozen["recipe_sha256"]
    assert evidence["diagnostic_sha256"] == sha256(diagnostic)
    assert evidence["split"] == "calibration" and evidence["recipe"] == parent["name"]
    metadata = json.loads(diagnostic.with_suffix(".json").read_text())
    assert metadata["edit"] == parent and metadata["full_prediction_and_mask_exact_parity"]
    row = next(r for r in evidence["rows"] if r["unit"] == 145)
    assert row["region"] == {"upper": -3}
    for group in ("step1", "step2", "both", "first_only", "final_only"):
        values = row["groups"][group]
        assert values["median_normalized_residual"] > 0 and values["mean_normalized_residual"] > 0
        assert values["positive_patient_median_fraction"] > .5
    # Preserve the contrary raw-unit observation instead of implying uniform
    # improvement across physical units or all temporal support groups.
    assert row["groups"]["first_only"]["mean_raw_residual"] < 0
    source = Path(frozen["source_checkpoint"])
    assert sha256(source) == frozen["source_sha256"]
    for path, digest in frozen["source_sha256_files"].items():
        assert sha256(path) == digest
    before = Inference(source, 0, parent)
    assert state_hash(before.model) == frozen["edited_state_sha256"]
    state, prefix = before.model.state_dict(), "out_proj.0.mlp."
    readout = state["out_proj.1.weight"].flatten()
    effective = readout @ state[prefix + "3.weight"]
    assert float(effective[145]) > 0

    # One predetermined program in coordinate q=-z145: a ramp from q=3 to4,
    # then a constant cap. Its 5% scale comes from the learned output weight.
    unit, coordinate_scale = 145, -1.
    thresholds, multipliers, destinations = [3., 4.], [1., -1.], [4, 133]
    fraction, sharpness = .05, 64.
    amplitude = -fraction * float(effective[unit])
    direction = coordinate_scale * state[prefix + "0.weight"][unit]
    bias = coordinate_scale * state[prefix + "0.bias"][unit]
    deltas = {key: torch.zeros_like(state[key]) for key in
              (prefix + "0.weight", prefix + "0.bias", prefix + "3.weight")}
    with np.load(diagnostic) as data:
        hidden = [torch.from_numpy(data[key]) for key in ("hidden_step1", "hidden")]
        coordinates = [-torch.from_numpy(data[key][:, unit]) for key in ("preactivation_step1", "preactivation")]
    removed = {}
    for destination, threshold, multiplier in zip(destinations, thresholds, multipliers):
        removed[str(destination)] = [float((h[:, destination] * effective[destination]).abs().max()) for h in hidden]
        assert max(removed[str(destination)]) < 1e-6
        assert abs(float(effective[destination])) < 1e-5
        deltas[prefix + "0.weight"][destination] = sharpness * direction - state[prefix + "0.weight"][destination]
        deltas[prefix + "0.bias"][destination] = sharpness * (bias - threshold) - state[prefix + "0.bias"][destination]
        desired = multiplier * amplitude / sharpness
        deltas[prefix + "3.weight"][:, destination] = readout * ((desired - effective[destination]) / readout.square().sum())
    recipe = copy.deepcopy(parent)
    recipe["name"] = "clinical_v0_gate145_inactive_tail"
    recipe["parent_recipe_sha256"] = sha256(parent_path)
    recipe["operations"] += [{"key": key, "kind": "add_values", "values": value.tolist()}
                              for key, value in deltas.items()]
    recipe["rationale"] += (
        " A separate hypothesis uses the strongly inactive tail of positive learned path145. "
        "In the unchanged V4 calibration512 dual-step trace, z145<-3 accompanies positive "
        "final normalized residual means and medians at both steps and in the both/first-only/"
        "final-only support groups. It covers1347 first-step and1030 final-step scored paths, "
        "with positive patient-median fractions above0.55 in all those groups. Most final "
        "entries are physiological. Several variables still underpredict; the first-only "
        "group has a negative mean in physical units, so this is not a uniform MSE guarantee. "
        "The existing positive GELU path is almost shut off in this region. Reuse quiet "
        "units4,133 for A*(GELU(64*(-z-3))-GELU(64*(-z-4)))/64, "
        "A=-0.05*effective_coefficient145. The inspected boundary-3 starts the correction; "
        "one additional preactivation unit caps it at-4. Sharpness64 and the5% learned-weight "
        "fraction are fixed before accuracy evaluation and never fitted to residual amplitudes. "
        "Both ordinary denoising steps use the same edited weights. This V4 child contains "
        "neither V8, V9 nor V7. No gradients, training, optimizer, coefficient sweep, confirmation "
        "or test residuals are used."
    )
    after = Inference(source, 0, recipe)
    h0, h1 = [copy.deepcopy(r.model.out_proj).double() for r in (before, after)]
    features = torch.randn(1024, 64, generator=torch.Generator().manual_seed(760)).double()
    z = h0[0].mlp[0](features)

    def program(q):
        return amplitude * sum(m * F.gelu(sharpness * (q - t))
                               for t, m in zip(thresholds, multipliers)) / sharpness

    old_effective = h0[1].weight.flatten() @ h0[0].mlp[3].weight
    removed_function = sum(old_effective[d] * F.gelu(z[:, d]) for d in destinations)
    actual = (h1(features) - h0(features)).flatten()
    discrepancy = float((actual - program(-z[:, unit]) + removed_function).abs().max())
    assert discrepancy < 1e-5, discrepancy
    grid = torch.linspace(-10, 10, 20001, dtype=torch.float64)
    response = program(grid)
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
        "recipe": str(output), "recipe_sha256": sha256(output), "builder_sha256": sha256(Path(__file__)),
        "diagnostic_sha256": sha256(diagnostic), "evidence": str(evidence_path), "evidence_sha256": sha256(evidence_path),
        "construction": {"source_unit": unit, "coordinate_scale": coordinate_scale,
                         "response_shape": "capped_ramp", "reused_units": destinations,
                         "thresholds": thresholds, "multipliers": multipliers, "sharpness": sharpness,
                         "fixed_relative_fraction": fraction, "amplitude": amplitude, "nominal_head_cap": amplitude},
        "removed_path_maximum_contributions_step1_step2": removed,
        "formula_error_on_stored_float32_weights_evaluated_in_float64": discrepancy,
        "head_response_range_on_coordinate_minus10_plus10": [float(response.min()), float(response.max())],
        "calibration_fixed_feature_response_quantiles_by_step": [torch.quantile(
            program(q.double()), torch.tensor([0., .01, .5, .99, 1.], dtype=torch.float64)).tolist() for q in coordinates],
        "changes_from_parent": parent_changes, "changes_from_original": changes,
        "changed_values_from_original": sum(changes.values()),
        "gradients_training_or_fitting_used": False, "test_targets_or_residuals_used": False,
        "scope": "One local head formula verified on stored float32 weights. Full feedback, all draws and median selection must be evaluated separately; no global output bound or accuracy claim."
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    freeze = {
        "time_unix": time.time(), "candidate": recipe["name"], "source_checkpoint": str(source),
        "source_sha256": sha256(source), "recipe": str(output), "recipe_sha256": sha256(output),
        "edited_state_sha256": state_hash(after.model), "source_sha256_files": frozen["source_sha256_files"],
        "reference_recipe": str(parent_path), "reference_recipe_sha256": sha256(parent_path),
        "construction_audit": str(audit_path), "construction_audit_sha256": sha256(audit_path),
        "selection": "Frozen before the first full candidate accuracy forward. One fixed negative-tail hypothesis from both calibration steps, no automatic search or retuning.",
        "test_history": "V1 complete test is already opened; V4 test seed1 completed and seed2 is running. This edit uses only V4 calibration evidence. Any later test is benchmark reuse."
    }
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    plan = {"created_unix": time.time(), "recipe_sha256": sha256(output),
            "selection": "One fixed gate145 inactive-tail program versus exact V4 calibration512. Both mean metrics must improve before confirmation; no coefficient retuning after failure.",
            "jobs": [{"dataset": "mimic-iii-challenge_point", "replicate": 0, "split": "calibration", "limit": 512,
                      "samples": 100, "seed": 20261103, "chunk": 20, "batch": 1, "device": "cuda", "backend": "triton",
                      "edit": str(output), "output": "artifacts/results/clinical_point_v0_gate145_inactive_tail_cal512x100_seed1_chunk20"}]}
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
