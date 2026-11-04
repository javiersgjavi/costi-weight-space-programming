"""Compile one fixed compensation for a learned negative GELU shoulder."""

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


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    parent_path = Path("weight_space/edits/physionet/clinical_v0_relative_tail.json")
    parent_freeze = read("artifacts/clinical_candidate_freeze_v4_relative_tail.json")
    diagnostic = Path("artifacts/audit/clinical_relative_tail_head_both_steps_cal512.npz")
    evidence_path = Path("artifacts/audit/clinical_gate36_near_zero_regions_cal512.json")
    output = Path("weight_space/edits/physionet/clinical_v0_gate36_near_zero.json")
    audit_path = Path("artifacts/audit/clinical_gate36_near_zero_construction.json")
    freeze_path = Path("artifacts/clinical_candidate_freeze_v13_gate36.json")
    plan_path = Path("artifacts/plans/clinical_gate36_near_zero_calibration512.json")
    for path in (output, audit_path, freeze_path, plan_path):
        assert not path.exists(), path
    parent, evidence = read(parent_path), read(evidence_path)
    assert sha256(parent_path) == parent_freeze["recipe_sha256"]
    assert evidence["split"] == "calibration"
    for row in evidence["rows"]:
        assert row["unit"] == 36 and row["region"] == [-.5, 0.]
        assert sha256(row["diagnostic"]) == row["diagnostic_sha256"]
        for values in row["groups"].values():
            assert values["mean_normalized_residual"] < 0
            assert values["median_normalized_residual"] < 0
            assert values["mean_physical_residual"] < 0
            assert values["negative_patient_median_fraction"] > .6
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
    unit, thresholds, multipliers = 36, [-.5, -.25, 0.], [1., -2., 1.]
    # These quiet paths are disjoint from V9's and V12's destinations. Their
    # negligible output contribution, rather than errors, identifies them.
    quiet, fraction, sharpness = [132, 203, 34], .05, 64.
    assert float(effective[unit]) > 0
    amplitude = fraction * float(effective[unit])
    direction, bias = state[prefix + "0.weight"][unit], state[prefix + "0.bias"][unit]
    deltas = {key: torch.zeros_like(state[key]) for key in
              (prefix + "0.weight", prefix + "0.bias", prefix + "3.weight")}
    with np.load(diagnostic) as data:
        hidden = [torch.from_numpy(data[key]) for key in ("hidden_step1", "hidden")]
    removed = {}
    for destination, threshold, multiplier in zip(quiet, thresholds, multipliers):
        removed[str(destination)] = [float((h[:, destination] * effective[destination]).abs().max()) for h in hidden]
        assert max(removed[str(destination)]) < 1e-6 and abs(float(effective[destination])) < 1e-5
        deltas[prefix + "0.weight"][destination] = sharpness * direction - state[prefix + "0.weight"][destination]
        deltas[prefix + "0.bias"][destination] = sharpness * (bias - threshold) - state[prefix + "0.bias"][destination]
        desired = multiplier * amplitude / sharpness
        deltas[prefix + "3.weight"][:, destination] = readout * ((desired - effective[destination]) / readout.square().sum())
    recipe = copy.deepcopy(parent)
    recipe["name"] = "clinical_v0_gate36_near_zero"
    recipe["parent_recipe_sha256"] = sha256(parent_path)
    recipe["operations"] += [{"key": key, "kind": "add_values", "values": value.tolist()} for key, value in deltas.items()]
    recipe["rationale"] += (
        " Learned path36 has a positive effective readout coefficient, so its negative-GELU "
        "shoulder subtracts signal. The preexisting calibration bin[-0.5,0) shows negative "
        "final residual means and medians at both noise steps, including the first-only "
        "and final-only groups. Physical residual means are negative in all these groups. "
        "The same pattern remains after V9. This independent candidate starts from V4 "
        "and uses quiet units132,203,34 for A*(GELU(64*(z+0.5))-2*GELU(64*(z+0.25))" 
        "+GELU(64*z))/64, A=0.05*effective_coefficient36. Support, midpoint, sharpness64 "
        "and5% learned-weight fraction are fixed before accuracy evaluation. It contains "
        "no V9 or V12 edits; only five final-support entries overlap V9's ideal gate50 "
        "support, which is not proof of independent full-model effects. Near-zero "
        "calibration residuals and GELU boundary lobes can still lead to overshoot. No "
        "gradients, training, residual coefficient fitting, optimizer, parameter sweep, "
        "confirmation residuals or test residuals are used."
    )
    after = Inference(source, 0, recipe)
    h0, h1 = [copy.deepcopy(r.model.out_proj).double() for r in (before, after)]
    features = torch.randn(1024, 64, generator=torch.Generator().manual_seed(1363)).double()
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
             "changes_from_original": changes, "changes_from_parent": parent_changes,
             "changed_values_from_original": sum(changes.values()),
             "gradients_training_or_fitting_used": False, "test_targets_or_residuals_used": False,
             "scope": "Fixed local compensation for a negative learned GELU contribution. Complete two-step feedback and median selection must be evaluated; the local response is not a global prediction bound."}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    freeze = {"time_unix": time.time(), "candidate": recipe["name"], "source_checkpoint": str(source),
              "source_sha256": sha256(source), "recipe": str(output), "recipe_sha256": sha256(output),
              "edited_state_sha256": state_hash(after.model), "source_sha256_files": parent_freeze["source_sha256_files"],
              "reference_recipe": str(parent_path), "reference_recipe_sha256": sha256(parent_path),
              "construction_audit": str(audit_path), "construction_audit_sha256": sha256(audit_path),
              "selection": "Frozen before first accuracy forward. One fixed gate36 near-zero compensation versus V4; no retuning or automatic promotion.",
              "test_history": "V1 and V4 tests already opened. No test targets used; any future test comparison reuses the benchmark."}
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    plan = {"created_unix": time.time(), "recipe_sha256": sha256(output),
            "selection": "One fixed learned gate36 shoulder compensation. Require lower MAE and MSE vs V4 on512 calibration patients before confirmation; no retuning if either worsens.",
            "jobs": [{"dataset": "mimic-iii-challenge_point", "replicate": 0, "split": "calibration", "limit": 512,
                      "samples": 100, "seed": 20261103, "chunk": 20, "batch": 1, "device": "cuda", "backend": "triton",
                      "edit": str(output), "output": "artifacts/results/clinical_point_v0_gate36_near_zero_cal512x100_seed1_chunk20"}]}
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
