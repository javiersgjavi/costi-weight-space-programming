"""Construct one fixed translation of a measured negative GELU output lobe.

No optimization, derivatives, parameter sweep or test-derived edit. The recipe
is frozen before its first evaluation and must be compared to its exact parent.
"""

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
    diagnostic = Path("artifacts/audit/clinical_gate255_head_cal512_median100_v0.npz")
    output = Path("weight_space/edits/physionet/clinical_v0_gate12_translate.json")
    audit_path = Path("artifacts/audit/clinical_gate12_translate_construction.json")
    freeze_path = Path("artifacts/clinical_candidate_freeze_v6_gate12.json")
    plan_path = Path("artifacts/plans/clinical_gate12_translate_calibration512.json")
    if any(p.exists() for p in (output, audit_path, freeze_path, plan_path)):
        raise FileExistsError("Preserve prior hypotheses and evaluations")
    parent = json.loads(parent_path.read_text())
    parent_freeze = json.loads(parent_freeze_path.read_text())
    assert sha256(parent_path) == parent_freeze["recipe_sha256"]
    for path, digest in parent_freeze["source_sha256_files"].items():
        assert sha256(path) == digest
    metadata = json.loads(diagnostic.with_suffix(".json").read_text())
    assert metadata["split"] == "calibration" and metadata["edit"] == parent
    assert metadata["evaluated_patients"] == 512 and metadata["samples"] == 100
    with np.load(diagnostic) as data:
        z, r = data["preactivation"][:, 12], data["residual"]
        nodes, windows = data["node"], data["window"]
        coefficient = float(data["effective"][12])
    assert coefficient < 0 and not np.any(z >= 0)
    rows = []
    # Fixed structural regions on the two sides of the negative GELU lobe.
    # Regional residuals describe the hypothesis; they never set the shift.
    for lo, hi in [(-2., -1.5), (-1., -.5)]:
        for group, selection in [("all", nodes >= 0), ("physiological", nodes < 34),
                                 ("static_admin_time", nodes >= 34)]:
            mask = (z >= lo) & (z < hi) & selection
            patients = np.unique(windows[mask])
            patient_medians = [np.median(r[mask & (windows == idx)]) for idx in patients]
            rows.append({"group": group, "z_range": [lo, hi], "n": int(mask.sum()),
                         "patients": len(patients), "median_residual": float(np.median(r[mask])),
                         "mean_residual": float(np.mean(r[mask], dtype=np.float64)),
                         "positive_patient_median_fraction": float(np.mean(np.array(patient_medians) > 0))})
    assert rows[0]["median_residual"] > 0 and rows[3]["median_residual"] < 0
    shift = -.01
    values = [0.] * 256
    values[12] = shift
    recipe = copy.deepcopy(parent)
    recipe["name"] = "clinical_v0_gate12_translate"
    recipe["parent_recipe_sha256"] = sha256(parent_path)
    recipe["operations"].append({"key": "out_proj.0.mlp.0.bias", "kind": "add_values", "values": values})
    recipe["rationale"] += (
        " Additional fixed hypothesis: translate unit12's negative GELU lobe with incoming bias -0.01. "
        "All 51979 median-selected calibration preactivations are negative; its effective coefficient is negative. "
        "On 512 calibration patients, residual medians are positive at z in [-2,-1.5) and negative at [-1,-0.5). "
        "The displacement lowers the left-lobe contribution and raises it closer to zero. "
        "Regional residual means are negative on both sides, so a joint MAE/MSE gain is uncertain. "
        "This is one fixed displacement, chosen before candidate evaluation, on the V3 parent. "
        "No V4 or rejected unit167-tail program is included. No gradients, fitting, sweep, confirmation "
        "residuals or test residuals are used to construct the edit. The previously opened V1 test remains benchmark reuse."
    )
    source = Path(parent_freeze["source_checkpoint"])
    assert sha256(source) == parent_freeze["source_sha256"]
    before, after, original = [Inference(source, 0, edit) for edit in (parent, recipe, None)]
    assert state_hash(before.model) == parent_freeze["edited_state_sha256"]
    before_state, after_state = before.model.state_dict(), after.model.state_dict()
    parent_changes = {key: int((value != before_state[key]).sum()) for key, value in after_state.items()
                      if not torch.equal(value, before_state[key])}
    assert parent_changes == {"out_proj.0.mlp.0.bias": 1}
    h0, h1 = [copy.deepcopy(runner.model.out_proj).double() for runner in (before, after)]
    x = torch.randn(512, 64, generator=torch.Generator().manual_seed(743)).double()
    zz = h0[0].mlp[0](x)[:, 12]
    c = (h0[1].weight @ h0[0].mlp[3].weight)[0, 12]
    stored_shift = float(h1[0].mlp[0].bias[12] - h0[0].mlp[0].bias[12])
    expected = c * (F.gelu(zz + stored_shift) - F.gelu(zz))
    actual = (h1(x) - h0(x)).flatten()
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    grid = torch.linspace(-10, 10, 20001, dtype=torch.float64)
    response = c * (F.gelu(grid + stored_shift) - F.gelu(grid))
    changes = {key: int((value != original.model.state_dict()[key]).sum()) for key, value in after_state.items()
               if not torch.equal(value, original.model.state_dict()[key])}
    assert all(not p.requires_grad and p.grad is None for p in after.model.parameters())
    output.write_text(json.dumps(recipe, indent=2) + "\n")
    audit = {"time_unix": time.time(), "parent": str(parent_path), "parent_sha256": sha256(parent_path),
             "diagnostic": str(diagnostic), "diagnostic_sha256": sha256(diagnostic), "calibration_evidence": rows,
             "source_sha256": sha256(source), "unit": 12, "positive_calibration_preactivations": int(np.sum(z > 0)),
             "intended_bias_shift": shift, "stored_bias_shift": stored_shift,
             "fixed_feature_formula_max_error": float((actual - expected).abs().max()),
             "fixed_feature_response_range_on_minus10_plus10": [float(response.min()), float(response.max())],
             "changes_from_parent": parent_changes, "changes_from_original": changes,
             "changed_values_from_original": sum(changes.values()),
             "recipe": str(output), "recipe_sha256": sha256(output),
             "caveat": "Regional means disagree with medians on the left lobe. The direct head response is not the two-step model response. Require both MAE and MSE to improve; do not retune after failure.",
             "gradients_or_fitting_used": False, "test_targets_or_residuals_used": False}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    freeze = {"time_unix": time.time(), "candidate": recipe["name"],
              "source_checkpoint": str(source), "source_sha256": sha256(source),
              "recipe": str(output), "recipe_sha256": sha256(output),
              "edited_state_sha256": state_hash(after.model),
              "source_sha256_files": parent_freeze["source_sha256_files"],
              "reference_recipe": str(parent_path), "reference_recipe_sha256": sha256(parent_path),
              "construction_audit": str(audit_path), "construction_audit_sha256": sha256(audit_path),
              "selection": "Frozen before first calibration forward. Primary calibration512 with exact V3 parent control. No automatic search or promotion.",
              "test_history": parent_freeze["test_history"]}
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    plan = {"created_unix": time.time(), "recipe_sha256": sha256(output),
            "selection": "Single fixed gate12 hypothesis. Compare both metrics to V3 on exactly the same 512 calibration patients; stop if either worsens. No test or automatic promotion.",
            "jobs": [{"dataset": "mimic-iii-challenge_point", "replicate": 0, "split": "calibration", "limit": 512,
                      "samples": 100, "seed": 20261103, "chunk": 20, "batch": 1, "device": "cuda", "backend": "triton",
                      "edit": str(output), "output": "artifacts/results/clinical_point_v0_gate12_translate_cal512x100_seed1_chunk20"}]}
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
