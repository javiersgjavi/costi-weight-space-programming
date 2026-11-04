"""Translate one measured GELU hump with a fixed, explicit bias edit."""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoints import sha256
from .run import Inference


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    parent_path = Path("weight_space/edits/physionet/clinical_v0_midrange_gate47.json")
    diagnostic = Path("artifacts/audit/clinical_composed_head_cal128_median100_v0.npz")
    output = Path("weight_space/edits/physionet/clinical_v0_composed_gate255_translate.json")
    audit_path = Path("artifacts/audit/clinical_composed_gate255_translate_construction.json")
    plan_path = Path("artifacts/plans/clinical_composed_gate255_translate_calibration128.json")
    if any(p.exists() for p in (output, audit_path, plan_path)):
        raise FileExistsError("Preserve prior hypotheses")
    parent = json.loads(parent_path.read_text())
    metadata = json.loads(diagnostic.with_suffix(".json").read_text())
    assert metadata["split"] == "calibration" and metadata["edit"] == parent
    with np.load(diagnostic) as data:
        z, r, nodes, windows = data["preactivation"][:, 255], data["residual"], data["node"], data["window"]
        coefficient = float(data["effective"][255])
    assert coefficient < 0 and not np.any(z > 0)
    rows = []
    for lo, hi in [(-1.5, -1), (-1, -.5)]:
        for group, selection in [("all", nodes >= 0), ("physiological", nodes < 34), ("static_admin_time", nodes >= 34)]:
            mask = (z >= lo) & (z < hi) & selection
            patients = np.unique(windows[mask])
            patient_medians = [float(np.median(r[mask & (windows == patient)])) for patient in patients]
            rows.append({"group": group, "z_range": [lo, hi], "n": int(mask.sum()),
                         "patients": len(patients), "median_residual": float(np.median(r[mask])),
                         "mean_residual": float(np.mean(r[mask], dtype=np.float64)),
                         "positive_patient_median_fraction": float(np.mean(np.array(patient_medians) > 0))})
    assert rows[0]["median_residual"] > 0 and rows[3]["median_residual"] < 0
    # A single conservative displacement in preactivation units, fixed here
    # before any candidate forward evaluation; no grid, fitting or derivatives.
    shift = -.01
    delta = [0.] * 256
    delta[255] = shift
    recipe = copy.deepcopy(parent)
    recipe["name"] = "clinical_v0_composed_gate255_translate"
    recipe["parent_recipe_sha256"] = sha256(parent_path)
    recipe["operations"].append({"key": "out_proj.0.mlp.0.bias", "kind": "add_values", "values": delta})
    recipe["rationale"] += (
        " New calibration median-draw preactivations show unit 255 remains entirely negative. "
        "Its negative coefficient turns GELU's negative lobe into a positive output hump. "
        "For z in [-1.5,-1), residual median is +0.00673; for z in [-1,-0.5), it is -0.00197, "
        "each across 127 patients. Translate this hump by changing only incoming bias 255 by -0.01. "
        "This lowers its left-tail contribution and raises its contribution near zero, unlike uniform attenuation. "
        "The pattern is stronger on static/admin/time variables; physiological subgroup signs are mixed. "
        "This is one fixed hypothesis on an experimental parent, not a demonstrated causal correction. "
        "No gradients, training, parameter sweep or test residuals are used. V1 test scores have already "
        "been seen in the ongoing benchmark; any later test is a repeated benchmark assessment."
    )
    source = next(Path("../weights/ctimba_mimic-iii-challenge_point/version_0").glob("*.ckpt"))
    before, after, original = [Inference(source, 0, edit) for edit in (parent, recipe, None)]
    before_state, after_state = before.model.state_dict(), after.model.state_dict()
    parent_changes = {key: int((value != before_state[key]).sum()) for key, value in after_state.items()
                      if not torch.equal(value, before_state[key])}
    assert parent_changes == {"out_proj.0.mlp.0.bias": 1}
    h0, h1 = [copy.deepcopy(runner.model.out_proj).double() for runner in (before, after)]
    x = torch.randn(512, 64, generator=torch.Generator().manual_seed(743)).double()
    zz = h0[0].mlp[0](x)[:, 255]
    c = (h0[1].weight @ h0[0].mlp[3].weight)[0, 255]
    stored_shift = float(h1[0].mlp[0].bias[255] - h0[0].mlp[0].bias[255])
    expected = c * (F.gelu(zz + stored_shift) - F.gelu(zz))
    actual = (h1(x) - h0(x)).flatten()
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    grid = torch.linspace(-10, 10, 20001, dtype=torch.float64)
    response = c * (F.gelu(grid + stored_shift) - F.gelu(grid))
    changes = {key: int((value != original.model.state_dict()[key]).sum()) for key, value in after_state.items()
               if not torch.equal(value, original.model.state_dict()[key])}
    audit = {"time_unix": time.time(), "parent": str(parent_path), "parent_sha256": sha256(parent_path),
             "diagnostic": str(diagnostic), "diagnostic_sha256": sha256(diagnostic), "calibration_evidence": rows,
             "source_sha256": sha256(source), "unit": 255, "positive_calibration_preactivations": int(np.sum(z > 0)),
             "intended_bias_shift": shift, "stored_bias_shift": stored_shift,
             "fixed_feature_formula_max_error": float((actual - expected).abs().max()),
             "fixed_feature_response_range_on_minus10_plus10": [float(response.min()), float(response.max())],
             "changes_from_parent": parent_changes, "changes_from_original": changes,
             "changed_values_from_original": sum(changes.values()),
             "caveat": "The direct head effect is not the full two-step denoising effect. Confirmation required.",
             "gradients_or_fitting_used": False, "test_targets_or_residuals_used": False}
    output.write_text(json.dumps(recipe, indent=2) + "\n")
    audit.update(recipe=str(output), recipe_sha256=sha256(output))
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    plan = {"created_unix": time.time(), "recipe_sha256": sha256(output),
            "selection": "Compare calibration with the composed parent and retained V1; advance only on joint MAE/MSE improvement. No automatic search.",
            "jobs": [{"dataset": "mimic-iii-challenge_point", "replicate": 0, "split": "calibration", "limit": 128,
                      "samples": 100, "seed": 20261103, "chunk": 20, "batch": 1, "device": "cuda", "backend": "triton",
                      "edit": str(output), "output": "artifacts/results/clinical_point_v0_composed_gate255_translate_cal128x100_seed1_chunk20"}]}
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
