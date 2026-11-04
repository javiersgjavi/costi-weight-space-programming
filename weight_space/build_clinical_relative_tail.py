"""Compile a bounded correction for a rare reversal of two learned head paths."""

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
    parent_path = Path("weight_space/edits/physionet/clinical_v0_composed_gate255_translate.json")
    diagnostic_path = Path("artifacts/audit/clinical_gate255_head_cal512_median100_v0.npz")
    evidence_path = Path("artifacts/audit/clinical_gate255_positive_contrast_cal512.json")
    steps_path = Path("artifacts/audit/clinical_gate255_contrast_steps_cal_enriched.json")
    output = Path("weight_space/edits/physionet/clinical_v0_relative_tail.json")
    audit_path = Path("artifacts/audit/clinical_relative_tail_construction.json")
    plan_path = Path("artifacts/plans/clinical_relative_tail_calibration.json")
    if any(path.exists() for path in (output, audit_path, plan_path)):
        raise FileExistsError("Preserve previous hypotheses")
    parent = json.loads(parent_path.read_text())
    evidence, steps = [json.loads(path.read_text()) for path in (evidence_path, steps_path)]
    metadata = json.loads(diagnostic_path.with_suffix(".json").read_text())
    assert metadata["edit"] == parent and metadata["split"] == "calibration"
    assert evidence["diagnostic_sha256"] == sha256(diagnostic_path)
    assert steps["recipe_sha256"] == sha256(parent_path)
    assert evidence["positive_contrast_patients"] >= 8
    assert evidence["positive_contrast_median_residual"] < 0 and evidence["positive_contrast_mean_residual"] < 0
    assert all(step["all_draw_entries"]["positive_fraction"] < .01 for step in steps["steps"])
    checkpoint = next(Path("../weights/ctimba_mimic-iii-challenge_point/version_0").glob("*.ckpt"))
    before = Inference(checkpoint, 0, parent)
    state = before.model.state_dict()
    prefix = "out_proj.0.mlp."
    readout = state["out_proj.1.weight"].flatten()
    effective = readout @ state[prefix + "3.weight"]
    # Fixed design constants, selected before any candidate accuracy result.
    # Zero detects reversal of the two learned paths; unit contrast caps it.
    relative_cap, sharpness = .05, 64.
    amplitude = relative_cap * float(effective[38])
    assert amplitude > 0
    direction = state[prefix + "0.weight"][38] - state[prefix + "0.weight"][198]
    bias = state[prefix + "0.bias"][38] - state[prefix + "0.bias"][198]
    destinations = [(181, 0., 1.), (150, 1., -1.)]
    deltas = {key: torch.zeros_like(state[key]) for key in
              (prefix + "0.weight", prefix + "0.bias", prefix + "3.weight")}
    with np.load(diagnostic_path) as diagnostic:
        hidden = torch.from_numpy(diagnostic["hidden"])
    removed = {}
    for unit, threshold, sign in destinations:
        removed[str(unit)] = float((hidden[:, unit] * effective[unit]).abs().max())
        assert removed[str(unit)] < 1e-6
        deltas[prefix + "0.weight"][unit] = sharpness * direction - state[prefix + "0.weight"][unit]
        deltas[prefix + "0.bias"][unit] = sharpness * (bias - threshold) - state[prefix + "0.bias"][unit]
        desired_coefficient = sign * amplitude / sharpness
        deltas[prefix + "3.weight"][:, unit] = readout * ((desired_coefficient - effective[unit]) / readout.square().sum())
    recipe = copy.deepcopy(parent)
    recipe["name"] = "clinical_v0_relative_tail"
    recipe["operations"] += [{"key": key, "kind": "add_values", "values": value.tolist()} for key, value in deltas.items()]
    recipe["parent_recipe_sha256"] = sha256(parent_path)
    recipe["rationale"] += (
        " The two positive head paths 38 and 198 have partly aligned incoming weights (cosine 0.69). "
        "Normally z38 is below z198. A positive contrast z38-z198 occurs in only 15 median-selected "
        "scoring entries across 11 of 512 calibration patients in the gate255 model; its median and mean "
        "normalized residuals are negative. Six cases overpredict, so the association is neither universal "
        "nor causal proof. An enriched target-free probe shows this contrast remains sparse at both noise "
        "steps. Reuse nearly disconnected units 181 and 150 to implement a small capped ramp: "
        "A*(GELU(64*d)-GELU(64*(d-1)))/64, d=z38-z198, A=0.05*effective_coefficient38. "
        "The zero threshold detects order reversal, the unit contrast caps the response, and the fixed "
        "5% amplitude is not fitted to residual magnitudes. Sharpness confines the small negative GELU "
        "lobe near the threshold. No derivatives, training, optimizer, coefficient sweep or test targets "
        "are used. This is a new hypothesis requiring complete forward evaluations."
    )
    candidate = Inference(checkpoint, 0, recipe)
    old_head, new_head = [copy.deepcopy(runner.model.out_proj).double() for runner in (before, candidate)]
    features = torch.randn(512, 64, generator=torch.Generator().manual_seed(751)).double()
    z = old_head[0].mlp[0](features)
    contrast = z[:, 38] - z[:, 198]
    programmed = amplitude * (F.gelu(sharpness * contrast) - F.gelu(sharpness * (contrast - 1))) / sharpness
    old_effective = old_head[1].weight.flatten() @ old_head[0].mlp[3].weight
    removed_function = sum(old_effective[unit] * F.gelu(z[:, unit]) for unit, _, _ in destinations)
    actual = (new_head(features) - old_head(features)).flatten()
    discrepancy = float((actual - programmed + removed_function).abs().max())
    assert discrepancy < 5e-6, discrepancy
    grid = torch.linspace(-10, 10, 20001, dtype=torch.float64)
    response = amplitude * (F.gelu(sharpness * grid) - F.gelu(sharpness * (grid - 1))) / sharpness
    original = Inference(checkpoint, 0).model.state_dict()
    changes = {key: int((value != original[key]).sum()) for key, value in candidate.model.state_dict().items()
               if not torch.equal(value, original[key])}
    audit = {"time_unix": time.time(), "source_sha256": sha256(checkpoint), "parent_sha256": sha256(parent_path),
             "diagnostic_sha256": sha256(diagnostic_path), "evidence_sha256": sha256(evidence_path),
             "step_probe_sha256": sha256(steps_path), "positive_contrast_evidence": evidence,
             "construction": {"contrast_units": [38, 198], "reused_units": [181, 150], "thresholds": [0, 1],
                              "sharpness": sharpness, "fixed_relative_cap": relative_cap, "nominal_head_output_cap": amplitude},
             "removed_path_maximum_calibration_contributions": removed,
             "formula_error_on_stored_float32_weights_evaluated_in_float64": discrepancy,
             "head_response_range_on_minus10_plus10": [float(response.min()), float(response.max())],
             "changes_from_original": changes, "changed_values_from_original": sum(changes.values()),
             "gradients_training_or_fitting_used": False, "test_targets_or_residuals_used": False,
             "scope": "Bounded direct-head correction; feedback through two denoising steps requires full validation. V1 test was previously opened in this ongoing benchmark."}
    output.write_text(json.dumps(recipe, indent=2) + "\n")
    audit.update(recipe=str(output), recipe_sha256=sha256(output))
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    common = dict(dataset="mimic-iii-challenge_point", replicate=0, split="calibration", samples=100,
                  seed=20261103, chunk=20, batch=1, device="cuda", backend="triton")
    jobs = [dict(common, limit=512, edit=str(parent_path), output="artifacts/results/clinical_point_v0_gate255_control_cal512x100_seed1_chunk20"),
            dict(common, limit=512, edit=str(output), output="artifacts/results/clinical_point_v0_relative_tail_cal512x100_seed1_chunk20"),
            dict(common, limit=128, edit=str(output), output="artifacts/results/clinical_point_v0_relative_tail_cal128x100_seed1_chunk20")]
    plan_path.write_text(json.dumps({"created_unix": time.time(), "recipe_sha256": sha256(output),
                                    "selection": "Primary screen: paired 512-patient calibration MAE and MSE versus gate255. The 128-patient evaluation is separately recorded for the existing common-cohort progress plot; no automatic promotion or coefficient fitting.",
                                    "jobs": jobs}, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
