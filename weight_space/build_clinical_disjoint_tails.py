"""Compose two fixed tail repairs by assigning their functions to distinct units."""

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
    base_path = Path("weight_space/edits/clinical_v0_composed_gate255_translate.json")
    relative_path = Path("weight_space/edits/clinical_v0_relative_tail.json")
    gate_path = Path("weight_space/edits/clinical_v0_gate167_tail.json")
    reports = [Path(f"artifacts/clinical_{name}_confirmation128_report.json")
               for name in ("relative_tail", "gate167_tail")]
    overlap_path = Path("artifacts/audit/clinical_tail_correction_support_overlap.json")
    output = Path("weight_space/edits/clinical_v0_disjoint_tails.json")
    audit_path = Path("artifacts/audit/clinical_disjoint_tails_construction.json")
    plan_path = Path("artifacts/plans/clinical_disjoint_tails_calibration.json")
    if any(p.exists() for p in (output, audit_path, plan_path)):
        raise FileExistsError("Preserve prior candidate recipes")
    # Both component decisions must exist before composing their unchanged
    # functions. These are validation gates, never an automatic parameter fit.
    for path in reports:
        report = json.loads(path.read_text())
        assert report["split"] == "confirmation" and report["patients_per_seed"] == 128
        assert report["seeds"] == [20261103, 20261104]
        assert report["both_mean_metrics_improve"] and report["both_metrics_improve_on_each_seed"]
    overlap = json.loads(overlap_path.read_text())
    assert overlap["recipe_sha256"] == sha256(base_path)
    assert overlap["same_scoring_entries"]["entries"] == 0
    source = next(Path("../weights/ctimba_mimic-iii-challenge_point/version_0").glob("*.ckpt"))
    paths = [base_path, relative_path, gate_path]
    recipes = [json.loads(path.read_text()) for path in paths]
    base, relative, gate = [Inference(source, 0, recipe) for recipe in recipes]
    states = [runner.model.state_dict() for runner in (base, relative, gate)]
    p = "out_proj.0.mlp."
    readout = states[0]["out_proj.1.weight"].flatten()
    for state in states[1:]:
        assert torch.equal(state["out_proj.1.weight"], states[0]["out_proj.1.weight"])
        assert torch.equal(state["out_proj.1.bias"], states[0]["out_proj.1.bias"])
    base_effective, relative_effective, gate_effective = [readout @ state[p + "3.weight"] for state in states]
    diagnostic = Path("artifacts/audit/clinical_gate255_head_cal512_median100_v0.npz")
    assert sha256(diagnostic) == overlap["diagnostic_sha256"]
    with np.load(diagnostic) as arrays:
        hidden = torch.from_numpy(arrays["hidden"])
    # Keep relative-tail units181/150 unchanged. Move the exact stored unit167
    # repair from those slots into two other almost disconnected head slots.
    mapping = [(181, 4), (150, 133)]
    deltas = {key: torch.zeros_like(states[1][key]) for key in
              (p + "0.weight", p + "0.bias", p + "3.weight")}
    removed = {}
    for source_unit, destination in mapping:
        for key in (p + "0.weight", p + "0.bias"):
            assert torch.equal(states[1][key][destination], states[0][key][destination])
            deltas[key][destination] = states[2][key][source_unit] - states[1][key][destination]
        removed[str(destination)] = float((hidden[:, destination] * base_effective[destination]).abs().max())
        assert removed[str(destination)] < 1e-6
        desired = gate_effective[source_unit]
        deltas[p + "3.weight"][:, destination] = readout * ((desired - relative_effective[destination]) / readout.square().sum())
    recipe = copy.deepcopy(recipes[1])
    recipe["name"] = "clinical_v0_disjoint_tails"
    recipe["operations"] += [{"key": key, "kind": "add_values", "values": delta.tolist()} for key, delta in deltas.items()]
    recipe["component_recipe_sha256"] = {str(path): sha256(path) for path in (relative_path, gate_path)}
    recipe["rationale"] += (
        " Compose the unchanged capped unit167 repair with the relative-tail correction. Both "
        "standalone functions lower mean MAE and MSE on both confirmation128 seeds. Their positive "
        "supports have no shared scoring entries or patients in the frozen calibration512 diagnostic, "
        "but both original recipes reuse units181/150. Preserve those relative-tail units exactly and "
        "copy the stored unit167-repair incoming functions into nearly disconnected units4/133. "
        "Set only their effective outgoing coefficients, using the minimum readout-direction edit, "
        "to the exact coefficients of the stored unit167 repair. All thresholds, slopes and caps "
        "remain unchanged; no coefficient fitting, search, gradient or test-derived edit occurs. "
        "Removing the two extra negligible old paths is accounted for in the head-additivity audit. "
        "Head-function addition is not an assumption of additive two-step imputation improvements."
    )
    combined = Inference(source, 0, recipe)
    combined_state = combined.model.state_dict()
    for unit in (181, 150):
        for key in (p + "0.weight", p + "0.bias"):
            assert torch.equal(combined_state[key][unit], states[1][key][unit])
        assert torch.equal(combined_state[p + "3.weight"][:, unit], states[1][p + "3.weight"][:, unit])
    heads = [copy.deepcopy(runner.model.out_proj).double() for runner in (base, relative, gate, combined)]
    features = torch.randn(1024, 64, generator=torch.Generator().manual_seed(4167)).double()
    outputs = [head(features).flatten() for head in heads]
    old_z = heads[0][0].mlp[0](features)
    old_effective = heads[0][1].weight.flatten() @ heads[0][0].mlp[3].weight
    def old_paths(units):
        return sum(old_effective[unit] * F.gelu(old_z[:, unit]) for unit in units)
    # Each standalone child removed181/150. The combined model removes them
    # once, plus4/133; account for this finite difference explicitly.
    expected = outputs[1] + outputs[2] - outputs[0] + old_paths([181, 150]) - old_paths([4, 133])
    error = float((outputs[3] - expected).abs().max())
    assert error < 5e-6, error
    original = Inference(source, 0).model.state_dict()
    changes = {key: int((value != original[key]).sum()) for key, value in combined_state.items()
               if not torch.equal(value, original[key])}
    audit = {"time_unix": time.time(), "source_sha256": sha256(source),
             "component_sha256": {str(path): sha256(path) for path in paths},
             "confirmation_reports": {str(path): sha256(path) for path in reports},
             "support_overlap_audit_sha256": sha256(overlap_path), "mapping": mapping,
             "newly_removed_path_maximum_calibration_contributions": removed,
             "relative_tail_programmed_units_preserved_bitwise": True,
             "head_additivity_error_after_accounting_for_removed_paths": error,
             "changed_values_from_original": sum(changes.values()), "changes_from_original": changes,
             "new_coefficients_thresholds_or_caps_selected": False, "gradients_training_fitting_or_test_targets_used": False,
             "scope": "Exact stored-function composition up to audited float32 rounding and removed old paths. Accuracy requires fresh full two-step forwards; no automatic promotion."}
    output.write_text(json.dumps(recipe, indent=2) + "\n")
    audit.update(recipe=str(output), recipe_sha256=sha256(output))
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    common = dict(dataset="mimic-iii-challenge_point", replicate=0, split="calibration", samples=100,
                  seed=20261103, chunk=20, batch=1, device="cuda", backend="triton", edit=str(output))
    jobs = [dict(common, limit=512, output="artifacts/results/clinical_point_v0_disjoint_tails_cal512x100_seed1_chunk20"),
            dict(common, limit=128, output="artifacts/results/clinical_point_v0_disjoint_tails_cal128x100_seed1_chunk20")]
    plan_path.write_text(json.dumps({"created_unix": time.time(), "recipe_sha256": sha256(output),
                                    "selection": "Compare to each standalone parent on the same512-patient calibration. Separate128 run for the common progress plot. No tuning or automatic promotion.", "jobs": jobs}, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
