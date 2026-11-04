"""Compose two already frozen, individually confirmed local head programs."""

import copy
import json
import time
from pathlib import Path

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
    output = Path("weight_space/edits/physionet/clinical_v0_gate50_gate145_composed.json")
    audit_path = Path("artifacts/audit/clinical_gate50_gate145_composed_construction.json")
    freeze_path = Path("artifacts/clinical_candidate_freeze_v11_gate50_gate145.json")
    plan_path = Path("artifacts/plans/clinical_gate50_gate145_composed_calibration512.json")
    for path in (output, audit_path, freeze_path, plan_path):
        assert not path.exists(), path
    freeze_paths = [Path("artifacts/clinical_candidate_freeze_" + name + ".json")
                    for name in ("v4_relative_tail", "v9_gate50", "v10_gate145")]
    freezes = [read(path) for path in freeze_paths]
    recipes = [read(freeze["recipe"]) for freeze in freezes]
    base, parent, component = recipes
    evidence_paths = []
    for name in ("gate50_midrange", "gate145_inactive_tail"):
        for suffix in ("calibration512_parent_report", "confirmation128_report"):
            path = Path(f"artifacts/clinical_{name}_{suffix}.json")
            report = read(path)
            assert report["both_mean_metrics_improve"]
            assert report["both_metrics_improve_on_each_seed"]
            assert report["reference_recipe"] == base["name"]
            assert report["samples"] == 100
            assert report["patients_per_seed"] == (512 if suffix.startswith("calibration") else 128)
            assert report["seeds"] == ([20261103] if suffix.startswith("calibration") else [20261103, 20261104])
            evidence_paths.append(path)
    compatibility_path = Path("artifacts/audit/clinical_gate50_gate145_composition_compatibility.json")
    compatibility = read(compatibility_path)
    assert compatibility["coordinate_overlap_zero"]
    assert compatibility["fixed_feature_head_additivity_maximum_error_float64"] < 1e-10
    for path, freeze in zip(freeze_paths, freezes):
        assert sha256(path) == compatibility["freeze_sha256"][str(path)]
        assert sha256(freeze["recipe"]) == freeze["recipe_sha256"]
        assert sha256(freeze["source_checkpoint"]) == freeze["source_sha256"]
        for file, digest in freeze["source_sha256_files"].items():
            assert sha256(file) == digest
    n_base = len(base["operations"])
    assert parent["operations"][:n_base] == component["operations"][:n_base] == base["operations"]
    extra = component["operations"][n_base:]
    assert len(extra) == 3 and all(op["kind"] == "add_values" for op in extra)
    recipe = copy.deepcopy(parent)
    recipe["name"] = "clinical_v0_gate50_gate145_composed"
    recipe["parent_recipe_sha256"] = freezes[1]["recipe_sha256"]
    recipe["component_recipe_sha256"] = freezes[2]["recipe_sha256"]
    recipe["operations"] += copy.deepcopy(extra)
    recipe["rationale"] += (
        " Compose the unchanged V10 inactive-tail program with V9. Both individual "
        "children of V4 reduced MAE and MSE in calibration512 and in each of two "
        "confirmation128 seeds. This is a single fixed composition, with no amplitude, "
        "threshold or coefficient tuning. V9 changes units31,237,120; V10 changes4,133. "
        "The changed parameter coordinates are disjoint and the fixed-feature head "
        "functions add. However, activation supports overlap strongly (612 first-step "
        "and648 final-step calibration entries), so independent or additive accuracy "
        "gains are not assumed. The complete composed forward must beat both component "
        "controls in MAE and MSE before advancing. No rejected V8/V7/V5 edits are added. "
        "No gradients, optimizer, training, residual-amplitude fitting or test residuals."
    )
    source = Path(freezes[0]["source_checkpoint"])
    runners = [Inference(source, 0, r) for r in recipes]
    after = Inference(source, 0, recipe)
    for runner, freeze in zip(runners, freezes):
        assert state_hash(runner.model) == freeze["edited_state_sha256"]
    states = [runner.model.state_dict() for runner in runners]
    after_state = after.model.state_dict()
    coordinate_audit = {}
    for key, a in states[0].items():
        b, c, d = states[1][key], states[2][key], after_state[key]
        mb, mc = b != a, c != a
        assert not bool((mb & mc).any()), key
        expected = torch.where(mc, c, b)
        assert torch.equal(d, expected), key
        if bool((mb | mc).any()):
            coordinate_audit[key] = {"v9": int(mb.sum()), "v10": int(mc.sum()), "overlap": 0}
    reverse = copy.deepcopy(component)
    reverse["operations"] += copy.deepcopy(parent["operations"][n_base:])
    reversed_runner = Inference(source, 0, reverse)
    assert state_hash(reversed_runner.model) == state_hash(after.model)
    component_audit_path = Path(freezes[2]["construction_audit"])
    assert sha256(component_audit_path) == freezes[2]["construction_audit_sha256"]
    construction = read(component_audit_path)["construction"]
    h0 = copy.deepcopy(runners[1].model.out_proj).double()
    h1 = copy.deepcopy(after.model.out_proj).double()
    features = torch.randn(1024, 64, generator=torch.Generator().manual_seed(762)).double()
    z = h0[0].mlp[0](features)
    coordinate = construction["coordinate_scale"] * z[:, construction["source_unit"]]
    program = construction["amplitude"] * sum(
        m * F.gelu(construction["sharpness"] * (coordinate - t))
        for t, m in zip(construction["thresholds"], construction["multipliers"])) / construction["sharpness"]
    effective = h0[1].weight.flatten() @ h0[0].mlp[3].weight
    removed = sum(effective[d] * F.gelu(z[:, d]) for d in construction["reused_units"])
    error = float(((h1(features) - h0(features)).flatten() - program + removed).abs().max())
    assert error < 1e-5, error
    original = Inference(source, 0).model.state_dict()
    changes = {k: int((v != original[k]).sum()) for k, v in after_state.items()
               if not torch.equal(v, original[k])}
    parent_changes = {k: int((v != states[1][k]).sum()) for k, v in after_state.items()
                      if not torch.equal(v, states[1][k])}
    assert sum(parent_changes.values()) == 245
    assert sum(changes.values()) == 1589
    assert all(not p.requires_grad and p.grad is None for p in after.model.parameters())
    output.write_text(json.dumps(recipe, indent=2) + "\n")
    audit = {"time_unix": time.time(), "recipe": str(output), "recipe_sha256": sha256(output),
             "builder_sha256": sha256(Path(__file__)), "parent_sha256": freezes[1]["recipe_sha256"],
             "component_sha256": freezes[2]["recipe_sha256"], "construction": construction,
             "construction_scope": "The added V10 program relative to parent V9; parameters copied unchanged.",
             "component_construction": str(component_audit_path),
             "component_construction_sha256": sha256(component_audit_path),
             "evidence_sha256": {str(p): sha256(p) for p in evidence_paths + [compatibility_path]},
             "coordinate_changes": coordinate_audit, "exact_state_commutativity_verified": True,
             "formula_error_on_stored_float32_weights_evaluated_in_float64": error,
             "changes_from_original": changes, "changes_from_parent": parent_changes,
             "changed_values_from_original": sum(changes.values()),
             "gradients_training_or_fitting_used": False, "test_targets_or_residuals_used": False,
             "scope": "One fixed composition, no additive accuracy claim. Full feedback and median selection must be measured."}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    freeze = {"time_unix": time.time(), "candidate": recipe["name"], "source_checkpoint": str(source),
              "source_sha256": sha256(source), "recipe": str(output), "recipe_sha256": sha256(output),
              "edited_state_sha256": state_hash(after.model),
              "source_sha256_files": freezes[0]["source_sha256_files"],
              "reference_recipe": freezes[1]["recipe"], "reference_recipe_sha256": freezes[1]["recipe_sha256"],
              "construction_audit": str(audit_path), "construction_audit_sha256": sha256(audit_path),
              "selection": "Frozen before first accuracy forward. No retuning of either previously fixed component; must improve both metrics versus both individual components before confirmation.",
              "test_history": "V1 test opened and V4 test underway. No test targets used for this composition; any later test reuses the benchmark."}
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    plan = {"created_unix": time.time(), "recipe_sha256": sha256(output),
            "selection": "One unchanged sum of two mechanistic programs; compare against V9 and V10 on identical calibration512. No automatic retuning or promotion.",
            "jobs": [{"dataset": "mimic-iii-challenge_point", "replicate": 0, "split": "calibration", "limit": 512,
                      "samples": 100, "seed": 20261103, "chunk": 20, "batch": 1, "device": "cuda", "backend": "triton",
                      "edit": str(output), "output": "artifacts/results/clinical_point_v0_gate50_gate145_composed_cal512x100_seed1_chunk20"}]}
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
