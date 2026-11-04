"""Compose unchanged head programs only after both full confirmations pass."""

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
    output = Path("weight_space/edits/clinical_v0_gate50_gate212_composed.json")
    audit_path = Path("artifacts/audit/clinical_gate50_gate212_composed_construction.json")
    freeze_path = Path("artifacts/clinical_candidate_freeze_v14_gate50_gate212.json")
    plan_path = Path("artifacts/plans/clinical_gate50_gate212_composed_calibration512.json")
    for path in (output, audit_path, freeze_path, plan_path):
        assert not path.exists(), path

    compatibility_path = Path("artifacts/audit/clinical_gate50_gate212_composition_compatibility.json")
    compatibility = read(compatibility_path)
    assert not any(compatibility["changed_coordinate_overlap"].values())
    assert compatibility["same_final_readout_and_output_bias"]
    assert compatibility["gate212_source_direction_and_destination_baseline_unchanged_in_v9"]
    sources = compatibility["weight_sources"]
    labels = ("V3", "V9", "V7")
    freezes, recipes = [], []
    for label in labels:
        record = sources[label]
        assert sha256(record["freeze"]) == record["freeze_sha256"]
        frozen = read(record["freeze"])
        assert sha256(frozen["recipe"]) == frozen["recipe_sha256"] == record["recipe_sha256"]
        assert sha256(frozen["source_checkpoint"]) == frozen["source_sha256"]
        for file, digest in frozen["source_sha256_files"].items():
            assert sha256(file) == digest
        freezes.append(frozen)
        recipes.append(read(frozen["recipe"]))
    common, parent, component = recipes

    evidence = []
    for name, reference, candidate in (
        ("gate212_shoulder", common["name"], component["name"]),
        ("gate50_midrange", "clinical_v0_relative_tail", parent["name"]),
    ):
        calibration_suffix = "calibration512_report" if name == "gate212_shoulder" else "calibration512_parent_report"
        confirmation_suffix = "full_confirmation_v3_report" if name == "gate212_shoulder" else "full_confirmation_v4_report"
        for suffix, patients, seeds in (
            (calibration_suffix, 512, [20261103]),
            (confirmation_suffix, 2016, [20261103, 20261104]),
        ):
            path = Path(f"artifacts/clinical_{name}_{suffix}.json")
            report = read(path)
            assert report["reference_recipe"] == reference
            assert report["candidate_recipe"] == candidate
            assert report["patients_per_seed"] == patients
            assert report["seeds"] == seeds and report["samples"] == 100
            assert report["both_mean_metrics_improve"]
            assert report["both_metrics_improve_on_each_seed"]
            if patients == 2016:
                assert report["full_partition_verified"]
            for raw in report["raw_audits"]:
                for role in ("reference", "candidate"):
                    result = Path(raw[role])
                    assert sha256(result) == raw[role + "_json_sha256"]
                    assert sha256(result.with_suffix(".npz")) == raw[role + "_npz_sha256"]
            evidence.append(path)

    n_common = len(common["operations"])
    assert parent["operations"][:n_common] == component["operations"][:n_common] == common["operations"]
    extra = component["operations"][n_common:]
    assert len(extra) == 3 and all(op["kind"] == "add_values" for op in extra)
    recipe = copy.deepcopy(parent)
    recipe["name"] = "clinical_v0_gate50_gate212_composed"
    recipe["parent_recipe_sha256"] = freezes[1]["recipe_sha256"]
    recipe["component_recipe_sha256"] = freezes[2]["recipe_sha256"]
    recipe["operations"] += copy.deepcopy(extra)
    recipe["rationale"] += (
        " Compose the unchanged V7 gate212 shoulder with V9. V7 is a child of V3; "
        "V9 contains the V4 relative-tail program and its own gate50 program. "
        "V7 and V9 each improved both metrics on calibration512 and on each of two "
        "full confirmation2016 seeds against their respective parents. V7's earlier "
        "confirmation128 result was mixed; the unchanged full-partition expansion "
        "is the declared selection evidence, not a retuned V7. Modified coordinates "
        "are disjoint and the readout is common. Final ideal supports overlap at54 "
        "entries on V9 calibration512, so additive accuracy is not assumed. "
        "Copy all three V7 delta operations exactly: no new amplitude, threshold, "
        "coefficient fit, gradient, optimizer, training or parameter sweep. First "
        "compare the complete composed forward against V9 and V7 on calibration512. "
        "If both metrics improve, advance unchanged to full confirmation2016 with "
        "both fixed seeds; do not select or retune from a smaller confirmation subset. "
        "Confirmation and benchmark are reused across hypotheses; no fresh-holdout claim."
    )

    source = Path(freezes[0]["source_checkpoint"])
    runners = [Inference(source, 0, r) for r in recipes]
    after = Inference(source, 0, recipe)
    for runner, frozen in zip(runners, freezes):
        assert state_hash(runner.model) == frozen["edited_state_sha256"]
    states = [runner.model.state_dict() for runner in runners]
    after_state = after.model.state_dict()
    coordinates = {}
    for key, a in states[0].items():
        b, c, d = states[1][key], states[2][key], after_state[key]
        mb, mc = b != a, c != a
        assert not bool((mb & mc).any()), key
        assert torch.equal(d, torch.where(mc, c, b)), key
        if bool((mb | mc).any()):
            coordinates[key] = {"v9": int(mb.sum()), "v7": int(mc.sum()), "overlap": 0}
    reverse = copy.deepcopy(component)
    reverse["operations"] += copy.deepcopy(parent["operations"][n_common:])
    assert state_hash(Inference(source, 0, reverse).model) == state_hash(after.model)

    component_audit_path = Path(freezes[2]["construction_audit"])
    assert sha256(component_audit_path) == freezes[2]["construction_audit_sha256"]
    construction = read(component_audit_path)["construction"]
    h0, h1 = [copy.deepcopy(r.model.out_proj).double() for r in (runners[1], after)]
    features = torch.randn(1024, 64, generator=torch.Generator().manual_seed(765)).double()
    z = h0[0].mlp[0](features)
    c = construction
    program = c["amplitude"] * sum(
        m * F.gelu(c["sharpness"] * (z[:, c["source_unit"]] - t))
        for t, m in zip(c["thresholds"], c["multipliers"])) / c["sharpness"]
    effective = h0[1].weight.flatten() @ h0[0].mlp[3].weight
    removed = sum(effective[d] * F.gelu(z[:, d]) for d in c["reused_units"])
    error = float(((h1(features) - h0(features)).flatten() - program + removed).abs().max())
    assert error < 1e-5, error
    original = Inference(source, 0).model.state_dict()
    changes = {k: int((v != original[k]).sum()) for k, v in after_state.items()
               if not torch.equal(v, original[k])}
    parent_changes = {k: int((v != states[1][k]).sum()) for k, v in after_state.items()
                      if not torch.equal(v, states[1][k])}
    assert sum(parent_changes.values()) == 384
    assert sum(changes.values()) == 1728
    assert all(not p.requires_grad and p.grad is None for p in after.model.parameters())

    output.write_text(json.dumps(recipe, indent=2) + "\n")
    audit = {
        "time_unix": time.time(), "recipe": str(output), "recipe_sha256": sha256(output),
        "builder_sha256": sha256(Path(__file__)), "parent_sha256": freezes[1]["recipe_sha256"],
        "component_sha256": freezes[2]["recipe_sha256"], "construction": construction,
        "construction_scope": "The unchanged V7 program added to V9; all parameters copied exactly.",
        "component_construction": str(component_audit_path),
        "component_construction_sha256": sha256(component_audit_path),
        "evidence_sha256": {str(p): sha256(p) for p in evidence + [compatibility_path]},
        "coordinate_changes": coordinates, "exact_state_commutativity_verified": True,
        "formula_error_on_stored_float32_weights_evaluated_in_float64": error,
        "changes_from_original": changes, "changes_from_parent": parent_changes,
        "changed_values_from_original": sum(changes.values()),
        "gradients_training_or_fitting_used": False, "test_targets_or_residuals_used": False,
        "scope": "One fixed composition. Local formula equality does not imply additive complete-model accuracy.",
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    freeze = {
        "time_unix": time.time(), "candidate": recipe["name"], "source_checkpoint": str(source),
        "source_sha256": sha256(source), "recipe": str(output), "recipe_sha256": sha256(output),
        "edited_state_sha256": state_hash(after.model),
        "source_sha256_files": freezes[0]["source_sha256_files"],
        "reference_recipe": freezes[1]["recipe"], "reference_recipe_sha256": freezes[1]["recipe_sha256"],
        "construction_audit": str(audit_path), "construction_audit_sha256": sha256(audit_path),
        "selection": "Frozen before any composed accuracy forward. Require both metrics lower than V9 and V7 on calibration512, then lower than V9 and retained V4 in each full confirmation2016 seed. No coefficient retuning or smaller confirmation-subset selection.",
        "test_history": "V1 and V4 complete tests already opened; V9 full test selected and queued. This composition uses no test targets or residuals. Any later test reuses the benchmark.",
    }
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    plan = {
        "created_unix": time.time(), "recipe_sha256": sha256(output),
        "selection": freeze["selection"],
        "jobs": [{"dataset": "mimic-iii-challenge_point", "replicate": 0, "split": "calibration", "limit": 512,
                  "samples": 100, "seed": 20261103, "chunk": 20, "batch": 1, "device": "cuda", "backend": "triton",
                  "edit": str(output), "output": "artifacts/results/clinical_point_v0_gate50_gate212_composed_cal512x100_seed1_chunk20"}],
    }
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
