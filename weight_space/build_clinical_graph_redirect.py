"""Redirect a fixed half of unsupported graph mixing to the saved graph."""

import copy
import json
import time
from pathlib import Path

import torch

from .checkpoints import sha256
from .finalize_clinical import state_hash
from .run import Inference


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    parent_freeze_path = Path("artifacts/clinical_candidate_freeze_v4_relative_tail.json")
    output = Path("weight_space/edits/physionet/clinical_v0_graph_redirect_half.json")
    audit_path = Path("artifacts/audit/clinical_graph_redirect_half_construction.json")
    freeze_path = Path("artifacts/clinical_candidate_freeze_v15_graph_redirect.json")
    plan_path = Path("artifacts/plans/clinical_graph_redirect_half_calibration512.json")
    for path in (output, audit_path, freeze_path, plan_path):
        assert not path.exists(), path
    frozen = json.loads(parent_freeze_path.read_text())
    parent_path, source = Path(frozen["recipe"]), Path(frozen["source_checkpoint"])
    assert sha256(source) == frozen["source_sha256"]
    assert sha256(parent_path) == frozen["recipe_sha256"]
    for path, digest in frozen["source_sha256_files"].items():
        assert sha256(path) == digest
    parent = json.loads(parent_path.read_text())
    before = Inference(source, 0, parent)
    assert state_hash(before.model) == frozen["edited_state_sha256"]
    module = before.model.input_sfe.gcn
    assert module.order == 2 and module.include_self and module.is_adp
    assert module.support_len == 3
    state = before.model.state_dict()
    assert not [key for key in state if "nodevec" in key or "support" in key]
    graph_factors = before.model.input_sfe.support[-1]
    registered = {id(p) for p in before.model.parameters()}
    assert all(id(factor) not in registered for factor in graph_factors)
    support = before.model.input_sfe.support
    physical = [a.detach().double() for a in support[:-1]]
    random_graph = torch.softmax(torch.relu(graph_factors[0].double() @ graph_factors[1].double()), dim=1)
    assert len(physical) == 2 and all(a.shape == (40, 40) for a in physical)
    for a in physical + [random_graph]:
        torch.testing.assert_close(a.sum(1), torch.ones(40, dtype=torch.float64), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(physical[0], physical[1], rtol=0, atol=0)

    # Channel groups are self, saved-forward^1,^2, saved-backward^1,^2,
    # unsupported-adaptive^1,^2. Redistribute each adaptive group's matrix
    # equally to the corresponding two saved-support groups. No loss is read.
    fraction = .5
    key = "input_sfe.gcn.mlp.weight"
    weight = state[key]
    assert weight.shape == (64, 7 * 64, 1, 1)
    blocks = weight.reshape(64, 7, 64)
    delta = torch.zeros_like(blocks)
    for power in (0, 1):
        adaptive = blocks[:, 5 + power]
        delta[:, 1 + power] = fraction * .5 * adaptive
        delta[:, 3 + power] = fraction * .5 * adaptive
        delta[:, 5 + power] = -fraction * adaptive
    recipe = copy.deepcopy(parent)
    recipe["name"] = "clinical_v0_graph_redirect_half"
    recipe["parent_recipe_sha256"] = sha256(parent_path)
    recipe["operations"].append({"key": key, "kind": "add_values", "values": delta.reshape_as(weight).tolist()})
    recipe["rationale"] += (
        " A separate structural hypothesis concerns the only AdaptiveGCN in this clinical model. "
        "Its adaptive factors are ordinary tensors in a Python support list: absent from both "
        "the released state_dict and model.parameters. The inference harness reconstructs and "
        "freezes them, but their variable relations are not learned checkpoint parameters. "
        "The saved clinical adjacency is symmetric and its normalized forward/backward supports "
        "are identical. Move a fixed half of each adaptive graph-power channel matrix to the "
        "corresponding saved supports, dividing it equally between the two existing paths. "
        "Retain the other half on the original adaptive branch. Every support is row stochastic; "
        "therefore the sum of channel matrices, and the response to features constant across "
        "variables, are preserved in real arithmetic. Float32 rounding is verified separately. "
        "This edits only one saved convolution tensor and retains the original runtime layout, "
        "all graph factors, and all V4 head changes. Half is a fixed structural midpoint, not "
        "a fitted coefficient or part of a sweep. This may still harm a network adapted to the "
        "original mixing, so require both MAE and MSE to improve in the complete paired forward. "
        "No targets, residuals, gradients, optimizer, training or metric-driven fitting are used "
        "to construct this hypothesis. No claim that the saved graph is necessarily superior."
    )
    after = Inference(source, 0, recipe)
    after_state = after.model.state_dict()
    changes_from_parent = {k: int((v != state[k]).sum()) for k, v in after_state.items()
                           if not torch.equal(v, state[k])}
    assert set(changes_from_parent) == {key}
    for old, new in zip(graph_factors, after.model.input_sfe.support[-1]):
        assert torch.equal(old, new)
    new_blocks = after_state[key].reshape(64, 7, 64).double()
    constant_matrix_error = float((new_blocks.sum(1) - blocks.double().sum(1)).abs().max())
    assert constant_matrix_error < 1e-6

    # Exercise the actual GCN packing with independent synthetic features.
    # This is a mechanism check, not a scored patient evaluation.
    generator = torch.Generator().manual_seed(766)
    x = torch.randn(2, 64, 1, 48, generator=generator).double().expand(-1, -1, 40, -1)
    packed = x.reshape(2, 64, 40 * 48)
    float64_support = physical + [[factor.double() for factor in graph_factors]]
    old_gcn, new_gcn = copy.deepcopy(module).double(), copy.deepcopy(after.model.input_sfe.gcn).double()
    a = old_gcn(packed, [2, 64, 40, 48], float64_support)
    b = new_gcn(packed, [2, 64, 40, 48], float64_support)
    constant_forward_error = float((b - a).abs().max())
    assert constant_forward_error < 1e-5, constant_forward_error
    original = Inference(source, 0).model.state_dict()
    changes = {k: int((v != original[k]).sum()) for k, v in after_state.items()
               if not torch.equal(v, original[k])}
    assert all(not p.requires_grad and p.grad is None for p in after.model.parameters())
    graph_summary = {
        "adaptive_factors_registered_or_saved": False,
        "saved_supports_identical": True,
        "saved_support_nonzero": int((physical[0] != 0).sum()),
        "adaptive_mean_self_weight": float(random_graph.diagonal().mean()),
        "adaptive_median_row_maximum": float(random_graph.max(1).values.median()),
        "adaptive_vs_saved_row_l1_median": float((random_graph - physical[0]).abs().sum(1).median()),
        "channel_group_norms_before": [float(w.norm()) for w in blocks.unbind(1)],
        "channel_group_norms_after": [float(w.norm()) for w in new_blocks.unbind(1)],
    }
    output.write_text(json.dumps(recipe, indent=2) + "\n")
    audit = {
        "time_unix": time.time(), "recipe": str(output), "recipe_sha256": sha256(output),
        "source_sha256": sha256(source), "parent_sha256": sha256(parent_path),
        "builder_sha256": sha256(Path(__file__)), "construction": {
            "response_shape": "graph_channel_redistribution", "key": key,
            "fraction_redirected": fraction, "saved_paths_share_each": fraction * .5,
            "adaptive_powers": [1, 2], "graph_factors_unchanged": True,
        },
        "graph_structure": graph_summary,
        "constant_channel_sum_maximum_error": constant_matrix_error,
        "actual_gcn_constant_across_variables_forward_maximum_error_float64": constant_forward_error,
        "changes_from_parent": changes_from_parent, "changes_from_original": changes,
        "changed_values_from_original": sum(changes.values()),
        "patient_inputs_targets_or_residuals_used": False, "gradients_training_or_fitting_used": False,
        "scope": "A fixed structural edit verified algebraically and on synthetic inputs. Accuracy is unknown until complete paired forward evaluation.",
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    freeze = {
        "time_unix": time.time(), "candidate": recipe["name"], "source_checkpoint": str(source),
        "source_sha256": sha256(source), "recipe": str(output), "recipe_sha256": sha256(output),
        "edited_state_sha256": state_hash(after.model), "source_sha256_files": frozen["source_sha256_files"],
        "reference_recipe": str(parent_path), "reference_recipe_sha256": sha256(parent_path),
        "construction_audit": str(audit_path), "construction_audit_sha256": sha256(audit_path),
        "selection": "One fixed structural redistribution. Require both MAE and MSE lower than V4 on calibration512 before unchanged full two-seed confirmation. Record failure without trying another fraction.",
        "test_history": "V1 and V4 benchmark tests already opened; V9 test queued. This construction reads weights and graph structure only, not test or confirmation targets. Any later test reuses the benchmark.",
    }
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    plan = {
        "created_unix": time.time(), "recipe_sha256": sha256(output), "selection": freeze["selection"],
        "jobs": [{"dataset": "mimic-iii-challenge_point", "replicate": 0, "split": "calibration", "limit": 512,
                  "samples": 100, "seed": 20261103, "chunk": 20, "batch": 1, "device": "cuda", "backend": "triton",
                  "edit": str(output), "output": "artifacts/results/clinical_point_v0_graph_redirect_half_cal512x100_seed1_chunk20"}],
    }
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
