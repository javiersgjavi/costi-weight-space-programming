"""Strengthen one learned availability-key component by direct weight algebra."""

import json
import time
from pathlib import Path

import torch

from .checkpoints import sha256
from .run import Inference


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    output = Path("weight_space/edits/clinical_v0_mask_key_component.json")
    audit = Path("artifacts/audit/clinical_mask_key_component_construction.json")
    if output.exists() or audit.exists():
        raise FileExistsError("Preserve previous constructions")
    diagnostic_path = Path("artifacts/audit/clinical_v1_mask_signal_cal32.json")
    diagnostic = json.loads(diagnostic_path.read_text())
    parent_path = Path("weight_space/edits/clinical_v0_affine_tail_readout.json")
    recipe = json.loads(parent_path.read_text())
    source = next(Path("../weights/ctimba_mimic-iii-challenge_point/version_0").glob("*.ckpt"))
    runner = Inference(source, 0, recipe)
    assert diagnostic["source_sha256"] == sha256(source) and diagnostic["split"] == "calibration"
    assert not diagnostic["targets_used"] and not diagnostic["gradients_used"]
    key = diagnostic["proposed_weight_key"]
    weights = runner.model.state_dict()[key]
    direction = torch.tensor(diagnostic["mask_direction"], dtype=weights.dtype)
    head, width, channels = 7, 8, 64
    rows = slice(channels + head * width, channels + (head + 1) * width)
    readout = weights[rows] @ direction
    delta = torch.zeros_like(weights)
    delta[rows] = .1 * readout[:, None] * direction[None, :]
    recipe["name"] = "clinical_v0_mask_key_component"
    recipe["rationale"] += (
        " Counterfactual forward passes through the conditioning input projection and LayerNorm, "
        "toggling only the supplied mask and keeping interpolated values fixed, identify a consistent "
        "availability direction (median cosine alignment 0.902). Spatial attention head 7 has the strongest "
        "observed-source preference. Increase only its existing key projection along this direction by 10%, "
        "leaving orthogonal key features and all query/value weights unchanged. A small-layer forward probe "
        "increased its observed-source mass from 0.29746 to 0.29798. Responses vary across queries, so this "
        "is a specific hypothesis requiring full imputation checks, not a guaranteed correction. "
        "No target values, gradients, fitting, optimizer, parameter sweep or test data determine the edit."
    )
    recipe["operations"].append({"key": key, "kind": "add_values", "values": delta.tolist()})
    recipe["construction"] = {"diagnostic": str(diagnostic_path), "diagnostic_sha256": sha256(diagnostic_path),
                               "head": head, "key_rows": [120, 128], "mask_component_factor": 1.1}
    candidate = Inference(source, 0, recipe)
    after = candidate.model.state_dict()[key]
    torch.testing.assert_close(after[:64], weights[:64], rtol=0, atol=0)
    torch.testing.assert_close(after[128:], weights[128:], rtol=0, atol=0)
    torch.testing.assert_close(after[64:120], weights[64:120], rtol=0, atol=0)
    original = Inference(source, 0)
    changes = {k: int((v != original.model.state_dict()[k]).sum())
               for k, v in candidate.model.state_dict().items()
               if not torch.equal(v, original.model.state_dict()[k])}
    output.write_text(json.dumps(recipe, indent=2) + "\n")
    record = {"time_unix": time.time(), "recipe": str(output), "recipe_sha256": sha256(output),
              "source_sha256": sha256(source), "parent_recipe_sha256": sha256(parent_path),
              "changed_tensors": changes, "changed_values": sum(changes.values()),
              "query_value_and_other_heads_exactly_preserved": True,
              "key_component_max_rounding_error": float(((after[rows] - weights[rows]) - delta[rows]).abs().max()),
              "targets_or_gradients_used": False, "hypothesis_not_yet_validated": True}
    audit.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
