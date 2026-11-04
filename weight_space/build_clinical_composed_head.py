"""Compose two disjoint, already interpreted head edits without new fitting."""

import copy
import json
import time
from pathlib import Path

import torch

from .checkpoints import sha256
from .run import Inference


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    directory = Path("weight_space/edits")
    paths = [directory / (name + ".json") for name in (
        "clinical_v0_affine_tail_readout", "clinical_v0_local_positive_repair", "clinical_v0_head47_sharp101")]
    base, local, gate = [json.loads(p.read_text()) for p in paths]
    shared = len(base["operations"])
    assert local["operations"][:shared] == gate["operations"][:shared] == base["operations"]
    recipe = copy.deepcopy(local)
    recipe["name"] = "clinical_v0_midrange_gate47"
    recipe["operations"] += copy.deepcopy(gate["operations"][shared:])
    recipe["rationale"] += (
        " Compose the independent 1% sharpening of negative-only head unit 47, using its already specified "
        "weights unchanged. The local positive repair reduced mean confirmation MSE but cost MAE; unit 47 "
        "sharpening reduced MAE on both confirmation seeds but slightly cost MSE. They address intermediate "
        "underestimation and a common positive negative-tail contribution through disjoint neurons. "
        "No coefficient is refitted, searched or derived from test. Test their composition as a single new "
        "hypothesis, retaining it only if full forward evaluations support joint improvement."
    )
    recipe["composition"] = {str(p): sha256(p) for p in paths}
    output = directory / "clinical_v0_midrange_gate47.json"
    audit_path = Path("artifacts/audit/clinical_midrange_gate47_construction.json")
    if output.exists() or audit_path.exists():
        raise FileExistsError("Preserve previous constructions")
    source = next(Path("../weights/ctimba_mimic-iii-challenge_point/version_0").glob("*.ckpt"))
    runners = [Inference(source, 0, edit) for edit in (base, local, gate, recipe)]
    states = [r.model.state_dict() for r in runners]
    for key, reference in states[0].items():
        a, b = states[1][key] != reference, states[2][key] != reference
        assert not bool((a & b).any()), key
        union = torch.where(a, states[1][key], states[2][key])
        assert torch.equal(union, states[3][key]), key
    heads = [copy.deepcopy(r.model.out_proj).double() for r in runners]
    x = torch.randn(512, 64, generator=torch.Generator().manual_seed(739)).double()
    y = [h(x) for h in heads]
    residual = y[3] - (y[1] + y[2] - y[0])
    assert float(residual.abs().max()) < 1e-10
    original = Inference(source, 0).model.state_dict()
    changes = {key: int((value != original[key]).sum()) for key, value in states[3].items()
               if not torch.equal(value, original[key])}
    output.write_text(json.dumps(recipe, indent=2) + "\n")
    audit = {"time_unix": time.time(), "recipe": str(output), "recipe_sha256": sha256(output),
             "source_sha256": sha256(source), "component_hashes": recipe["composition"],
             "component_changes_disjoint": True, "composed_state_is_exact_union": True,
             "shared_base_applied_once": True, "fixed_feature_head_additivity_max_error": float(residual.abs().max()),
             "scope_of_additivity": "Output head at fixed hidden inputs only; denoising feedback still requires full evaluation.",
             "changed_values": sum(changes.values()), "changed_tensors": changes,
             "new_coefficients_fitted": False, "gradients_or_test_data_used": False}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
