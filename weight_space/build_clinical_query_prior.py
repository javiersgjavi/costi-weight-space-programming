"""Program a constant availability-query offset through LayerNorm's affine identity."""

import copy
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .checkpoints import sha256
from .data import load_data
from .run import Inference


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    parent_path = Path("weight_space/edits/physionet/clinical_v0_affine_tail_readout.json")
    output = Path("weight_space/edits/physionet/clinical_v0_constant_mask_query_prior.json")
    audit_path = Path("artifacts/audit/clinical_constant_mask_query_prior_construction.json")
    plan_path = Path("artifacts/plans/clinical_constant_mask_query_prior_calibration128.json")
    if any(p.exists() for p in (output, audit_path, plan_path)):
        raise FileExistsError("Hypotheses are immutable")
    source = next(Path("../weights/ctimba_mimic-iii-challenge_point/version_0").glob("*.ckpt"))
    parent = json.loads(parent_path.read_text())
    before = Inference(source, 0, parent)
    data = load_data("mimic-iii-challenge_point")
    ids = data.indices("calibration", 32)
    batch = data.batch(ids)
    values, observed = batch["x_interpolated"], batch["mask"]
    cfem = before.model.cfem
    embedding = cfem.fe.node_embd_proj(before.model.get_node_embd())
    ln = cfem.fe.attn_s.ln1

    def encode(mask):
        return ln(cfem.input_proj(torch.cat([values, mask], -1)) + embedding)

    hidden = encode(observed)
    difference = encode(torch.ones_like(values)) - encode(torch.zeros_like(values))
    attention = cfem.fe.attn_s.attn.attn
    assert attention.in_proj_bias is None and attention.num_heads == 8
    wq, wk, wv = attention.in_proj_weight.chunk(3)
    head, width = 7, 8
    rows = slice(head * width, (head + 1) * width)
    key_differences = F.linear(difference, wk[rows])
    mean_key_difference = key_differences.mean((0, 1, 2))
    # The magnitude is fixed before candidate accuracy evaluation. The closed
    # form only converts the desired logit increment into weight units.
    fixed_mean_logit_increment = .05
    query_offset = fixed_mean_logit_increment * width ** .5 * mean_key_difference / mean_key_difference.square().sum()
    # For LayerNorm(x)=gamma*(x-mean(x))/std(x)+beta, the dot product
    # with 1/gamma is constant: sum(beta/gamma). This creates a query
    # offset by editing existing weights, without adding a new bias tensor.
    assert float(ln.weight.abs().min()) > 1e-7
    reciprocal_gamma = ln.weight.reciprocal()
    constant = reciprocal_gamma @ ln.bias
    assert abs(float(constant)) > 1e-4
    extractor = reciprocal_gamma / constant
    constant_values = hidden @ extractor
    torch.testing.assert_close(constant_values, torch.ones_like(constant_values), rtol=0, atol=2e-5)
    delta = torch.zeros_like(attention.in_proj_weight)
    delta[rows] = query_offset[:, None] * extractor[None, :]
    key = "cfem.fe.attn_s.attn.attn.in_proj_weight"
    recipe = copy.deepcopy(parent)
    recipe["name"] = "clinical_v0_constant_mask_query_prior"
    recipe["operations"].append({"key": key, "kind": "add_values", "values": delta.tolist()})
    recipe["rationale"] += (
        " A previous key-component amplification had a query-dependent sign and failed calibration. "
        "Instead, add a constant query offset to conditioning attention head 7 along its mean observed-mask "
        "key displacement. The fixed intended average counterfactual logit increment is +0.05. "
        "For LayerNorm affine output h, dot(h,1/gamma)=sum(beta/gamma); use this identity to encode "
        "the constant offset as a rank-one edit of existing query weights. Keys, values and all other "
        "heads remain exact. Calibration inputs supply the mean mask displacement while the interpolated "
        "values stay fixed; hidden target values do not enter construction. This is one analytic design, "
        "not fitting, a parameter sweep or optimization. Accuracy must be measured independently. "
        "V1 benchmark scores are already known; later tests are repeated benchmark assessments."
    )
    after = Inference(source, 0, recipe)
    changed = after.model.state_dict()[key]
    assert torch.equal(changed[:56], attention.in_proj_weight[:56])
    assert torch.equal(changed[64:], attention.in_proj_weight[64:])
    q0 = F.linear(hidden, wq[rows])
    q1 = F.linear(hidden, changed[rows])
    offset_error = (q1 - q0 - query_offset).abs()
    assert float(offset_error.max()) < 2e-5
    increments = (key_differences * query_offset).sum(-1) / width ** .5
    assert abs(float(increments.mean()) - fixed_mean_logit_increment) < 1e-6
    # Observe the mechanism, not prediction accuracy. No target is read here.
    b, t, n, _ = hidden.shape
    available = observed[..., 0].reshape(b * t, n)
    missing_queries = ~available
    k = F.linear(hidden, wk[rows]).reshape(b * t, n, width)

    def observed_mass(q):
        scores = q.reshape(b * t, n, width) @ k.transpose(-1, -2) / width ** .5
        mass = (scores.softmax(-1) * available[:, None, :]).sum(-1)
        return float(mass[missing_queries].mean())

    original = Inference(source, 0).model.state_dict()
    changes = {name: int((value != original[name]).sum()) for name, value in after.model.state_dict().items()
               if not torch.equal(value, original[name])}
    audit = {"time_unix": time.time(), "source_sha256": sha256(source), "parent_sha256": sha256(parent_path),
             "split": "calibration", "patients": 32, "patient_ids": ids.tolist(), "head": head,
             "mask_counterfactual": "Mask 0 to 1, same interpolated input and node embedding.",
             "fixed_mean_counterfactual_logit_increment": fixed_mean_logit_increment,
             "observed_mean_counterfactual_logit_increment": float(increments.mean()),
             "counterfactual_increment_quantiles": torch.quantile(increments.flatten(), torch.tensor([0., .01, .1, .5, .9, .99, 1.])).tolist(),
             "positive_counterfactual_increment_fraction": float((increments > 0).float().mean()),
             "layernorm_constant": float(constant), "constant_extractor_norm": float(extractor.norm()),
             "constant_extraction_max_error": float((constant_values - 1).abs().max()),
             "query_offset": query_offset.tolist(), "query_offset_norm": float(query_offset.norm()),
             "weight_delta_norm": float(delta.norm()), "old_head_query_weight_norm": float(wq[rows].norm()),
             "actual_constant_query_offset_max_error": float(offset_error.max()),
             "observed_source_mass_before": observed_mass(q0), "observed_source_mass_after": observed_mass(q1),
             "keys_values_and_other_heads_exactly_preserved": True,
             "changed_values": sum(changes.values()), "changed_tensors": changes,
             "targets_or_gradients_used": False, "fitting_or_parameter_sweep_used": False,
             "limitation": "Observed-source attention is a mechanism probe, not evidence of lower imputation error. The large weight-space norm exploits a constrained LayerNorm direction; output perturbation is verified directly."}
    output.write_text(json.dumps(recipe, indent=2) + "\n")
    audit.update(recipe=str(output), recipe_sha256=sha256(output))
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    plan = {"created_unix": time.time(), "recipe_sha256": sha256(output),
            "selection": "One fixed hypothesis; advance only if both calibration metrics improve over V1.",
            "jobs": [{"dataset": "mimic-iii-challenge_point", "replicate": 0, "split": "calibration", "limit": 128,
                      "samples": 100, "seed": 20261103, "chunk": 20, "batch": 1, "device": "cuda", "backend": "triton",
                      "edit": str(output), "output": "artifacts/results/clinical_point_v0_constant_mask_query_prior_cal128x100_seed1_chunk20"}]}
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({key: value for key, value in audit.items() if key not in ("patient_ids", "query_offset")}, indent=2))


if __name__ == "__main__":
    main()
