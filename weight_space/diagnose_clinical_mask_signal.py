"""Trace the observed-mask direction through the actual conditioning weights."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .checkpoints import sha256
from .data import load_data
from .run import Inference


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    output = Path("artifacts/audit/clinical_v1_mask_signal_cal32.json")
    if output.exists():
        raise FileExistsError(output)
    source = next(Path("../weights/ctimba_mimic-iii-challenge_point/version_0").glob("*.ckpt"))
    recipe = json.loads(Path("weight_space/edits/clinical_v0_affine_tail_readout.json").read_text())
    runner = Inference(source, 0, recipe)
    data = load_data("mimic-iii-challenge_point")
    ids = data.indices("calibration", 32)
    batch = data.batch(ids)
    v, observed = batch["x_interpolated"], batch["mask"]
    cfem = runner.model.cfem
    embedding = cfem.fe.node_embd_proj(runner.model.get_node_embd())

    def encode(mask):
        return cfem.fe.attn_s.ln1(cfem.input_proj(torch.cat([v, mask], -1)) + embedding)

    h = encode(observed)
    # Toggle only the supplied mask bit, keeping the interpolated input fixed.
    # The hidden truth at missing entries is never used.
    differences = encode(torch.ones_like(v)) - encode(torch.zeros_like(v))
    direction = differences.mean((0, 1, 2))
    direction = direction / direction.norm()
    cosine = F.cosine_similarity(differences, direction, dim=-1)
    layer = cfem.fe.attn_s.attn.attn
    assert layer.in_proj_bias is None and layer.num_heads == 8
    wq, wk, _ = layer.in_proj_weight.chunk(3)
    b, t, n, c = h.shape
    q = F.linear(h, wq).reshape(b * t, n, 8, -1).transpose(1, 2)
    k = F.linear(h, wk).reshape(b * t, n, 8, -1).transpose(1, 2)
    available = observed[..., 0].reshape(b * t, n)
    missing = ~available[:, None, :].expand(-1, 8, -1)
    logits = q @ k.transpose(-1, -2) / q.shape[-1] ** .5

    def mass(scores):
        values = (scores.softmax(-1) * available[:, None, None, :]).sum(-1)
        return [values[:, head][missing[:, head]].mean().item() for head in range(8)]

    # This head had the largest observed-source attention in an independent
    # conditioning probe. Test its learned mask projection, not its value path.
    head = 7
    readout = wk[head * 8:(head + 1) * 8] @ direction
    delta = torch.zeros_like(wk)
    delta[head * 8:(head + 1) * 8] = .1 * readout[:, None] * direction[None, :]
    new_k = F.linear(h, wk + delta).reshape(b * t, n, 8, -1).transpose(1, 2)
    new_logits = q @ new_k.transpose(-1, -2) / q.shape[-1] ** .5
    q_mask_response = (q[:, head] @ readout)[missing[:, head]]
    result = {"source_sha256": sha256(source), "split": "calibration", "patients": 32,
              "patient_ids": ids.tolist(), "targets_used": False, "gradients_used": False,
              "counterfactual": "Only mask input 0 -> 1; interpolation and node embedding unchanged.",
              "mask_direction": direction.tolist(),
              "mask_direction_alignment_quantiles": torch.quantile(cosine.flatten(), torch.tensor([.1, .5, .9])).tolist(),
              "available_source_fraction": float(available.float().mean()),
              "missing_query_observed_attention_before": mass(logits),
              "missing_query_observed_attention_after_key_mask_component_x1p1": mass(new_logits),
              "head": head, "selected_head_positive_mask_logit_response_fraction": float((q_mask_response > 0).float().mean()),
              "selected_head_mask_logit_response_quantiles": torch.quantile(q_mask_response, torch.tensor([.1, .5, .9])).tolist(),
              "proposed_weight_key": "cfem.fe.attn_s.attn.attn.in_proj_weight",
              "proposed_delta_norm": float(delta.norm()), "checkpoint_edited": False,
              "caution": "Observed-source attention is a mechanism diagnostic, not an accuracy improvement claim."}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("mask_direction", "patient_ids")}, indent=2))


if __name__ == "__main__":
    main()
