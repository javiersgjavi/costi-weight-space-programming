"""Check a fixed query-prior edit on broader calibration inputs, without targets."""

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
    output = Path("artifacts/audit/clinical_constant_query_prior_input_probe_cal512.json")
    if output.exists():
        raise FileExistsError(output)
    parent_path = Path("weight_space/edits/clinical_v0_affine_tail_readout.json")
    recipe_path = Path("weight_space/edits/clinical_v0_constant_mask_query_prior.json")
    source = next(Path("../weights/ctimba_mimic-iii-challenge_point/version_0").glob("*.ckpt"))
    parent, candidate = [Inference(source, 0, json.loads(path.read_text())) for path in (parent_path, recipe_path)]
    data = load_data("mimic-iii-challenge_point")
    ids = data.indices("calibration", 512)
    cfem = parent.model.cfem
    embedding = cfem.fe.node_embd_proj(parent.model.get_node_embd())
    wq, wk, _ = cfem.fe.attn_s.attn.attn.in_proj_weight.chunk(3)
    new_wq = candidate.model.cfem.fe.attn_s.attn.attn.in_proj_weight[:64]
    construction = json.loads(Path("artifacts/audit/clinical_constant_mask_query_prior_construction.json").read_text())
    query_offset = torch.tensor(construction["query_offset"])
    increments, before_mass, after_mass = [], [], []
    offset_max_error = 0.
    for start in range(0, len(ids), 32):
        batch = data.batch(ids[start:start + 32])
        values, observed = batch["x_interpolated"], batch["mask"]

        def encode(mask):
            return cfem.fe.attn_s.ln1(cfem.input_proj(torch.cat([values, mask], -1)) + embedding)

        hidden = encode(observed)
        key_difference = F.linear(encode(torch.ones_like(values)) - encode(torch.zeros_like(values)), wk[56:64])
        increments.append((key_difference * query_offset).sum(-1).flatten() / 8 ** .5)
        b, t, n, _ = hidden.shape
        old_q = F.linear(hidden, wq[56:64])
        new_q = F.linear(hidden, new_wq[56:64])
        offset_max_error = max(offset_max_error, float((new_q - old_q - query_offset).abs().max()))
        key = F.linear(hidden, wk[56:64]).reshape(b * t, n, 8)
        available = observed[..., 0].reshape(b * t, n)
        for q, destination in ((old_q, before_mass), (new_q, after_mass)):
            logits = q.reshape(b * t, n, 8) @ key.transpose(-1, -2) / 8 ** .5
            mass = (logits.softmax(-1) * available[:, None, :]).sum(-1)
            destination.append(mass[~available])
        print(json.dumps({"done": min(start + 32, len(ids)), "total": len(ids)}), flush=True)
    increments, before_mass, after_mass = [torch.cat(values) for values in (increments, before_mass, after_mass)]
    assert offset_max_error < 2e-5
    result = {"time_unix": time.time(), "source_sha256": sha256(source), "recipe_sha256": sha256(recipe_path),
              "split": "calibration", "patients": len(ids), "patient_ids": ids.tolist(),
              "mask_counterfactual_cells": len(increments), "missing_queries": len(before_mass),
              "counterfactual_increment_quantiles": torch.quantile(increments, torch.tensor([0., .01, .1, .5, .9, .99, 1.])).tolist(),
              "positive_counterfactual_increment_fraction": float((increments > 0).float().mean()),
              "nonpositive_counterfactual_increments": int((increments <= 0).sum()),
              "mean_counterfactual_increment": float(increments.mean()),
              "observed_source_mass_before": float(before_mass.mean()),
              "observed_source_mass_after": float(after_mass.mean()),
              "maximum_actual_constant_query_offset_error": offset_max_error,
              "targets_or_gradients_used": False, "candidate_revised_after_probe": False,
              "scope": "Conditioning input projection, LayerNorm and attention-head forward calculations only. No claim about full imputation accuracy."}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "patient_ids"}, indent=2))


if __name__ == "__main__":
    main()
