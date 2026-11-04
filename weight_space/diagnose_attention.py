"""Read conditioning attention behavior on calibration inputs only."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_data
from .run import Inference


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="metr-la_point")
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--edit", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/audit/metr_point_conditioning_attention_v0.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = next((Path("../weights") / ("ctimba_" + args.dataset) / f"version_{args.replicate}").glob("*.ckpt"))
    edit = json.loads(args.edit.read_text()) if args.edit else {"name": "original", "operations": []}
    runner = Inference(path, args.replicate, edit, device="cuda", backend="triton")
    data = load_data(args.dataset)
    layer = runner.model.cfem.fe.attn_s.attn.attn
    records = []
    conditioned = None

    def inspect(module, args, kwargs):
        x = kwargs["query"]
        assert module.in_proj_bias is None, "Include attention biases before analyzing this layer"
        wq, wk, wv = module.in_proj_weight.chunk(3)
        q, k = [F.linear(x, w).reshape(len(x), x.shape[1], 8, -1).transpose(1, 2)
                for w in (wq, wk)]
        attention = (q @ k.transpose(-1, -2) / q.shape[-1]**.5).softmax(-1)
        entropy = -(attention * attention.clamp_min(1e-30).log()).sum(-1)
        observed_mass = (attention * conditioned[:, None, None, :]).sum(-1)
        unknown = ~conditioned[:, None, :].expand(-1, 8, -1)
        records.append({"normalized_entropy": (entropy.mean((0, 2)) / np.log(x.shape[1])).cpu().tolist(),
                        "mean_max_attention": attention.amax(-1).mean((0, 2)).cpu().tolist(),
                        "missing_query_observed_mass": [observed_mass[:, h][unknown[:, h]].mean().item()
                                                        for h in range(8)],
                        "available_source_fraction": conditioned.float().mean().item()})

    handle = layer.register_forward_pre_hook(inspect, with_kwargs=True)
    for index in data.indices("calibration", args.limit):
        batch = data.batch([index])
        conditioned = batch["mask"][0, ..., 0].cuda()
        runner.model.get_cond_info(batch["x_interpolated"].cuda(), batch["mask"].cuda())
        print(json.dumps({"diagnosed_patient": int(index), "done": len(records), "total": args.limit}), flush=True)
    handle.remove()
    result = {"dataset": args.dataset, "replicate": args.replicate, "split": "calibration", "edit": edit,
              "patient_ids": data.indices("calibration", args.limit).tolist(),
              "module": "cfem.fe.attn_s.attn.attn", "windows": len(records),
              "mean": {key: np.mean([r[key] for r in records], axis=0).tolist() for key in records[0]},
              "per_window": records}
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
