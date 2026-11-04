"""Inspect actual conditioning-state timescales, including data-dependent deltas."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_data
from .reference_mamba import selective_scan
from .run import Inference


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="metr-la_point")
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--edit", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/audit/metr_point_conditioning_memory_v0.json"))
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
    layer = runner.model.cfem.fe.mamba.mamba.mamba_block[2]
    records = []

    def inspect(module, args):
        hidden = args[0]
        xz = F.linear(hidden, module.in_proj.weight, module.in_proj.bias).transpose(1, 2)
        record = {}
        for direction, suffix in [("forward", ""), ("reverse", "_b")]:
            inputs = xz if not suffix else xz.flip(-1)
            conv = getattr(module, "conv1d" + suffix)
            xp = getattr(module, "x_proj" + suffix)
            dp = getattr(module, "dt_proj" + suffix)
            A = -getattr(module, "A" + suffix + "_log").exp()
            D = getattr(module, "D" + suffix)
            x, z = inputs.chunk(2, 1)
            x = F.silu(F.conv1d(x, conv.weight, conv.bias, padding=conv.weight.shape[-1]-1,
                               groups=x.shape[1])[..., :x.shape[-1]])
            projected = F.linear(x.transpose(1, 2), xp.weight)
            rank, states = dp.weight.shape[1], A.shape[1]
            raw_delta = F.linear(projected[..., :rank], dp.weight).transpose(1, 2)
            delta = F.softplus(raw_delta + dp.bias[None, :, None])
            B = projected[..., rank:rank+states].transpose(1, 2).contiguous()
            C = projected[..., -states:].transpose(1, 2).contiguous()
            recurrent = selective_scan(x, raw_delta, A, B, C, delta_bias=dp.bias)
            skip = x * D[None, :, None]
            half = np.log(2) / (delta[..., None] * -A[None, :, None, :])
            record[direction] = {
                "delta_quantiles": torch.quantile(delta.flatten(), torch.tensor([.1,.5,.9], device="cuda")).cpu().tolist(),
                "slowest_state_half_life_quantiles": torch.quantile(half.amax(-1).flatten(), torch.tensor([.1,.5,.9], device="cuda")).cpu().tolist(),
                "fraction_states_half_life_above_one": (half > 1).float().mean().item(),
                "recurrent_to_skip_mean_abs": (recurrent.abs().mean() / skip.abs().mean()).item(),
                "channel_mean_delta": delta.mean((0,2)).cpu().tolist(),
            }
        records.append(record)

    handle = layer.register_forward_pre_hook(inspect)
    for index in data.indices("calibration", args.limit):
        batch = data.batch([index])
        runner.model.get_cond_info(batch["x_interpolated"].cuda(), batch["mask"].cuda())
        print(json.dumps({"diagnosed_patient": int(index), "done": len(records), "total": args.limit}), flush=True)
    handle.remove()
    result = {"dataset": args.dataset, "replicate": args.replicate, "split": "calibration", "edit": edit,
              "patient_ids": data.indices("calibration", args.limit).tolist(),
              "module": "cfem.fe.mamba.mamba.mamba_block.2", "windows": len(records),
              "mean": {d: {k: np.mean([r[d][k] for r in records], axis=0).tolist()
                            for k in records[0][d]} for d in records[0]}, "per_window": records}
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({d: {k: v for k,v in result['mean'][d].items() if k != 'channel_mean_delta'}
                      for d in result['mean']}, indent=2))


if __name__ == "__main__":
    main()
