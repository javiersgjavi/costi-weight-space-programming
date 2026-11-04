"""Inspect all released tensors; no optimization or gradients are used."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_checkpoint, plain_metadata, sha256


@torch.inference_mode()
def audit(path):
    checkpoint = load_checkpoint(path)
    state = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()
             if k.startswith("model.")}
    hp = checkpoint["hyper_parameters"]
    tensors = {}
    mamba = {}
    for name, tensor in state.items():
        if not tensor.is_floating_point():
            continue
        v = tensor.float()
        tensors[name] = dict(shape=list(v.shape), n=v.numel(),
                             finite=bool(v.isfinite().all()),
                             mean=v.mean().item(), std=v.std().item() if v.numel() > 1 else 0,
                             minimum=v.min().item(), maximum=v.max().item(),
                             norm=v.norm().item())
        if name.endswith("dt_proj.bias"):
            prefix = name.removesuffix("dt_proj.bias")
            rates_f = torch.nn.functional.softplus(v)[:, None] * state[prefix + "A_log"].exp()
            rates_b = torch.nn.functional.softplus(state[prefix + "dt_proj_b.bias"])[:, None] * state[prefix + "A_b_log"].exp()
            mamba[prefix] = {
                "forward_nominal_half_life_median": (np.log(2) / rates_f).median().item(),
                "backward_nominal_half_life_median": (np.log(2) / rates_b).median().item(),
                "backward_dt_bias_mean": state[prefix + "dt_proj_b.bias"].mean().item(),
                "forward_dt_bias_mean": v.mean().item(),
            }
    return {
        "checkpoint": str(path.resolve()), "sha256": sha256(path),
        "epoch": checkpoint["epoch"], "num_tensors": len(state),
        "num_values": sum(v.numel() for v in state.values()),
        "config": plain_metadata(hp["model_kwargs"] | {"hist_patterns": None}),
        "saved_adjacency_shape": list(hp["adj"].shape),
        "adaptive_graph_keys": [k for k in state if "nodevec" in k or "support" in k],
        "mamba": mamba, "tensors": tensors,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=Path("../weights"))
    p.add_argument("--output", type=Path, default=Path("artifacts/audit"))
    args = p.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in sorted(args.weights.glob("*/*/*.ckpt")):
        result = audit(path)
        name = path.parent.parent.name + "_" + path.parent.name
        (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        row = {k: result[k] for k in ("checkpoint", "sha256", "num_tensors", "num_values", "adaptive_graph_keys")}
        row["all_finite"] = all(v["finite"] for v in result["tensors"].values())
        row["forward_half_life"] = np.median([v["forward_nominal_half_life_median"] for v in result["mamba"].values()])
        row["backward_half_life"] = np.median([v["backward_nominal_half_life_median"] for v in result["mamba"].values()])
        summary.append(row)
        print(name, row["num_values"], "finite", row["all_finite"],
              "half-life", round(row["forward_half_life"], 3), round(row["backward_half_life"], 3), flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
