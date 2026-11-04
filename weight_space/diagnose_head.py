"""Inspect final-MLP contributions on calibration data, without derivatives."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .run import Inference
from .data import load_data
from .checkpoints import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--replicate", type=int, default=0)
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--backend", choices=["reference", "triton"], default="reference")
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--edit", type=Path)
    p.add_argument("--preactivations", action="store_true", help="Also retain the selected draw's inputs to GELU")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("Preserve existing diagnostic artifacts")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    checkpoint = next((Path("../weights") / ("ctimba_" + args.dataset) / f"version_{args.replicate}").glob("*.ckpt"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    edit = json.loads(args.edit.read_text()) if args.edit else {"name": "original", "operations": []}
    runner = Inference(checkpoint, args.replicate, edit, device=args.device, backend=args.backend)
    data = load_data(args.dataset)
    w = runner.model.out_proj[1].weight.detach()
    w2 = runner.model.out_proj[0].mlp[3].weight.detach()
    effective = (w @ w2).flatten().cpu()
    chosen = torch.arange(len(effective))
    records = []
    captures = []
    preactivation_captures = []
    preactivation_records = []
    call_count = 0
    scoring_mask = None
    def capture(module, inputs, output):
        nonlocal call_count
        # Two noise steps per chunk: retain the final-step hidden activations.
        if call_count % 2 == 1:
            captures.append(output[:, scoring_mask].detach().cpu())
            if args.preactivations:
                preactivation_captures.append(inputs[0][:, scoring_mask].detach().cpu())
        call_count += 1
    handle = runner.model.out_proj[0].mlp[1].register_forward_hook(capture)
    sigmas = [80, .821] if args.dataset.startswith("metr") else [80, 20.92]
    ids = data.indices("calibration", args.limit)
    for idx in ids:
        batch = data.batch([idx])
        captures.clear()
        preactivation_captures.clear()
        call_count = 0
        scoring_mask = batch["eval_mask"][0, ..., 0].to(args.device)
        prediction, draws = runner.predict(batch, [idx], args.samples, sigmas, 20261103,
                                           args.chunk, return_samples=True)
        mask = batch["eval_mask"]
        selected_samples = draws[0, ..., 0][:, scoring_mask.cpu()].median(0).indices
        all_features = torch.cat(captures, dim=0)
        features = all_features[selected_samples, torch.arange(len(selected_samples))]
        if args.preactivations:
            preactivation_records.append(torch.cat(preactivation_captures, dim=0)[selected_samples, torch.arange(len(selected_samples))])
        pred = prediction[mask]
        error = (prediction - (batch["target"] - data.bias) / data.scale)[mask]
        nodes = torch.arange(mask.shape[2]).expand(mask.shape[1], -1)[scoring_mask.cpu()]
        records.append((pred, error, features, nodes, torch.full_like(nodes, int(idx))))
        print("diagnosed window", int(idx), flush=True)
    handle.remove()
    pred, error, features, nodes, windows = [torch.cat([r[i] for r in records]).numpy() for i in range(5)]
    importance = np.mean(np.abs(features * effective.numpy()[None, :]), axis=0)
    active = np.argsort(importance)[-12:][::-1].copy()
    chosen = torch.from_numpy(active)
    full_features = features
    features = full_features[:, active]
    bounds = np.quantile(pred, [0, .2, .4, .6, .8, 1])
    result = {"dataset": args.dataset, "replicate": args.replicate, "split": "calibration",
              "edit": edit,
              "checkpoint_sha256": sha256(checkpoint),
              "diagnostic_script_sha256": sha256(Path(__file__)),
              "patient_ids": ids.tolist(), "evaluated_patients": len(ids),
              "patients_with_scoring_entries": int(len(np.unique(windows))),
              "seed": 20261103, "chunk": args.chunk,
              "samples": args.samples, "hidden_neurons": chosen.tolist(),
              "activation_selection": "Hidden activations of the actual median-selected draw at each scoring entry, final noise step",
              "backend": args.backend, "device": args.device,
              "effective_output_coefficients": effective[chosen].tolist(),
              "mean_absolute_output_contributions": importance[active].tolist(),
              "bins": []}
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mask = (pred >= lo) & (pred <= hi)
        result["bins"].append({"pred_normalized_range": [float(lo), float(hi)],
                               "n": int(mask.sum()), "median_residual": float(np.median(error[mask])),
                               "mean_hidden_activations": features[mask].mean(0).tolist()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"prediction": pred, "residual": error, "hidden": full_features,
              "effective": effective.numpy(), "node": nodes, "window": windows}
    if args.preactivations:
        preactivation = torch.cat(preactivation_records)
        torch.testing.assert_close(torch.nn.functional.gelu(preactivation), torch.from_numpy(full_features), rtol=2e-5, atol=2e-6)
        arrays["preactivation"] = preactivation.numpy()
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
