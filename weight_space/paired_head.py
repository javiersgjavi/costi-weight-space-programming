"""Exact paired inference for edits confined to the output head.

The first noisy input and all layers before out_proj are identical. Reuse
that first backbone evaluation, then run each model's remaining denoising
steps normally. No approximation, fitted surrogate, or gradients are used.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoints import sha256
from .data import load_data
from .run import Inference, metrics


class SharedHeadPair:
    def __init__(self, checkpoint, replicate, edit, device="cuda", backend="triton"):
        if not edit.get("operations") or any(
            not op["key"].startswith("out_proj.") for op in edit["operations"]
        ):
            raise ValueError("Sharing is valid only for output-head edits")
        self.original = Inference(checkpoint, replicate, device=device, backend=backend)
        self.edited = Inference(checkpoint, replicate, edit, device=device, backend=backend)
        self.device = self.original.device
        left, right = self.original.model.state_dict(), self.edited.model.state_dict()
        for name, tensor in left.items():
            if not name.startswith("out_proj.") and not torch.equal(tensor, right[name]):
                raise ValueError(f"Backbone differs: {name}")
        for a, b in zip(self.original.model.input_sfe.support[-1],
                        self.edited.model.input_sfe.support[-1]):
            if not torch.equal(a, b):
                raise ValueError("Unregistered graph factors differ")
        if self.original.cfg != self.edited.cfg:
            raise ValueError("Model configurations differ")

    @torch.inference_mode()
    def predict(self, batch, ids, samples, sigmas, seed, chunk):
        if not sigmas:
            raise ValueError("At least one denoising step is required")
        x = batch["x"].to(self.device)
        mask = batch["mask"].to(self.device)
        cond = self.original.model.get_cond_info(
            batch["x_interpolated"].to(self.device), mask)
        noise = torch.stack([
            torch.randn((samples, len(sigmas), *x.shape[1:]),
                        generator=torch.Generator().manual_seed(seed + int(i) * 1009))
            for i in ids]).to(self.device)
        predictions = [[], []]
        cache = {}

        def capture_first_head(module, args):
            if "features" not in cache:
                cache["features"] = args[0]

        native_estimate = self.edited.model.estimate_noise

        def estimate_with_shared_first_step(noisy, conditioning, timestep):
            if cache["edited_calls"] == 0:
                cache["edited_calls"] += 1
                return self.edited.model.out_proj(cache["features"])
            cache["edited_calls"] += 1
            return native_estimate(noisy, conditioning, timestep)

        handle = self.original.model.out_proj.register_forward_pre_hook(capture_first_head)
        self.edited.model.estimate_noise = estimate_with_shared_first_step
        try:
            for start in range(0, samples, chunk):
                count = min(chunk, samples - start)
                bx, bm = x.repeat_interleave(count, 0), mask.repeat_interleave(count, 0)
                bc = tuple(v.repeat_interleave(count, 0) for v in cond)
                bn = noise[:, start:start + count].reshape(len(bx), len(sigmas), *x.shape[1:])
                cache.clear()
                cache["edited_calls"] = 0
                original = self.original._denoise_chunk(bx, bm, bc, bn, sigmas)
                edited = self.edited._denoise_chunk(bx, bm, bc, bn, sigmas)
                if cache["edited_calls"] != len(sigmas):
                    raise ValueError("Unexpected denoising call count")
                for group, prediction in zip(predictions, (original, edited)):
                    group.append(prediction.reshape(len(x), count, *x.shape[1:]))
        finally:
            self.edited.model.estimate_noise = native_estimate
            handle.remove()
            cache.clear()
        return tuple(torch.cat(group, 1).median(1).values.cpu() for group in predictions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--replicate", type=int, default=0)
    p.add_argument("--split", choices=["calibration", "confirmation", "test"], required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--chunk", type=int, default=100)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--backend", choices=["reference", "triton"], default="triton")
    p.add_argument("--sigmas", nargs="+", type=float)
    p.add_argument("--weights", type=Path, default=Path("../weights"))
    p.add_argument("--data", type=Path, default=Path("artifacts/data"))
    p.add_argument("--edit", type=Path, required=True)
    p.add_argument("--output-original", type=Path, required=True)
    p.add_argument("--output-edited", type=Path, required=True)
    args = p.parse_args()
    outputs = [args.output_original, args.output_edited]
    if outputs[0].resolve() == outputs[1].resolve():
        raise ValueError("Original and edited outputs must be distinct")
    for output in outputs:
        if output.with_suffix(".json").exists() or output.with_suffix(".npz").exists():
            raise FileExistsError(f"Results are immutable: {output}")
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_files = [Path(__file__).with_name(name) for name in (
        "paired_head.py", "run.py", "data.py", "checkpoints.py", "reference_mamba.py", "cuda_scan.py")]
    source_files += sorted(Path("src/nn").rglob("*.py"))
    hashes = {str(path): sha256(path) for path in source_files}
    checkpoints = list((args.weights / ("ctimba_" + args.dataset) /
                        f"version_{args.replicate}").glob("*.ckpt"))
    if len(checkpoints) != 1:
        raise ValueError(f"Expected one checkpoint, found {checkpoints}")
    checkpoint = checkpoints[0]
    edit = json.loads(args.edit.read_text())
    pair = SharedHeadPair(checkpoint, args.replicate, edit, args.device, args.backend)
    data = load_data(args.dataset, args.data)
    ids = data.indices(args.split, args.limit)
    sigmas = args.sigmas or ([80, .821] if args.dataset.startswith("metr") else [80, 20.92])
    predictions, rows = [[], []], [[], []]
    targets, masks = [], []
    start = time.perf_counter()
    for offset in range(0, len(ids), args.batch):
        selected = ids[offset:offset + args.batch]
        batch = data.batch(selected)
        paired = pair.predict(batch, selected, args.samples, sigmas, args.seed, args.chunk)
        progress = []
        for i, normalized in enumerate(paired):
            pred = normalized * data.scale + data.bias
            if not bool(pred.isfinite().all()):
                raise ValueError("Non-finite predictions")
            predictions[i].append(pred)
            for j, index in enumerate(selected):
                rows[i].append({"id": int(index),
                                **metrics(pred[j], batch["target"][j], batch["eval_mask"][j])})
            progress.append(metrics(pred, batch["target"], batch["eval_mask"])["mae"])
        targets.append(batch["target"])
        masks.append(batch["eval_mask"])
        print(json.dumps({"done": min(offset + args.batch, len(ids)), "total": len(ids),
                          "seconds": round(time.perf_counter() - start, 2),
                          "mae_original": progress[0], "mae_edited": progress[1]}), flush=True)
    target, mask = torch.cat(targets), torch.cat(masks)
    runtime = {"torch": torch.__version__, "backend": f"{args.device.upper()} {args.backend} BiMamba-v2 shared-head",
               "harness_version": 3, "graph_rng_draws": pair.original.graph_rng_draws,
               "source_sha256": hashes, "threads": args.threads,
               "device": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
               "sharing": "Exact common first-step backbone; all subsequent steps evaluated separately."}
    reports = []
    for i, (runner, recipe, output) in enumerate(zip(
            (pair.original, pair.edited), ({"name": "original", "operations": []}, edit), outputs)):
        pred = torch.cat(predictions[i])
        diagnostic = ((pred - target) / data.scale)[mask].double()
        report = {
            "dataset": args.dataset, "replicate": args.replicate, "split": args.split,
            "window_ids": ids.tolist(), "partition_sizes": {
                "train": len(data.train), "validation": len(data.validation), "test": len(data.test)},
            "samples": args.samples, "sigmas": sigmas, "seed": args.seed,
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
            "runtime": runtime, "edit": recipe, "changes": runner.changes,
            "seconds": time.perf_counter() - start, "seconds_scope": "Combined paired inference",
            "metrics": metrics(pred, target, mask), "per_window": rows[i],
            "normalized_residual_quantiles": {
                str(q): torch.quantile(diagnostic, q).item() for q in [.1, .25, .5, .75, .9]},
            "per_node": [metrics(pred[:, :, n], target[:, :, n], mask[:, :, n])
                         for n in range(pred.shape[2])],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(output) + ".partial.npz", prediction=pred.numpy(),
                            target=target.numpy(), mask=mask.numpy(), bias=data.bias.numpy(),
                            scale=data.scale.numpy(), window_ids=ids)
        Path(str(output) + ".partial.json").write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
    for output in outputs:
        for extension in ("npz", "json"):
            Path(str(output) + ".partial." + extension).rename(output.with_suffix("." + extension))
    print(json.dumps({"original": reports[0]["metrics"], "edited": reports[1]["metrics"],
                      "seconds": time.perf_counter() - start}, indent=2), flush=True)


if __name__ == "__main__":
    main()
