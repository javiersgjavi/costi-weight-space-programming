"""Paired, gradient-free evaluation of explicit checkpoint edits."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import COSTI_ROOT
from .checkpoints import load_checkpoint, plain_metadata, sha256
from .data import load_data
from .reference_mamba import install_reference_backend


def apply_edit(model, edit):
    """Only explicit, human-readable tensor operations; no search or fitting."""
    changed = []
    state = model.state_dict()
    with torch.no_grad():
        for op in edit.get("operations", []):
            tensor = state[op["key"]]
            before = tensor.clone()
            if op["kind"] == "add":
                tensor.add_(op["value"])
            elif op["kind"] == "multiply":
                tensor.mul_(op["value"])
            elif op["kind"] == "multiply_columns":
                tensor[:, op["start"]:op["stop"]].mul_(op["value"])
            elif op["kind"] == "add_columns":
                tensor[:, op["start"]:op["stop"]].add_(op["value"])
            elif op["kind"] == "multiply_rows":
                tensor[op["start"]:op["stop"]].mul_(op["value"])
            elif op["kind"] == "multiply_rectangle":
                tensor[op["row_start"]:op["row_stop"], op["column_start"]:op["column_stop"]].mul_(op["value"])
            elif op["kind"] == "equalize_temporal_phase":
                direction = torch.as_tensor(op["direction"], dtype=tensor.dtype, device=tensor.device)
                if tensor.ndim != 4 or tensor.shape[0] != len(direction) or tensor.shape[-1] != 1:
                    raise ValueError("Incompatible temporal upsampling tensor")
                direction = direction / direction.norm()
                average = tensor[..., 0].mean(2)
                for phase in range(tensor.shape[2]):
                    delta = average - tensor[:, :, phase, 0]
                    tensor[:, :, phase, 0].add_(op["value"] * direction[:, None] * (direction @ delta)[None, :])
            elif op["kind"] == "add_values":
                values = torch.as_tensor(op["values"], dtype=tensor.dtype, device=tensor.device)
                if values.shape != tensor.shape:
                    raise ValueError("Explicit tensor edit has the wrong shape")
                tensor.add_(values)
            elif op["kind"] == "scale_effective_columns":
                # The following scalar linear layer has no intervening
                # nonlinearity. Change only each chosen neuron's effective
                # coefficient, along the downstream readout direction.
                readout = state[op["readout_key"]].reshape(-1)
                if tensor.ndim != 2 or tensor.shape[0] != len(readout):
                    raise ValueError("Incompatible composed linear layers")
                columns = op["columns"]
                old_columns = tensor[:, columns]
                coefficients = readout @ old_columns
                delta = readout[:, None] * ((op["value"] - 1) * coefficients / readout.square().sum())
                tensor[:, columns] = old_columns + delta
            else:
                raise ValueError(f"Unknown operation {op['kind']}")
            if not tensor.isfinite().all():
                raise ValueError("Edit generated non-finite weights")
            changed.append({**op, "changed_values": int((tensor != before).sum()),
                            "delta_norm": (tensor - before).norm().item()})
    return changed


class Inference:
    def __init__(self, checkpoint, replicate, edit=None, device="cpu", backend="reference", execution="eager"):
        if backend == "triton" and device != "cuda":
            raise ValueError("Triton backend requires CUDA")
        if execution == "graph" and (device != "cuda" or backend != "triton"):
            raise ValueError("CUDA graph execution requires the fused CUDA backend")
        install_reference_backend(backend)
        self.backend = backend
        self.execution = execution
        self.graph_cache = {}
        from src.nn.unet import UnetContinuous
        record = load_checkpoint(checkpoint)
        cfg = plain_metadata(record["hyper_parameters"]["model_kwargs"])
        self.cfg = cfg
        self.device = torch.device(device)
        # Same model-initialization seed as AverageExperiment / KTestAverageExperiment.
        torch.manual_seed(42 + replicate)
        self.graph_rng_draws = 0
        if "historical" in cfg["missing_pattern"].values():
            # prepare_data iterates one unshuffled historical-pattern DataLoader
            # before constructing the model. Its base-seed draw advances the RNG.
            torch.empty((), dtype=torch.int64).random_()
            self.graph_rng_draws = 1
        self.model = UnetContinuous(
            n_nodes=cfg["dataset"]["num_nodes"],
            t_steps=cfg["dataset"]["time_steps"],
            adj=record["hyper_parameters"]["adj"], **cfg["model_config"])
        if "weight_space_graph_factors" in record:
            self.model.input_sfe.support[-1] = [p.detach().clone() for p in record["weight_space_graph_factors"]]
        state = {k.removeprefix("model."): v for k, v in record["state_dict"].items()
                 if k.startswith("model.")}
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False)
        self.changes = apply_edit(self.model, edit or {})
        self.model.to(self.device)
        # The upstream graph factors are unregistered; freeze them too.
        for part in self.model.input_sfe.support[-1]:
            part.requires_grad_(False)
        self.model.input_sfe.support[-1] = [p.to(self.device) for p in self.model.input_sfe.support[-1]]
        for i in range(len(self.model.input_sfe.support) - 1):
            self.model.input_sfe.support[i] = self.model.input_sfe.support[i].to(self.device)
        scheduler = cfg["scheduler_kwargs"]
        self.sigma_min, self.sigma_data = scheduler["sigma_min"], scheduler["sigma_data"]

    def _denoise_chunk(self, x, mask, cond, noise, sigmas):
        pred = x
        for step, sigma in enumerate(sigmas):
            sd = self.sigma_data
            cskip = sd**2 / ((sigma - self.sigma_min)**2 + sd**2)
            cout = sd * (sigma - self.sigma_min) / (sd**2 + sigma**2)**.5
            cin = 1 / (sigma**2 + sd**2)**.5
            magnitude = sigma if step == 0 else (sigma**2 - self.sigma_min**2)**.5
            noisy = pred + noise[:, step] * magnitude
            t = torch.full((len(x),), np.log(sigma) / 4, dtype=x.dtype, device=self.device)
            y = self.model.estimate_noise(cin * noisy, cond, t)
            pred = torch.where(mask, x, cskip * noisy + cout * y)
        return pred

    def _graph_chunk(self, x, mask, cond, noise, sigmas):
        key = (tuple(x.shape), tuple(sigmas))
        if key not in self.graph_cache:
            sx, sm, sc, sn = x.clone(), mask.clone(), tuple(v.clone() for v in cond), noise.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._denoise_chunk(sx, sm, sc, sn, sigmas)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._denoise_chunk(sx, sm, sc, sn, sigmas)
            self.graph_cache[key] = (graph, sx, sm, sc, sn, out)
        graph, sx, sm, sc, sn, out = self.graph_cache[key]
        sx.copy_(x); sm.copy_(mask); sn.copy_(noise)
        for dst, src in zip(sc, cond):
            dst.copy_(src)
        graph.replay()
        return out.clone()

    @torch.inference_mode()
    def predict(self, batch, ids, samples, sigmas, seed, chunk, return_samples=False):
        install_reference_backend(self.backend)
        x = batch["x"].to(self.device)
        mask = batch["mask"].to(self.device)
        itp = batch["x_interpolated"].to(self.device)
        cond = self.model.get_cond_info(itp, mask)
        # Generate sample-major noise once, independent of chunk/batch size.
        noise = torch.stack([
            torch.randn((samples, len(sigmas), *x.shape[1:]),
                        generator=torch.Generator().manual_seed(seed + int(i) * 1009))
            for i in ids]).to(self.device)
        predictions = []
        for start in range(0, samples, chunk):
            count = min(chunk, samples - start)
            bx, bm = x.repeat_interleave(count, 0), mask.repeat_interleave(count, 0)
            bc = tuple(v.repeat_interleave(count, 0) for v in cond)
            bn = noise[:, start:start + count].reshape(len(bx), len(sigmas), *x.shape[1:])
            method = self._graph_chunk if self.execution == "graph" else self._denoise_chunk
            pred = method(bx, bm, bc, bn, sigmas)
            predictions.append(pred.reshape(len(x), count, *x.shape[1:]))
        # torch.median selects the lower central sample when samples is even,
        # exactly as the released evaluation code does.
        draws = torch.cat(predictions, dim=1)
        median = draws.median(dim=1).values.cpu()
        return (median, draws.cpu()) if return_samples else median


def metrics(pred, target, mask):
    error = (pred - target)[mask].double()
    truth = target[mask].double()
    if not len(error):
        return {"n": 0, "mae": None, "mse": None, "mre": None,
                "mean_error": None, "negative_prediction_fraction": None}
    return {"n": len(error), "mae": error.abs().mean().item(),
            "mse": error.square().mean().item(),
            "mre": (error.abs().sum() / truth.abs().sum()).item() if truth.abs().sum() else None,
            "mean_error": error.mean().item(),
            "negative_prediction_fraction": (pred[mask] < 0).double().mean().item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--replicate", type=int, default=0)
    p.add_argument("--split", choices=["calibration", "confirmation", "test"], default="calibration")
    p.add_argument("--limit", type=int)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--sigmas", nargs="+", type=float)
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--backend", choices=["reference", "triton"], default="reference")
    p.add_argument("--execution", choices=["eager", "graph"], default="eager")
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--weights", type=Path, default=Path("../weights"))
    p.add_argument("--data", type=Path, default=Path("artifacts/data"))
    p.add_argument("--edit", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.with_suffix(".json").exists():
        raise FileExistsError("Use a new result path; previous measurements are immutable")
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_files = [Path(__file__).with_name(name) for name in
                    ("run.py", "data.py", "checkpoints.py", "reference_mamba.py", "cuda_scan.py")]
    source_files += sorted((COSTI_ROOT / "src/nn").rglob("*.py"))
    source_hashes = {str(path): sha256(path) for path in source_files}
    edit = json.loads(args.edit.read_text()) if args.edit else {"name": "original", "operations": []}
    paths = list((args.weights / ("ctimba_" + args.dataset) / f"version_{args.replicate}").glob("*.ckpt"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one checkpoint, found {paths}")
    checkpoint = paths[0]
    model = Inference(checkpoint, args.replicate, edit, device=args.device,
                      backend=args.backend, execution=args.execution)
    data = load_data(args.dataset, args.data)
    sigmas = args.sigmas or ([80, .821] if args.dataset.startswith("metr") else [80, 20.92])
    ids = data.indices(args.split, args.limit)
    all_pred, all_target, all_mask, all_norm_error = [], [], [], []
    per_window = []
    start = time.perf_counter()
    for offset in range(0, len(ids), args.batch):
        selected = ids[offset:offset + args.batch]
        batch = data.batch(selected)
        normalized = model.predict(batch, selected, args.samples, sigmas, args.seed, args.chunk)
        pred = normalized * data.scale + data.bias
        mask, target = batch["eval_mask"], batch["target"]
        if not bool(pred.isfinite().all()):
            raise ValueError("Non-finite predictions")
        for j, i in enumerate(selected):
            per_window.append({"id": int(i), **metrics(pred[j], target[j], mask[j])})
        all_pred.append(pred); all_target.append(target); all_mask.append(mask)
        all_norm_error.append((pred - target) / data.scale)
        print(json.dumps({"done": min(offset + args.batch, len(ids)), "total": len(ids),
                          "seconds": round(time.perf_counter() - start, 2),
                          "batch_mae": metrics(pred, target, mask)["mae"]}), flush=True)
    pred, target, mask = map(torch.cat, (all_pred, all_target, all_mask))
    norm_error = torch.cat(all_norm_error)
    diagnostic = norm_error[mask].double()
    result = {
        "dataset": args.dataset, "replicate": args.replicate,
        "split": args.split, "window_ids": ids.tolist(),
        "partition_sizes": {"train": len(data.train), "validation": len(data.validation), "test": len(data.test)},
        "samples": args.samples, "sigmas": sigmas, "seed": args.seed,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
        "runtime": {"torch": torch.__version__, "backend": f"{args.device.upper()} {args.backend}{' graph' if args.execution == 'graph' else ''} BiMamba-v2", "threads": args.threads,
                    "device": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
                    "source_sha256": source_hashes,
                    "harness_version": 2, "graph_rng_draws": model.graph_rng_draws},
        "edit": edit, "changes": model.changes,
        "seconds": time.perf_counter() - start,
        "metrics": metrics(pred, target, mask),
        "normalized_residual_quantiles": {str(q): torch.quantile(diagnostic, q).item() for q in [.1, .25, .5, .75, .9]},
        "per_window": per_window,
        "per_node": [metrics(pred[:, :, n], target[:, :, n], mask[:, :, n])
                     for n in range(pred.shape[2])],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), prediction=pred.numpy(),
                        target=target.numpy(), mask=mask.numpy(),
                        bias=data.bias.numpy(), scale=data.scale.numpy(), window_ids=ids)
    print(json.dumps({"output": str(args.output), "metrics": result["metrics"],
                      "normalized_residual_quantiles": result["normalized_residual_quantiles"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
