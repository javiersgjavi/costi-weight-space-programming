"""Observe a frozen local head program on parent features at both noise steps."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoints import sha256
from .data import load_data
from .run import Inference


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    freeze = json.loads(args.freeze.read_text())
    source, parent_path, recipe_path = map(Path, (freeze["source_checkpoint"], freeze["reference_recipe"], freeze["recipe"]))
    assert sha256(source) == freeze["source_sha256"]
    assert sha256(parent_path) == freeze["reference_recipe_sha256"]
    assert sha256(recipe_path) == freeze["recipe_sha256"]
    for path, digest in freeze["source_sha256_files"].items():
        assert sha256(path) == digest
    audit_path = Path(freeze["construction_audit"])
    assert sha256(audit_path) == freeze["construction_audit_sha256"]
    construction = json.loads(audit_path.read_text())["construction"]
    unit = construction["source_unit"]
    thresholds = construction["thresholds"]
    multipliers = construction["multipliers"]
    assert 0 <= unit < 256 and len(thresholds) == len(multipliers)
    assert thresholds == sorted(thresholds)
    parent = Inference(source, 0, json.loads(parent_path.read_text()), device="cuda", backend="triton")
    child = Inference(source, 0, json.loads(recipe_path.read_text()), device="cuda", backend="triton")
    child_head = copy.deepcopy(child.model.out_proj)
    del child
    old_head = parent.model.out_proj
    effective = old_head[1].weight.flatten() @ old_head[0].mlp[3].weight
    dataset = load_data("mimic-iii-challenge_point")
    ids = dataset.indices("calibration", args.limit)
    values, call, scoring = [[], []], 0, None
    formula_errors = [0., 0.]

    def capture(module, inputs, output):
        nonlocal call
        step = call % 2
        h = inputs[0][:, scoring]
        original = output[..., 0][:, scoring]
        z_all = old_head[0].mlp[0](h)
        raw_z = z_all[..., unit]
        z = construction.get("coordinate_scale", 1.) * raw_z
        c = construction
        programmed = sum(m * F.gelu(c["sharpness"] * (z - t))
                         for t, m in zip(thresholds, multipliers))
        programmed *= c["amplitude"] / c["sharpness"]
        removed = sum(effective[d] * F.gelu(z_all[..., d]) for d in c["reused_units"])
        actual = child_head(h)[..., 0] - original
        if actual.numel():
            error = float((actual - programmed + removed).abs().max())
            formula_errors[step] = max(formula_errors[step], error)
            assert error < 1e-4, error
        values[step].append(torch.stack([z.flatten(), actual.flatten(), programmed.flatten(), raw_z.flatten()], -1).cpu())
        call += 1

    handle = old_head.register_forward_hook(capture)
    for number, patient in enumerate(ids):
        batch = dataset.batch([patient])
        scoring = batch["eval_mask"][0, ..., 0].cuda()
        call = 0
        parent.predict(batch, [patient], 100, [80, 20.92], 20261103, 20)
        assert call == 10
        print(json.dumps({"done": number + 1, "total": len(ids)}), flush=True)
    handle.remove()
    steps = []
    for step, sigma in enumerate([80, 20.92]):
        array = torch.cat(values[step]).numpy()
        region = (array[:, 0] >= thresholds[0]) & (array[:, 0] < thresholds[-1])
        support = array[:, 0] >= thresholds[0] if construction.get("response_shape") == "capped_ramp" else region
        steps.append({"sigma": sigma, "draw_entries": len(array),
                      "local_interval_fraction": float(np.mean(region)),
                      "ideal_support_fraction": float(np.mean(support)),
                      "program_coordinate_quantiles": np.quantile(array[:, 0], [0, .01, .5, .99, 1]).tolist(),
                      "source_preactivation_quantiles": np.quantile(array[:, 3], [0, .01, .5, .99, 1]).tolist(),
                      "actual_head_delta_quantiles": np.quantile(array[:, 1], [0, .01, .5, .99, 1]).tolist(),
                      "programmed_head_delta_quantiles": np.quantile(array[:, 2], [0, .01, .5, .99, 1]).tolist(),
                      "maximum_formula_error": formula_errors[step]})
    report = {"source_sha256": sha256(source), "recipe_sha256": sha256(recipe_path),
              "parent_sha256": sha256(parent_path), "diagnostic_script_sha256": sha256(Path(__file__)),
              "construction": construction, "split": "calibration", "patient_ids": ids.tolist(),
              "patients": len(ids), "samples": 100, "seed": 20261103, "chunk": 20, "steps": steps,
              "hidden_targets_or_residuals_scored": False, "gradients_or_fitting_used": False,
              "weights_changed_by_probe": False,
              "scope": "Head-only counterfactual on actual unchanged parent features, both noise steps and all100 draws. No accuracy measurement or new candidate selection; complete candidate feedback is evaluated separately."}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "steps": steps}, indent=2))


if __name__ == "__main__":
    main()
