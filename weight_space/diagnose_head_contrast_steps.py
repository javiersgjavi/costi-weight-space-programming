"""Observe a fixed head-neuron contrast at both denoising steps; no target use."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoints import sha256
from .data import load_data
from .run import Inference


def stats(values):
    values = torch.cat(values).flatten()
    return {"n": len(values), "positive": int((values > 0).sum()),
            "positive_fraction": float((values > 0).float().mean()),
            "quantiles": torch.quantile(values, torch.tensor([0., .01, .1, .5, .9, .99, 1.])).tolist()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit", type=Path, required=True)
    parser.add_argument("--contrast-diagnostic", type=Path, required=True)
    parser.add_argument("--unit", type=int, help="Observe this preactivation above --threshold instead of the 38-198 contrast")
    parser.add_argument("--threshold", type=float, default=0.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    recipe = json.loads(args.edit.read_text())
    metadata = json.loads(args.contrast_diagnostic.with_suffix(".json").read_text())
    assert metadata["split"] == "calibration" and metadata["edit"] == recipe
    if args.unit is None:
        assert args.threshold == 0., "Threshold applies only to a specified unit"
    else:
        assert 0 <= args.unit < 256

    def contrast_of(z):
        if args.unit is not None:
            return z[..., args.unit] - args.threshold
        return z[..., 38] - z[..., 198]

    with np.load(args.contrast_diagnostic) as data:
        contrast = contrast_of(data["preactivation"])
        extra_ids = np.unique(data["window"][contrast > 0])
    dataset = load_data("mimic-iii-challenge_point")
    ids = np.unique(np.concatenate([dataset.indices("calibration", 32), extra_ids]))
    assert set(ids).issubset(set(dataset.indices("calibration")))
    checkpoint = next(Path("../weights/ctimba_mimic-iii-challenge_point/version_0").glob("*.ckpt"))
    runner = Inference(checkpoint, 0, recipe, device="cuda", backend="triton")
    captures, all_values, selected_values = [[], []], [[], []], [[], []]
    call_count, scoring_mask = 0, None

    def capture(module, inputs, output):
        nonlocal call_count
        contrast = contrast_of(inputs[0])
        captures[call_count % 2].append(contrast[:, scoring_mask].cpu())
        call_count += 1

    handle = runner.model.out_proj[0].mlp[1].register_forward_hook(capture)
    per_patient = []
    for number, patient in enumerate(ids):
        captures = [[], []]
        call_count = 0
        batch = dataset.batch([patient])
        scoring_mask = batch["eval_mask"][0, ..., 0].cuda()
        _, draws = runner.predict(batch, [patient], 100, [80, 20.92], 20261103, 20, return_samples=True)
        chosen = draws[0, ..., 0][:, scoring_mask.cpu()].median(0).indices
        row = {"id": int(patient), "scoring_entries": len(chosen), "steps": []}
        for step in (0, 1):
            values = torch.cat(captures[step])
            selected = values[chosen, torch.arange(len(chosen))]
            all_values[step].append(values.flatten())
            selected_values[step].append(selected)
            row["steps"].append({"positive_draw_entries": int((values > 0).sum()),
                                 "positive_final_median_paths": int((selected > 0).sum()),
                                 "maximum_contrast": float(values.max()) if values.numel() else None})
        per_patient.append(row)
        print(json.dumps({"done": number + 1, "total": len(ids)}), flush=True)
    handle.remove()
    result = {"source_sha256": sha256(checkpoint), "recipe_sha256": sha256(args.edit),
              "diagnostic_sha256": sha256(args.contrast_diagnostic), "split": "calibration",
              "patients": len(ids), "patient_ids": ids.tolist(), "samples": 100, "seed": 20261103,
              "contrast": "preactivation38 - preactivation198" if args.unit is None else f"preactivation{args.unit} - {args.threshold:g}",
              "selection": "32 evenly spaced calibration patients plus patients with positive contrast in the frozen median-draw diagnostic. Selection does not inspect residuals or targets; this is an enriched mechanism probe, not a prevalence estimate for the full cohort.",
              "steps": [{"sigma": sigma, "all_draw_entries": stats(all_values[step]),
                         "paths_selected_by_final_prediction_median": stats(selected_values[step])}
                        for step, sigma in enumerate([80, 20.92])],
              "per_patient": per_patient, "targets_or_gradients_used": False,
              "weights_changed": False}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("per_patient", "patient_ids")}, indent=2))


if __name__ == "__main__":
    main()
