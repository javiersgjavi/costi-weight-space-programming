"""Observe both head paths of the actual median-selected calibration draw."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoints import sha256
from .data import load_data
from .finalize_clinical import state_hash
from .run import Inference


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--reference-result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists() and not args.output.with_suffix(".npz").exists()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    freeze = json.loads(args.freeze.read_text())
    reference = json.loads(args.reference_result.read_text())
    source, recipe_path = map(Path, (freeze["source_checkpoint"], freeze["recipe"]))
    recipe = json.loads(recipe_path.read_text())
    assert sha256(source) == freeze["source_sha256"]
    assert sha256(recipe_path) == freeze["recipe_sha256"]
    assert reference["edit"] == recipe and reference["checkpoint_sha256"] == freeze["source_sha256"]
    assert reference["dataset"] == "mimic-iii-challenge_point" and reference["replicate"] == 0
    assert reference["split"] == "calibration" and reference["samples"] == 100
    assert reference["seed"] == 20261103 and reference["sigmas"] == [80, 20.92]
    for path, digest in freeze["source_sha256_files"].items():
        assert sha256(path) == digest
    with np.load(args.reference_result.with_suffix(".npz")) as arrays:
        reference_prediction = arrays["prediction"]
        reference_mask = arrays["mask"]
        reference_ids = arrays["window_ids"]
    assert reference_ids.tolist() == reference["window_ids"] and len(reference_ids) == 512
    dataset = load_data("mimic-iii-challenge_point")
    assert np.array_equal(dataset.indices("calibration", 512), reference_ids)
    runner = Inference(source, 0, recipe, device="cuda", backend="triton")
    assert state_hash(runner.model) == freeze["edited_state_sha256"]
    effective = (runner.model.out_proj[1].weight @ runner.model.out_proj[0].mlp[3].weight).flatten().cpu()
    call, scoring = 0, None
    hidden_captures, z_captures = [[], []], [[], []]

    def capture(module, inputs, output):
        nonlocal call
        step = call % 2
        hidden_captures[step].append(output[:, scoring].cpu())
        z_captures[step].append(inputs[0][:, scoring].cpu())
        call += 1

    handle = runner.model.out_proj[0].mlp[1].register_forward_hook(capture)
    records = {key: [] for key in ("prediction", "residual", "node", "window", "median_selected_draw",
                                  "hidden", "preactivation", "hidden_step1", "preactivation_step1")}
    try:
        for number, patient in enumerate(reference_ids):
            batch = dataset.batch([patient])
            mask = batch["eval_mask"]
            assert np.array_equal(mask.numpy()[0], reference_mask[number])
            scoring = mask[0, ..., 0].cuda()
            for lists in (hidden_captures, z_captures):
                for values in lists:
                    values.clear()
            call = 0
            prediction, draws = runner.predict(batch, [patient], 100, [80, 20.92], 20261103, 20,
                                               return_samples=True)
            assert call == 10
            physical = prediction * dataset.scale + dataset.bias
            assert np.array_equal(physical.numpy()[0], reference_prediction[number]), int(patient)
            selected = draws[0, ..., 0][:, scoring.cpu()].median(0).indices
            entries = torch.arange(len(selected))
            for step, suffix in ((0, "_step1"), (1, "")):
                h = torch.cat(hidden_captures[step])[selected, entries]
                z = torch.cat(z_captures[step])[selected, entries]
                torch.testing.assert_close(F.gelu(z), h, rtol=2e-5, atol=2e-6)
                records["hidden" + suffix].append(h)
                records["preactivation" + suffix].append(z)
            nodes = torch.arange(mask.shape[2]).expand(mask.shape[1], -1)[scoring.cpu()]
            records["prediction"].append(prediction[mask])
            # The same final residual annotates both selected paths. This is
            # not a step1 prediction error or a derivative through step1.
            records["residual"].append((prediction - (batch["target"] - dataset.bias) / dataset.scale)[mask])
            records["node"].append(nodes)
            records["window"].append(torch.full_like(nodes, int(patient)))
            records["median_selected_draw"].append(selected)
            print(json.dumps({"done": number + 1, "total": len(reference_ids), "patient": int(patient),
                              "full_prediction_exact_parity": True}), flush=True)
    finally:
        handle.remove()
    arrays = {key: torch.cat(values).numpy() for key, values in records.items()}
    arrays["effective"] = effective.numpy()
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    report = {
        "dataset": "mimic-iii-challenge_point", "replicate": 0, "split": "calibration", "edit": recipe,
        "checkpoint_sha256": sha256(source), "recipe_sha256": sha256(recipe_path),
        "freeze": str(args.freeze), "freeze_sha256": sha256(args.freeze),
        "diagnostic_script_sha256": sha256(Path(__file__)),
        "reference_result": str(args.reference_result), "reference_result_sha256": sha256(args.reference_result),
        "reference_raw_sha256": sha256(args.reference_result.with_suffix(".npz")),
        "full_prediction_and_mask_exact_parity": True, "patient_ids": reference_ids.tolist(),
        "evaluated_patients": len(reference_ids), "patients_with_scoring_entries": int(len(np.unique(arrays["window"]))),
        "seed": 20261103, "chunk": 20, "samples": 100, "sigmas": [80, 20.92],
        "backend": "triton", "device": "cuda",
        "activation_selection": "Both noise-step paths of the actual final median-selected draw at each scored entry. Unchanged full model feedback, no head substitutions.",
        "array_convention": "hidden/preactivation are final-step arrays; suffix_step1 contains the corresponding first-step path of the same selected draw. Residual is always final normalized prediction minus truth.",
        "gradients_or_fitting_used": False, "weights_changed_by_probe": False, "test_targets_or_residuals_used": False,
        "scope": "Descriptive calibration features for interpreting shared weights across noise levels. Associations with the final residual do not establish a first-step causal error, and no candidate or coefficient is selected by this script."
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "scoring_entries": len(arrays["residual"]),
                      "evaluated_patients": len(reference_ids), "both_steps_recorded": True,
                      "exact_full_prediction_parity": True}), flush=True)


if __name__ == "__main__":
    main()
