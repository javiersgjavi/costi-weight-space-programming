"""Inspect a fixed noise-only template against actual calibration head features.

This runs the learned time branch on the two prescribed noise levels. It does
not fit a classifier, read errors, change weights, or differentiate anything.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoints import sha256
from .finalize_clinical import state_hash
from .run import Inference


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4)
    freeze = json.loads(args.freeze.read_text())
    meta = json.loads(args.diagnostic.with_suffix(".json").read_text())
    assert meta["split"] == "calibration" and meta["full_prediction_and_mask_exact_parity"]
    assert meta["sigmas"] == [80, 20.92] and meta["samples"] == 100
    assert meta["recipe_sha256"] == freeze["recipe_sha256"]
    assert sha256(freeze["recipe"]) == freeze["recipe_sha256"]
    assert sha256(freeze["source_checkpoint"]) == freeze["source_sha256"]
    for path, digest in freeze["source_sha256_files"].items():
        assert sha256(path) == digest
    recipe = json.loads(Path(freeze["recipe"]).read_text())
    runner = Inference(Path(freeze["source_checkpoint"]), 0, recipe)
    assert state_hash(runner.model) == freeze["edited_state_sha256"]
    model = runner.model
    sigmas = torch.tensor([80., 20.92])
    noise_coordinates = sigmas.log() / 4
    embeddings = model.t_embd(noise_coordinates)
    projected = model.up_t.t_emb_linear(embeddings)
    # With spatially constant inputs this is exactly the same GroupNorm
    # operation as broadcasting the time branch across the entire window.
    templates = model.up_t.group_norm(projected[:, :, None, None])[:, :, 0, 0].double()
    direction = templates[1] - templates[0]
    norm = float(direction.norm())
    assert norm > 0
    direction /= norm
    template_scores = templates @ direction
    threshold = float(template_scores.mean())
    assert float(template_scores[1]) > float(template_scores[0])
    affine = model.out_proj[0].mlp[0]
    weight, bias = affine.weight.double(), affine.bias.double()
    singular_values = torch.linalg.svdvals(weight)
    assert int(torch.linalg.matrix_rank(weight)) == 64
    inverse = torch.linalg.pinv(weight)
    with np.load(args.diagnostic) as data:
        # Only hidden affine outputs and identities are read. Inverting the
        # known weight matrix recovers features; no targets label this solve.
        zs = [torch.from_numpy(data[key]).double()
              for key in ("preactivation_step1", "preactivation")]
        windows = data["window"]
        nodes = data["node"]
    ids = np.unique(windows)
    scores, steps = [], []
    for i, z in enumerate(zs):
        features = (z - bias) @ inverse.T
        reconstruction = features @ weight.T + bias
        residual = (reconstruction - z).abs()
        assert float(residual.max()) < 1e-3
        score = (features @ direction).numpy()
        scores.append(score)
        patient_means = np.array([score[windows == patient].mean() for patient in ids])
        steps.append({
            "sigma": float(sigmas[i]), "entries": len(score), "patients": len(ids),
            "affine_equation_reconstruction_max_error": float(residual.max()),
            "affine_equation_reconstruction_rms_error": float(residual.square().mean().sqrt()),
            "score_quantiles": np.quantile(score, [0, .01, .1, .5, .9, .99, 1]).tolist(),
            "fraction_above_fixed_template_midpoint": float((score > threshold).mean()),
            "patient_mean_score_quantiles": np.quantile(patient_means, [0, .1, .5, .9, 1]).tolist(),
            "patient_mean_fraction_above_fixed_template_midpoint": float((patient_means > threshold).mean()),
            "by_variable": [{"index": int(node), "entries": int((nodes == node).sum()),
                             "score_mean": float(score[nodes == node].mean()),
                             "score_median": float(np.median(score[nodes == node])),
                             "fraction_above_threshold": float((score[nodes == node] > threshold).mean())}
                            for node in np.unique(nodes)],
        })
    balanced = .5 * ((scores[0] <= threshold).mean() + (scores[1] > threshold).mean())
    paired_shift = scores[1] - scores[0]
    report = {
        "freeze": str(args.freeze), "freeze_sha256": sha256(args.freeze),
        "diagnostic": str(args.diagnostic), "diagnostic_sha256": sha256(args.diagnostic),
        "script_sha256": sha256(Path(__file__)), "split": "calibration", "sigmas": meta["sigmas"],
        "recipe": recipe["name"], "source_sha256": freeze["source_sha256"],
        "time_input_coordinates": noise_coordinates.tolist(),
        "learned_time_embedding_distance": float((embeddings[1] - embeddings[0]).norm()),
        "last_time_projection_distance": float((projected[1] - projected[0]).norm()),
        "normalized_noise_only_template_distance": norm,
        "fixed_template_scores": template_scores.tolist(), "fixed_template_midpoint": threshold,
        "template_direction": direction.tolist(),
        "head_affine_weight_rank": 64,
        "head_affine_weight_condition_number": float(singular_values[0] / singular_values[-1]),
        "steps": steps,
        "balanced_step_identification_fraction_at_fixed_threshold": float(balanced),
        "paired_selected_draw_score_shift_quantiles": np.quantile(paired_shift, [0, .01, .1, .5, .9, .99, 1]).tolist(),
        "fraction_of_entries_with_positive_paired_shift": float((paired_shift > 0).mean()),
        "weights_changed": False, "targets_or_errors_read": False,
        "gradients_training_or_classifier_fitting_used": False,
        "interpretation": (
            "The separator is fixed entirely by two noise-only forward templates from existing weights. "
            "Actual head features contain content and GroupNorm coupling; their affine reconstruction "
            "has the recorded numerical error. This evaluates this particular analytic template, "
            "not the best possible separator and not a proof about all noise-conditioned programs. "
            "These are the two paths of each final median-selected draw, not a balanced sample of all draws. "
            "No threshold, direction, or coefficient is fitted to these features or to imputation losses."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "template_distance": norm,
                      "threshold": threshold, "balanced_step_identification": float(balanced),
                      "steps": [{k: v for k, v in s.items() if k != "by_variable"} for s in steps]}, indent=2))


if __name__ == "__main__":
    main()
