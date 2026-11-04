"""Compile one localized repair of the positive readout path, without fitting.

The existing unit 198 underestimates in its positive intermediate range
and overestimates above activation 2. A compact, asymmetric triangular
correction acts inside that range and returns to zero in the high tail.
"""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoints import sha256
from .run import Inference


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    parent = Path("weight_space/edits/physionet/clinical_v0_affine_tail_readout.json")
    checkpoint = next(Path("../weights/ctimba_mimic-iii-challenge_point/version_0").glob("*.ckpt"))
    output = Path("weight_space/edits/physionet/clinical_v0_local_positive_repair.json")
    audit_path = Path("artifacts/audit/clinical_local_positive_repair_construction.json")
    if output.exists() or audit_path.exists():
        raise FileExistsError("Hypotheses are immutable")
    assert not Path("artifacts/results/clinical_point_v0_targeted_test_full_seed1_paired.json").exists()
    evidence = Path("artifacts/audit/clinical_targeted_positive_path_residuals.json")
    diagnosis = json.loads(evidence.read_text())
    rows = [r for r in diagnosis["rows"] if r["unit"] == 198]
    assert all(r["median_normalized_error"] < 0 and r["patients"] >= 80 for r in rows[:3])
    assert rows[3]["median_normalized_error"] > 0
    recipe = json.loads(parent.read_text())
    original = Inference(checkpoint, 0)
    selected = Inference(checkpoint, 0, recipe)
    state = selected.model.state_dict()
    prefix = "out_proj.0.mlp."
    readout = state["out_proj.1.weight"].reshape(-1)
    effective = readout @ state[prefix + "3.weight"]
    # One fixed conservative slope, chosen before any candidate evaluation.
    # No numerical fit, optimizer, gradient or parameter sweep is performed.
    slope = .04 * float(effective[198])
    sharpness = 16.0
    destinations = [(246, 0.0, 1.0), (226, 1.5, -4.0), (179, 2.0, 3.0)]
    deltas = {k: torch.zeros_like(state[k]) for k in
              (prefix + "0.weight", prefix + "0.bias", prefix + "3.weight")}
    with np.load("artifacts/audit/clinical_targeted_head_cal128_median100_v0.npz") as diagnostic:
        hidden = torch.from_numpy(diagnostic["hidden"])
    removed_contributions = {}
    for unit, threshold, multiplier in destinations:
        removed_contributions[str(unit)] = float((hidden[:, unit] * effective[unit]).abs().max())
        assert removed_contributions[str(unit)] < 1e-6
        deltas[prefix + "0.weight"][unit] = (
            sharpness * state[prefix + "0.weight"][198] - state[prefix + "0.weight"][unit])
        deltas[prefix + "0.bias"][unit] = (
            sharpness * (state[prefix + "0.bias"][198] - threshold) - state[prefix + "0.bias"][unit])
        desired = multiplier * slope / sharpness
        deltas[prefix + "3.weight"][:, unit] = (
            readout * ((desired - effective[unit]) / readout.square().sum()))
    recipe = copy.deepcopy(recipe)
    recipe["name"] = "clinical_v0_local_positive_repair"
    recipe["rationale"] += (
        " Exact median-draw activations of the edited model show unit 198 still underestimating "
        "in its positive intermediate range (median normalized errors -0.0128, -0.0327, -0.0492 "
        "in activation bins 0-0.5, 0.5-1, 1-2), across 93, 81, 96 calibration patients. Above 2 "
        "the median error reverses sign to +0.0223. Add a localized triangular correction in "
        "preactivation z198: k*(GELU(16*z)-4*GELU(16*(z-1.5))+3*GELU(16*(z-2)))/16, "
        "k=0.04*the current effective coefficient. It rises toward the strongest intermediate "
        "underestimation and switches off before the high tail. A single fixed 4% slope is used; "
        "no fitting or sweep. Reuse three existing neurons with negligible effective contributions. "
        "Unit 38 is unchanged because its remaining residual signs are inconsistent. "
        "The running benchmark candidate remains unchanged; no clinical test result informed this proposal.")
    recipe["parent_recipe_sha256"] = sha256(parent)
    recipe["construction"] = {
        "time_unix": time.time(), "source_unit": 198,
        "reused_units": [r[0] for r in destinations], "breakpoints": [0, 1.5, 2],
        "multipliers": [1, -4, 3], "sharpness": sharpness,
        "fixed_relative_slope": .04, "output_slope": slope,
        "nominal_peak_output_correction": 1.5 * slope,
        "calibration_diagnostic": str(evidence), "calibration_diagnostic_sha256": sha256(evidence),
        "clinical_test_metrics_seen": False,
    }
    recipe["operations"] += [{"key": key, "kind": "add_values", "values": value.tolist()}
                              for key, value in deltas.items()]
    candidate = Inference(checkpoint, 0, recipe)
    old_head, new_head = copy.deepcopy(selected.model.out_proj).double(), copy.deepcopy(candidate.model.out_proj).double()
    generator = torch.Generator().manual_seed(20261103)
    features = torch.randn(512, 64, generator=generator, dtype=torch.float64)
    z = old_head[0].mlp[0](features)
    programmed = slope * sum(multiplier * F.gelu(sharpness * (z[:, 198] - threshold))
                             for _, threshold, multiplier in destinations) / sharpness
    old_effective = old_head[1].weight.reshape(-1) @ old_head[0].mlp[3].weight
    removed = sum(old_effective[unit] * F.gelu(z[:, unit]) for unit, _, _ in destinations)
    measured = (new_head(features) - old_head(features)).reshape(-1)
    discrepancy = float((measured - programmed + removed).abs().max())
    assert discrepancy < 1e-5, discrepancy
    grid = torch.linspace(-10, 10, 20001, dtype=torch.float64)
    response = slope * sum(multiplier * F.gelu(sharpness * (grid - threshold))
                           for _, threshold, multiplier in destinations) / sharpness
    changes = {k: int((v != original.model.state_dict()[k]).sum())
               for k, v in candidate.model.state_dict().items()
               if not torch.equal(v, original.model.state_dict()[k])}
    audit = {"source_sha256": sha256(checkpoint), "parent_sha256": sha256(parent),
             "construction": recipe["construction"], "maximum_formula_error": discrepancy,
             "response_range_on_grid": [float(response.min()), float(response.max())],
             "removed_units_maximum_calibration_contributions": removed_contributions,
             "changed_values_by_tensor": changes, "total_changed_values": sum(changes.values()),
             "gradients_disabled": not torch.is_grad_enabled()}
    output.write_text(json.dumps(recipe, indent=2) + "\n")
    audit["recipe_sha256"] = sha256(output)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
