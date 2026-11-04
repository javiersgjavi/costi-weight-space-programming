"""Compile one explicit, bounded repair of the head's negative readout path.

There is no fitting, parameter search, differentiation, or training here.
The correction undoes the earlier 0.5% gain expansion of unit 20 only in
its strongly active tail, using two nearly disconnected existing neurons.
"""

import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .checkpoints import sha256
from .run import Inference


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    checkpoint = next(Path("../weights/ctimba_mimic-iii-challenge_point/version_0").glob("*.ckpt"))
    parent = Path("weight_space/edits/clinical_v0_affine_tail_readout.json")
    output = Path("weight_space/edits/clinical_v0_bounded_negative_tail.json")
    audit_path = Path("artifacts/audit/clinical_bounded_negative_tail_construction.json")
    if output.exists() or audit_path.exists():
        raise FileExistsError("The explicit construction is immutable")
    original = Inference(checkpoint, 0)
    edited = Inference(checkpoint, 0, json.loads(parent.read_text()))
    state = edited.model.state_dict()
    original_state = original.model.state_dict()
    prefix = "out_proj.0.mlp."
    readout = state["out_proj.1.weight"].reshape(-1)
    effective = readout @ state[prefix + "3.weight"]
    original_effective = (original_state["out_proj.1.weight"].reshape(-1)
                          @ original_state[prefix + "3.weight"])
    # Undo only the added gain, not the trained negative coefficient.
    correction = -.005 * float(original_effective[20])
    assert correction > 0
    sharpness = 16.0
    destinations = [(246, 2.0, 1.0), (226, 4.0, -1.0)]
    deltas = {key: torch.zeros_like(state[key]) for key in
              (prefix + "0.weight", prefix + "0.bias", prefix + "3.weight")}
    for unit, threshold, sign in destinations:
        deltas[prefix + "0.weight"][unit] = (
            sharpness * state[prefix + "0.weight"][20] - state[prefix + "0.weight"][unit])
        deltas[prefix + "0.bias"][unit] = (
            sharpness * (state[prefix + "0.bias"][20] - threshold) - state[prefix + "0.bias"][unit])
        target_coefficient = sign * correction / sharpness
        deltas[prefix + "3.weight"][:, unit] = (
            readout * ((target_coefficient - effective[unit]) / readout.square().sum()))
    recipe = copy.deepcopy(json.loads(parent.read_text()))
    recipe["name"] = "clinical_v0_bounded_negative_tail"
    recipe["rationale"] += (
        " A second, bounded path partly reverses the affine edit's extra expansion of negative unit 20 "
        "only above preactivation 2, capped beyond 4. Calibration residuals remain positive at "
        "activations 0-2 but turn negative above 2, so a global gain reversal would be inappropriate. "
        "Units 246 and 226 have negligible original effective coefficients and are repurposed to "
        "implement c*(GELU(16*(z20-2))-GELU(16*(z20-4)))/16, c=-0.005*original_coefficient20. "
        "The coefficient follows algebraically from the earlier fixed 0.5% gain change; no coefficient "
        "is fitted to targets. This is one specified hypothesis, not a parameter sweep.")
    recipe["parent_recipe_sha256"] = sha256(parent)
    recipe["construction"] = {
        "source_unit": 20, "reused_units": [246, 226], "thresholds": [2, 4],
        "sharpness": sharpness, "original_source_coefficient": float(original_effective[20]),
        "correction_slope": correction, "approximate_output_cap": 2 * correction,
        "calibration_evidence": "clinical_head_cal128_median100_v0.npz and clinical_point_v0_affine_tail_cal128x100_seed1.npz",
        "clinical_test_seen": False,
    }
    recipe["operations"] += [
        {"kind": "add_values", "key": key, "values": delta.tolist()}
        for key, delta in deltas.items()]
    candidate = Inference(checkpoint, 0, recipe)
    # Verify the compiled equation on deterministic arbitrary features. These
    # probes check algebra only and do not select or fit any coefficient.
    generator = torch.Generator().manual_seed(20261103)
    features = torch.randn(512, 64, generator=generator, dtype=torch.float64)
    old_head = copy.deepcopy(edited.model.out_proj).double()
    new_head = copy.deepcopy(candidate.model.out_proj).double()
    z = old_head[0].mlp[0](features)
    c = float(correction)
    programmed = c * (F.gelu(sharpness * (z[:, 20] - 2))
                      - F.gelu(sharpness * (z[:, 20] - 4))) / sharpness
    effective_double = old_head[1].weight.reshape(-1) @ old_head[0].mlp[3].weight
    removed = sum(effective_double[u] * F.gelu(z[:, u]) for u, _, _ in destinations)
    measured = (new_head(features) - old_head(features)).reshape(-1)
    maximum_error = float((measured - (programmed - removed)).abs().max())
    # Stored float32 deltas introduce rounding before the float64 check.
    assert maximum_error < 5e-6, maximum_error
    changes = {key: int((value != original_state[key]).sum())
               for key, value in candidate.model.state_dict().items()
               if not torch.equal(value, original_state[key])}
    audit = {"checkpoint_sha256": sha256(checkpoint), "parent_sha256": sha256(parent),
             "construction": recipe["construction"], "changed_values": changes,
             "total_changed_values": sum(changes.values()),
             "maximum_equation_error_after_float32_storage": maximum_error,
             "gradients_disabled": not torch.is_grad_enabled(),
             "all_parameters_frozen": all(not p.requires_grad for p in candidate.model.parameters())}
    output.write_text(json.dumps(recipe, indent=2) + "\n")
    audit["recipe_sha256"] = sha256(output)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
