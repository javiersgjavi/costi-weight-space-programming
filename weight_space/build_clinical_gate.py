"""Program a conservative shape change to a negative-only output gate."""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoints import sha256
from .run import Inference


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    output = Path("weight_space/edits/clinical_v0_head47_sharp101.json")
    audit_path = Path("artifacts/audit/clinical_head47_sharp101_construction.json")
    if output.exists() or audit_path.exists():
        raise FileExistsError("Preserve existing candidates")
    source = next(Path("../weights/ctimba_mimic-iii-challenge_point/version_0").glob("*.ckpt"))
    parent = json.loads(Path("weight_space/edits/clinical_v0_affine_tail_readout.json").read_text())
    with np.load("artifacts/audit/clinical_targeted_head_cal128_median100_v0.npz") as d:
        h, coefficient = d["hidden"][:, 47], float(d["effective"][47])
        active = h < -1e-3
        diagnostic = {"unit": 47, "coefficient": coefficient, "strictly_positive_cases": int((h > 0).sum()),
                      "negative_tail_cases": int(active.sum()), "negative_tail_patients": len(np.unique(d["window"][active])),
                      "negative_tail_median_residual": float(np.median(d["residual"][active]))}
    assert diagnostic["strictly_positive_cases"] == 0 and coefficient < 0
    k = 1.01
    recipe = copy.deepcopy(parent)
    recipe["name"] = "clinical_v0_head47_sharp101"
    recipe["rationale"] += (
        " Unit 47 has no positive median-selected activations on the 128-patient calibration cohort. "
        "Its negative GELU tail adds positive output with effective coefficient -1.67555; 10300 such "
        "scored cases across 127 patients have positive median residual. Sharpen this gate by a fixed "
        "conservative 1%: multiply its incoming row and bias by 1.01 and divide its composed outgoing "
        "coefficient by 1.01. This preserves the positive linear asymptote and reduces the positive "
        "negative-tail contribution. The direct head correction is at most about 0.0049 normalized "
        "units; later denoising can propagate it. No fitting, gradient information, optimizer, parameter "
        "sweep or test targets determine this explicit edit."
    )
    recipe["operations"] += [
        {"key": "out_proj.0.mlp.0.weight", "kind": "multiply_rows", "start": 47, "stop": 48, "value": k},
        {"key": "out_proj.0.mlp.0.bias", "kind": "multiply_rows", "start": 47, "stop": 48, "value": k},
        {"key": "out_proj.0.mlp.3.weight", "kind": "scale_effective_columns", "readout_key": "out_proj.1.weight", "columns": [47], "value": 1 / k}]
    recipe["diagnostic_negative_tail"] = diagnostic
    before, after, original = Inference(source, 0, parent), Inference(source, 0, recipe), Inference(source, 0)
    changes = {name: int((v != original.model.state_dict()[name]).sum())
               for name, v in after.model.state_dict().items() if not torch.equal(v, original.model.state_dict()[name])}
    # Evaluate the stored float32 weights in float64 so cancellation in a
    # synthetic full-head output does not hide the small programmed change.
    h0, h1 = copy.deepcopy(before.model.out_proj).double(), copy.deepcopy(after.model.out_proj).double()
    x = torch.randn(512, 64, generator=torch.Generator().manual_seed(731)).double()
    z = h0[0].mlp[0](x)[:, 47]
    c = (h0[1].weight @ h0[0].mlp[3].weight)[0, 47]
    expected = c * (torch.nn.functional.gelu(k * z) / k - torch.nn.functional.gelu(z))
    actual = (h1(x) - h0(x))[:, 0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    record = {"time_unix": time.time(), "source_sha256": sha256(source), "changed_values": sum(changes.values()),
              "changed_tensors": changes, "diagnostic": diagnostic,
              "algebra_max_error_float64_on_stored_weights": float((actual - expected).abs().max()),
              "initial_float32_full_output_check": {"passed": False, "maximum_error": 1.63583317771554e-5,
                                                     "absolute_tolerance": 2e-6,
                                                     "resolution": "Independent float64 evaluation of stored float32 tensors separates output-subtraction cancellation from weight rounding."},
              "training_or_gradients_used": False, "hypothesis_not_yet_validated": True}
    output.write_text(json.dumps(recipe, indent=2) + "\n")
    record["recipe"] = str(output); record["recipe_sha256"] = sha256(output)
    audit_path.write_text(json.dumps(record, indent=2) + "\n")
    plan = {"created_unix": time.time(), "recipe_sha256": sha256(output),
            "selection": "Advance only if both calibration metrics improve over V1, then confirm on separate patients.",
            "jobs": [{"dataset": "mimic-iii-challenge_point", "replicate": 0, "split": "calibration", "limit": 128,
                      "samples": 100, "seed": 20261103, "chunk": 20, "batch": 1, "device": "cuda", "backend": "triton",
                      "edit": str(output), "output": "artifacts/results/clinical_point_v0_head47_sharp101_cal128x100_seed1_chunk20"}]}
    path = Path("artifacts/plans/clinical_head47_sharp101_calibration128.json")
    assert not path.exists()
    path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
