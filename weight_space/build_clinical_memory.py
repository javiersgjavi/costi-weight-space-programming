"""Program a longer reverse memory with finite-window input-gain compensation.

The constants follow the existing recurrence and calibration input timescales.
No targets, derivatives, fitting, parameter search or optimizer are used.
"""

import json
import math
import time
from pathlib import Path

import torch

from .checkpoints import sha256
from .run import Inference


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    recipe_path = Path("weight_space/edits/physionet/clinical_v0_reverse_memory_finite_gain.json")
    audit_path = Path("artifacts/audit/clinical_reverse_memory_finite_gain_construction.json")
    if recipe_path.exists() or audit_path.exists():
        raise FileExistsError("Construction records are immutable")
    source = next(Path("../weights/ctimba_mimic-iii-challenge_point/version_0").glob("*.ckpt"))
    parent_path = Path("weight_space/edits/physionet/clinical_v0_affine_tail_readout.json")
    diagnostic_path = Path("artifacts/audit/clinical_v1_conditioning_memory_cal32.json")
    parent = json.loads(parent_path.read_text())
    diagnostic = json.loads(diagnostic_path.read_text())
    assert diagnostic["split"] == "calibration" and diagnostic["edit"] == parent
    runner = Inference(source, 0, parent)
    module = runner.model.cfem.fe.mamba.mamba.mamba_block[2]
    rates = module.A_b_log.double().exp()
    delta = torch.tensor(diagnostic["mean"]["reverse"]["channel_mean_delta"], dtype=torch.float64)
    decay = rates * delta[:, None]
    slow = decay.median(0).values.argsort()[:4].tolist()
    assert slow == [0, 1, 2, 3]
    horizon = 48
    times = torch.arange(1, horizon + 1, dtype=torch.float64)

    def average_gain(lam):
        return ((-torch.expm1(-lam[..., None] * times)) /
                (-torch.expm1(-lam))[..., None]).mean(-1)

    old_gain, longer_gain = average_gain(decay), average_gain(decay / 2)
    compensation = (old_gain / longer_gain).median(0).values
    rank = module.dt_proj_b.weight.shape[1]
    prefix = "cfem.fe.mamba.mamba.mamba_block.2."
    recipe = json.loads(json.dumps(parent))
    recipe["name"] = "clinical_v0_reverse_memory_finite_gain"
    recipe["rationale"] += (
        " Actual conditioning forward passes on 32 calibration patients show a much shorter reverse memory "
        "(median slowest-state half-life about 3.9 hours) than forward memory (about 96 hours). "
        "A previous uncorrected reverse-memory extension traded MAE for MSE on a smaller exploratory cohort. "
        "Halve decay rates only for reverse states 0-3, and reduce their B-projection input drive to separate "
        "memory duration from amplitude. For each learned channel/state recurrence, compute its constant-input "
        "gain averaged over prefixes 1..48 using calibration-input mean delta. Its old/new gain ratio gives "
        "the compensating input scale; a median across channels supplies one scale per shared B state. "
        "This is an analytic, finite-horizon correction, not a target fit or parameter sweep. "
        "The shared B projection cannot exactly preserve every channel's gain; this is a forward-tested hypothesis. "
        "Only calibration inputs determine these factors; no clinical test outcome determines coefficients."
    )
    recipe["operations"].append({"key": prefix + "A_b_log", "kind": "add_columns",
                                  "start": 0, "stop": 4, "value": -math.log(2)})
    for state in slow:
        recipe["operations"].append({"key": prefix + "x_proj_b.weight", "kind": "multiply_rows",
                                      "start": rank + state, "stop": rank + state + 1,
                                      "value": float(compensation[state])})
    edited = Inference(source, 0, recipe)
    # Independent explicit constant-input recurrence checks the closed formula.
    h0, h1 = torch.zeros_like(decay), torch.zeros_like(decay)
    means0, means1 = [], []
    for _ in range(horizon):
        h0 = torch.exp(-decay) * h0 + 1
        h1 = torch.exp(-decay / 2) * h1 + 1
        means0.append(h0); means1.append(h1)
    direct0, direct1 = torch.stack(means0).mean(0), torch.stack(means1).mean(0)
    torch.testing.assert_close(direct0, old_gain, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(direct1, longer_gain, rtol=1e-12, atol=1e-12)
    original = Inference(source, 0)
    changes = {k: int((v != original.model.state_dict()[k]).sum())
               for k, v in edited.model.state_dict().items()
               if not torch.equal(v, original.model.state_dict()[k])}
    gain_ratio = longer_gain[:, slow] * compensation[slow] / old_gain[:, slow]
    recipe["construction"] = {"diagnostic": str(diagnostic_path), "diagnostic_sha256": sha256(diagnostic_path),
                               "horizon_hours": horizon, "states": slow,
                               "input_drive_scales": compensation[slow].tolist()}
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
    audit = {"time_unix": time.time(), "source_sha256": sha256(source),
             "parent_recipe_sha256": sha256(parent_path), "recipe": str(recipe_path),
             "recipe_sha256": sha256(recipe_path), "changed_values": sum(changes.values()),
             "changed_tensors": changes, "state_input_drive_scales": compensation[slow].tolist(),
             "constant_input_formula_max_error": max(float((direct0 - old_gain).abs().max()),
                                                      float((direct1 - longer_gain).abs().max())),
             "compensated_channel_gain_ratio_quantiles": torch.quantile(gain_ratio.flatten(), torch.tensor([.1, .5, .9], dtype=torch.float64)).tolist(),
             "labels_or_gradients_used": False, "selection_split": "calibration", "hypothesis_not_yet_validated": True}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
