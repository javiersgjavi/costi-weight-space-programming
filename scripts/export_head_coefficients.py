"""Export the effective output-head coefficients of the original and V4 models.

CoSTI's output head is

    h (64) -> Linear(64, 256) -> GELU -> Linear(256, 64) -> Linear(64, 1) -> y

The last two maps are linear with nothing in between, so they collapse into a
single vector. For every hidden neuron j:

    z_j = w_j . h + b_j          (pre-activation, row j of the first Linear)
    a_j = r . W3[:, j]           (effective coefficient on the scalar output)
    y   = b_eff + sum_j a_j * GELU(z_j)

This script reads both checkpoints and writes one row per hidden neuron to
results/data/output_head_coefficients.csv. It only reads tensors; there is no
forward pass and no gradient.

Usage (checkpoints are not distributed with this repository):

    python scripts/export_head_coefficients.py \
        --original ../weights/ctimba_mimic-iii-challenge_point/version_0/epoch=49-val_loss=612.00507.ckpt \
        --edited   path/to/clinical_v4/weights.ckpt
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from weight_space.checkpoints import load_checkpoint  # noqa: E402

ROLES = {
    38: "positive readout path; boosted 1%",
    198: "positive readout path; boosted 1%",
    47: "negative-lobe path; sharpened by 1.01",
    255: "negative-lobe path; bias shifted by -0.01",
    246: "reused: 16x copy of unit 198 (bump, rising edge)",
    226: "reused: 16x copy of unit 198 (bump, peak at 1.5)",
    179: "reused: 16x copy of unit 198 (bump, off at 2)",
    181: "reused: 64x copy of z38 - z198 (ramp, on at 0)",
    150: "reused: 64x copy of z38 - z198 (ramp, cap at 1)",
}


def head(path):
    state = load_checkpoint(path)["state_dict"]
    get = lambda k: state["model.out_proj." + k].double()  # noqa: E731
    return {
        "w_in": get("0.mlp.0.weight"), "b_in": get("0.mlp.0.bias"),
        "w_mid": get("0.mlp.3.weight"), "b_mid": get("0.mlp.3.bias"),
        "r": get("1.weight"), "b_out": get("1.bias"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--edited", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "results/data/output_head_coefficients.csv")
    args = p.parse_args()

    o, e = head(args.original), head(args.edited)
    a_o = (o["r"] @ o["w_mid"]).flatten()
    a_e = (e["r"] @ e["w_mid"]).flatten()
    row_changed = (o["w_in"] != e["w_in"]).any(1) | (o["b_in"] != e["b_in"])
    col_changed = (o["w_mid"] != e["w_mid"]).any(0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["unit", "a_original", "a_v4", "bias_original", "bias_v4",
                      "input_norm_original", "input_norm_v4", "input_row_changed",
                      "output_column_changed", "role"])
        for j in range(len(a_o)):
            out.writerow([j, f"{a_o[j]:.9g}", f"{a_e[j]:.9g}",
                          f"{o['b_in'][j]:.9g}", f"{e['b_in'][j]:.9g}",
                          f"{o['w_in'][j].norm():.9g}", f"{e['w_in'][j].norm():.9g}",
                          int(row_changed[j]), int(col_changed[j]), ROLES.get(j, "")])
    idle = int((a_o.abs() < 0.01).sum())
    print(f"wrote {args.output}: {len(a_o)} units, {idle} with |a| < 0.01 in the original")


if __name__ == "__main__":
    main()
