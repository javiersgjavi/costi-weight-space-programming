"""Compare paired evaluations; bootstrap complete windows, not scalar entries."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(base, edited, bootstrap=10000):
    for field in ("dataset", "replicate", "split", "window_ids", "samples", "sigmas", "seed", "checkpoint_sha256"):
        if base[field] != edited[field]:
            raise ValueError(f"Unpaired evaluations: {field}")
    for field in ("backend", "harness_version", "graph_rng_draws"):
        if base["runtime"].get(field) != edited["runtime"].get(field):
            raise ValueError(f"Different inference protocol: {field}")
    if "source_sha256" in base["runtime"] and "source_sha256" in edited["runtime"]:
        if base["runtime"]["source_sha256"] != edited["runtime"]["source_sha256"]:
            raise ValueError("Inference source files changed between paired evaluations")
    b = base["per_window"]
    e = edited["per_window"]
    if [(r["id"], r["n"]) for r in b] != [(r["id"], r["n"]) for r in e]:
        raise ValueError("Windows or scoring masks differ")
    counts = np.array([r["n"] for r in b])
    rng = np.random.default_rng(20261103)
    results = {}
    for metric in ("mae", "mse"):
        valid = counts > 0
        delta = np.array([ee[metric] - bb[metric] for bb, ee in zip(b, e) if bb["n"] > 0])
        weights = counts[valid]
        replicates = []
        for start in range(0, bootstrap, 256):
            indices = rng.integers(0, len(delta), size=(min(256, bootstrap - start), len(delta)))
            replicates.extend(((delta[indices] * weights[indices]).sum(1) / weights[indices].sum(1)).tolist())
        ci = np.quantile(replicates, [.025, .975]).tolist()
        results[metric] = {
            "original": base["metrics"][metric], "edited": edited["metrics"][metric],
            "change_percent": 100 * (edited["metrics"][metric] / base["metrics"][metric] - 1),
            "delta_ci95_window_bootstrap": ci,
            "improved_windows": int((delta < 0).sum()), "windows": len(delta),
        }
    primary = results["mae"]
    secondary = results["mse"]
    established = primary["delta_ci95_window_bootstrap"][1] < 0 and secondary["change_percent"] < 0
    return {"dataset": base["dataset"], "replicate": base["replicate"], "split": base["split"],
            "edit": edited["edit"]["name"], "samples": base["samples"], "metrics": results,
            "criterion_met_on_this_partition": established,
            "final_benchmark_claim": bool(established and base["split"] == "test" and base["samples"] == 100
                                          and len(b) == base["partition_sizes"]["test"]),
            "uncertainty_scope": "Paired resampling of windows; excludes Monte Carlo seed uncertainty. Adjacent windows may be temporally correlated."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("original", type=Path)
    p.add_argument("edited", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = compare(json.loads(args.original.read_text()), json.loads(args.edited.read_text()))
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
