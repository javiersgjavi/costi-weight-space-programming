"""Report fixed-seed pairs with the appropriate independent sampling unit."""

import argparse
import json
from pathlib import Path

import numpy as np

from .compare import compare
from .checkpoints import sha256


def block_interval(loss_deltas, counts, block_size=12, draws=10000):
    """Resample contiguous windows, preserving pairing and averaging seeds first."""
    rng = np.random.default_rng(20261103)
    n = len(counts)
    estimates = []
    offsets = np.arange(block_size)
    for start in range(0, draws, 256):
        rows = min(256, draws - start)
        starts = rng.integers(0, n, size=(rows, int(np.ceil(n / block_size))))
        ids = ((starts[..., None] + offsets) % n).reshape(rows, -1)[:, :n]
        denominator = counts[ids].sum(1)
        valid = denominator > 0
        estimates.extend((loss_deltas[ids].sum(1)[valid] / denominator[valid]).tolist())
    return np.quantile(estimates, [.025, .975]).tolist()


def summarize(pairs, block_size=12, reference_equivalences=None):
    loaded = [(json.loads(Path(a).read_text()), json.loads(Path(b).read_text())) for a, b in pairs]
    equivalence_rows = reference_equivalences or []
    reference_map = {str(Path(a).resolve()): Path(b).resolve() for a, b in equivalence_rows}
    if len(reference_map) != len(equivalence_rows):
        raise ValueError("Repeated reference-equivalence mapping")
    pair_reference_paths = {str(Path(a).resolve()) for a, b in pairs}
    if not set(reference_map).issubset(pair_reference_paths):
        raise ValueError("Equivalence evidence refers to a different reference result")
    per_seed, bridges, raw_audits = [], [], []
    first_raw_invariants = None
    for (a_path, b_path), (a, b) in zip(pairs, loaded):
        comparable_reference = a
        reference_path = str(Path(a_path).resolve())
        if reference_path in reference_map:
            from .evaluation_audit import verified_native_reference
            comparable_reference, bridge = verified_native_reference(a, Path(reference_path), reference_map[reference_path])
            bridges.append(bridge)
        per_seed.append(compare(comparable_reference, b))
        if reference_map:
            from .evaluation_audit import audit_raw
            aa, bb = audit_raw(Path(a_path), a), audit_raw(Path(b_path), b)
            invariants = {key: aa[key] for key in ("target", "mask", "bias", "scale", "window_ids")}
            for key, value in invariants.items():
                if not np.array_equal(value, bb[key]):
                    raise ValueError(f"Raw paired arrays differ in {key}")
                if first_raw_invariants is not None and not np.array_equal(value, first_raw_invariants[key]):
                    raise ValueError(f"Raw seed arrays differ in {key}")
            first_raw_invariants = invariants
            raw_audits.append({"original_npz_sha256": sha256(Path(a_path).with_suffix(".npz")),
                               "edited_npz_sha256": sha256(Path(b_path).with_suffix(".npz")),
                               "same_targets_masks_scalers_ids": True,
                               "global_and_patient_metrics_recomputed_from_raw": True})
    first, edit = loaded[0]
    clinical = first["dataset"].startswith("mimic")
    if clinical and block_size != 1:
        raise ValueError("Clinical windows represent separate patients; use --block-size 1")
    interval_key = "delta_ci95_patient_bootstrap" if clinical else "delta_ci95_day_block_bootstrap"
    if len({a["seed"] for a, _ in loaded}) != len(loaded):
        raise ValueError("Repeated stochastic seed")
    for a, b in loaded:
        for field in ("dataset", "replicate", "split", "window_ids", "samples", "sigmas", "checkpoint_sha256"):
            if a[field] != first[field]:
                raise ValueError(f"Seed evaluations differ in {field}")
        if b["edit"] != edit["edit"]:
            raise ValueError("Different weight edits across seeds")
        if a["edit"] != first["edit"]:
            raise ValueError("Different reference weight edits across seeds")
        if first["split"] != "test" or first["samples"] != 100 or len(a["window_ids"]) != a["partition_sizes"]["test"]:
            raise ValueError("Only complete 100-imputation tests may enter this report")
    counts = np.array([w["n"] for w in first["per_window"]], dtype=float)
    results = {}
    for metric in ("mae", "mse", "mre"):
        reference = float(np.mean([a["metrics"][metric] for a, _ in loaded]))
        edited = float(np.mean([b["metrics"][metric] for _, b in loaded]))
        row = {"original": reference, "edited": edited, "change_percent": 100 * (edited / reference - 1)}
        if metric != "mre":
            delta_sums = np.mean([
                [((ee[metric] - aa[metric]) * aa["n"]) if aa["n"] else 0.
                 for aa, ee in zip(a["per_window"], b["per_window"])] for a, b in loaded], axis=0)
            if not np.isclose(delta_sums.sum() / counts.sum(), edited - reference, atol=1e-10):
                raise ValueError("Window aggregation does not reproduce global metric difference")
            row[interval_key] = block_interval(delta_sums, counts, block_size)
            row["improved_windows_on_seed_mean"] = int((delta_sums < 0).sum())
        results[metric] = row
    primary = results["mae"][interval_key][1] < 0
    balanced = primary and results["mse"]["change_percent"] < 0
    result = {
        "dataset": first["dataset"], "replicate": first["replicate"], "split": "test",
        "samples_per_seed": 100, "seeds": [a["seed"] for a, _ in loaded],
        "windows_per_seed": len(counts), "scored_entries_per_seed": int(counts.sum()),
        "edit": edit["edit"], "source_checkpoint_sha256": first["checkpoint_sha256"],
        "metrics": results, "primary_mae_improvement_established": bool(primary),
        "balanced_improvement_established": bool(balanced),
        "mse_improvement_established": bool(results["mse"][interval_key][1] < 0),
        "block_size_windows": block_size, "block_duration_hours": None if clinical else block_size * 2,
        "uncertainty_scope": (
            "Paired patient bootstrap, after averaging each patient's loss differences over the fixed seeds. Each 48-hour window is a separate patient; adjacent patient indices are not consecutive time windows. Does not capture all possible stochastic-seed uncertainty."
            if clinical else
            "Paired circular temporal-block bootstrap, after averaging per-window differences over the fixed seeds. Captures local time dependence; not all possible stochastic-seed uncertainty."),
        "selection_policy": "This statistical report does not establish an untouched holdout. Consult the candidate freeze and retained benchmark history for selection independence; later candidates are repeated benchmark assessments.",
        "per_seed": per_seed,
        "evidence": [{"original": str(Path(a).resolve()), "edited": str(Path(b).resolve()),
                      "original_sha256": sha256(a), "edited_sha256": sha256(b)} for a, b in pairs],
    }
    if bridges:
        result["runtime_equivalence_evidence"] = bridges
        result["raw_array_audits"] = raw_audits
        result["runtime_equivalence_scope"] = (
            "Stored shared-head references are reused only after bitwise native prediction parity on at least 128 matching patients per reference and matching core source hashes, weights, seed, data, scaling and noise settings. "
            "Only wrapper metadata is canonicalized in memory; original result files and measured scores remain unchanged. Full paired metrics are recomputed from all raw arrays. The subset parity check is not a claim that every native reference prediction was rerun.")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", nargs=2, action="append", required=True)
    p.add_argument("--block-size", type=int, default=12)
    p.add_argument("--reference-equivalence", nargs=2, action="append", metavar=("SHARED_REFERENCE", "NATIVE_SUBSET"),
                   help="Explicit bitwise native subset evidence for each reused shared-head reference")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(args.pair, args.block_size, args.reference_equivalence)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"metrics": result["metrics"],
                      "primary_mae_improvement_established": result["primary_mae_improvement_established"],
                      "balanced_improvement_established": result["balanced_improvement_established"]}, indent=2))


if __name__ == "__main__":
    main()
