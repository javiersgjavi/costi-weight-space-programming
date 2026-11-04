"""Audit paired clinical calibration/confirmation results and average fixed seeds."""

import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoints import sha256
from .compare import compare
from .data import load_data
from .evaluation_audit import audit_raw, verified_native_reference
from .summarize_test import block_interval


def report_pairs(pairs, require_full=False, native_reference_subset=None):
    loaded = [(json.loads(a.read_text()), json.loads(b.read_text())) for a, b in pairs]
    first, candidate = loaded[0]
    assert first["dataset"] == "mimic-iii-challenge_point" and first["split"] in ("calibration", "confirmation")
    assert first["samples"] == 100
    assert len({a["seed"] for a, b in loaded}) == len(loaded)
    if require_full:
        expected_ids = load_data(first["dataset"]).indices(first["split"], None)
        assert np.array_equal(first["window_ids"], expected_ids)
    raw_audits, per_seed, runtime_equivalence = [], [], []
    first_invariants = None
    for (a_path, b_path), (a, b) in zip(pairs, loaded):
        for key in ("dataset", "replicate", "split", "window_ids", "samples", "sigmas", "checkpoint_sha256"):
            assert a[key] == first[key], key
        assert a["edit"] == first["edit"] and b["edit"] == candidate["edit"]
        comparable_reference = a
        if a["runtime"]["backend"] != b["runtime"]["backend"]:
            if native_reference_subset is None:
                raise ValueError("Different backends require explicit native-reference subset parity evidence")
            comparable_reference, evidence = verified_native_reference(a, a_path, native_reference_subset)
            runtime_equivalence.append(evidence)
        per_seed.append(compare(comparable_reference, b))
        aa, bb = audit_raw(a_path, a), audit_raw(b_path, b)
        invariants = {key: aa[key] for key in ("target", "mask", "bias", "scale", "window_ids")}
        for key, array in invariants.items():
            assert np.array_equal(array, bb[key]), key
            if first_invariants is not None:
                assert np.array_equal(array, first_invariants[key]), key
        first_invariants = invariants
        raw_audits.append({"reference": str(a_path), "candidate": str(b_path),
                           "reference_json_sha256": sha256(a_path), "candidate_json_sha256": sha256(b_path),
                           "reference_npz_sha256": sha256(a_path.with_suffix(".npz")),
                           "candidate_npz_sha256": sha256(b_path.with_suffix(".npz")),
                           "same_targets_masks_scalers_ids": True,
                           "global_and_patient_metrics_recomputed_from_raw": True})
    counts = np.array([row["n"] for row in first["per_window"]])
    metrics = {}
    for metric in ("mae", "mse"):
        reference = float(np.mean([a["metrics"][metric] for a, b in loaded]))
        edited = float(np.mean([b["metrics"][metric] for a, b in loaded]))
        sums = np.mean([[((br[metric] - ar[metric]) * ar["n"]) if ar["n"] else 0.
                        for ar, br in zip(a["per_window"], b["per_window"])] for a, b in loaded], axis=0)
        assert np.isclose(sums.sum() / counts.sum(), edited - reference, rtol=1e-9, atol=1e-10)
        metrics[metric] = {"reference": reference, "candidate": edited,
                           "change_percent": 100 * (edited / reference - 1),
                           "delta_ci95_patient_bootstrap": block_interval(sums, counts, block_size=1),
                           "improved_patients_on_seed_mean": int((sums < 0).sum())}
    both_point = all(v["change_percent"] < 0 for v in metrics.values())
    both_intervals = all(v["delta_ci95_patient_bootstrap"][1] < 0 for v in metrics.values())
    return {"dataset": first["dataset"], "replicate": first["replicate"], "split": first["split"],
            "full_partition_verified": require_full, "samples": 100, "seeds": [a["seed"] for a, b in loaded],
            "patients_per_seed": len(counts), "patients_with_scoring_entries": int(np.sum(counts > 0)),
            "scoring_entries_per_seed": int(counts.sum()), "reference_recipe": first["edit"]["name"],
            "candidate_recipe": candidate["edit"]["name"], "metrics": metrics,
            "both_mean_metrics_improve": both_point, "both_patient_intervals_negative": both_intervals,
            "both_metrics_improve_on_each_seed": all(all(row["metrics"][m]["change_percent"] < 0 for m in ("mae", "mse")) for row in per_seed),
            "per_seed": per_seed, "raw_audits": raw_audits, "runtime_equivalence_evidence": runtime_equivalence,
            "uncertainty_scope": "Paired patient bootstrap after averaging each patient's loss differences over fixed seeds; not all stochastic seed uncertainty. This report does not establish an untouched test or authorize automatic promotion."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, action="append", type=Path, required=True)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--native-reference-subset", type=Path,
                        help="Explicit exact-prediction evidence for reusing a shared-head reference with native inference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = report_pairs(args.pair, args.require_full, args.native_reference_subset)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("split", "patients_per_seed", "seeds", "metrics", "both_mean_metrics_improve", "both_patient_intervals_negative", "both_metrics_improve_on_each_seed")}, indent=2))


if __name__ == "__main__":
    main()
