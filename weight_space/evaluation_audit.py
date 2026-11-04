"""Audit stored inference measurements and narrowly verified wrapper equivalence."""

import copy
import json
from pathlib import Path

import numpy as np

from .checkpoints import sha256


def audit_raw(path, report):
    path = Path(path)
    with np.load(path.with_suffix(".npz")) as stored:
        arrays = {key: stored[key] for key in stored.files}
    assert np.array_equal(arrays["window_ids"], report["window_ids"])
    assert np.isfinite(arrays["prediction"]).all()
    mask = arrays["mask"]
    error = (arrays["prediction"] - arrays["target"]).astype(np.float64)
    axes = tuple(range(1, mask.ndim))
    counts = mask.sum(axis=axes)
    assert np.array_equal(counts, [w["n"] for w in report["per_window"]])
    for metric, loss in [("mae", np.abs(error)), ("mse", error ** 2)]:
        sums = np.where(mask, loss, 0).sum(axis=axes)
        assert np.isclose(sums.sum() / counts.sum(), report["metrics"][metric], rtol=1e-12, atol=1e-12)
        for i, row in enumerate(report["per_window"]):
            if counts[i]:
                assert np.isclose(sums[i] / counts[i], row[metric], rtol=1e-12, atol=1e-12)
            else:
                assert row[metric] is None
    return arrays


def verified_native_reference(reference, reference_path, subset_path):
    """Accept only the documented clinical shared-head wrapper with bitwise evidence."""
    reference_path, subset_path = Path(reference_path), Path(subset_path)
    native = json.loads(subset_path.read_text())
    assert reference["dataset"] == "mimic-iii-challenge_point"
    assert reference["runtime"]["backend"] == "CUDA triton BiMamba-v2 shared-head"
    assert reference["runtime"]["harness_version"] == 3
    assert native["runtime"]["backend"] == "CUDA triton BiMamba-v2"
    assert native["runtime"]["harness_version"] == 2
    for key in ("dataset", "replicate", "split", "samples", "sigmas", "seed", "checkpoint_sha256", "edit"):
        assert reference[key] == native[key], key
    for key in ("torch", "device", "threads", "graph_rng_draws"):
        assert reference["runtime"][key] == native["runtime"][key], key
    assert native["samples"] == 100
    assert len(set(native["window_ids"])) == len(native["window_ids"]) >= 128
    shared_hashes = reference["runtime"]["source_sha256"]
    core_hashes = {key: value for key, value in shared_hashes.items() if Path(key).name != "paired_head.py"}
    assert core_hashes == native["runtime"]["source_sha256"]
    assert len(shared_hashes) == len(core_hashes) + 1
    with np.load(reference_path.with_suffix(".npz")) as full, np.load(subset_path.with_suffix(".npz")) as sub:
        assert np.array_equal(full["window_ids"], reference["window_ids"])
        assert np.array_equal(sub["window_ids"], native["window_ids"])
        assert np.all(np.diff(full["window_ids"]) > 0)
        positions = np.searchsorted(full["window_ids"], sub["window_ids"])
        assert np.all(positions < len(full["window_ids"]))
        assert np.array_equal(full["window_ids"][positions], sub["window_ids"])
        for key in ("prediction", "target", "mask"):
            assert np.array_equal(full[key][positions], sub[key]), key
        for key in ("bias", "scale"):
            assert np.array_equal(full[key], sub[key]), key
    canonical = copy.deepcopy(reference)
    canonical["runtime"]["backend"] = native["runtime"]["backend"]
    canonical["runtime"]["harness_version"] = native["runtime"]["harness_version"]
    canonical["runtime"]["source_sha256"] = core_hashes
    evidence = {"reference": str(reference_path), "native_subset": str(subset_path),
                "native_subset_sha256": sha256(subset_path), "native_subset_npz_sha256": sha256(subset_path.with_suffix(".npz")),
                "patients_compared": len(native["window_ids"]), "all_stored_predictions_bitwise_equal": True,
                "core_source_hashes_identical": True,
                "normalization": "Only shared-head wrapper backend/version and the extra wrapper source entry are canonicalized in memory for comparison. Source result files remain unchanged. Exact native parity is verified on 128 patients; full reference metrics are separately recomputed from raw arrays."}
    return canonical, evidence
