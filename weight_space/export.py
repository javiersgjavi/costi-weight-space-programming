"""Export a validated edited checkpoint, retaining the original unchanged."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_checkpoint, plain_metadata, sha256
from .compare import compare
from .data import load_data
from .run import Inference


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original-result", type=Path, required=True)
    p.add_argument("--edited-result", type=Path, required=True)
    p.add_argument("--aggregate-report", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Do not overwrite an existing exported model")
    original = json.loads(args.original_result.read_text())
    edited = json.loads(args.edited_result.read_text())
    if args.aggregate_report:
        from .summarize_test import summarize
        report = json.loads(args.aggregate_report.read_text())
        pairs = [(r["original"], r["edited"]) for r in report["evidence"]]
        for row in report["evidence"]:
            for label in ("original", "edited"):
                if sha256(row[label]) != row[label + "_sha256"]:
                    raise ValueError("Aggregate report evidence changed")
        equivalences = [(row["reference"], row["native_subset"])
                        for row in report.get("runtime_equivalence_evidence", [])]
        recomputed = summarize(pairs, report["block_size_windows"], equivalences)
        if recomputed != report:
            raise ValueError("Aggregate report does not reproduce from its evidence")
        if (str(args.original_result.resolve()), str(args.edited_result.resolve())) not in pairs:
            raise ValueError("Requested checkpoint evidence is absent from aggregate report")
        if report["edit"] != edited["edit"] or not report["balanced_improvement_established"]:
            raise ValueError("Export requires a consistent improvement in the aggregate test")
        if report["dataset"].startswith("mimic") and len(report["seeds"]) < 2:
            raise ValueError("Clinical export requires at least two distinct test seeds")
        evidence = report
    else:
        evidence = compare(original, edited)
        if not evidence["final_benchmark_claim"]:
            raise ValueError("Export requires improvement on a complete paired 100-imputation test")
    source = Path(original["checkpoint"])
    if sha256(source) != original["checkpoint_sha256"]:
        raise ValueError("Original checkpoint changed since evaluation")
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    runner = Inference(source, original["replicate"], edited["edit"])
    record = load_checkpoint(source)
    state = {key: value.detach().cpu().clone() for key, value in record["state_dict"].items()}
    deltas = {}
    for key, value in runner.model.state_dict().items():
        full_key = "model." + key
        before = state[full_key]
        after = value.detach().cpu().clone()
        if not after.isfinite().all():
            raise ValueError(f"Non-finite export tensor: {key}")
        if not torch.equal(before, after):
            deltas[full_key] = after - before
        state[full_key] = after
    provenance = {
        "source_checkpoint": str(source), "source_sha256": original["checkpoint_sha256"],
        "dataset": original["dataset"], "replicate": original["replicate"],
        "edit": edited["edit"], "operations": runner.changes,
        "changed_tensors": len(deltas),
        "changed_values": sum(int(torch.count_nonzero(v)) for v in deltas.values()),
        "test_original_result": str(args.original_result.resolve()),
        "test_edited_result": str(args.edited_result.resolve()),
        "test_comparison": evidence,
        "training_or_gradients_used": False,
        "format": "Inference checkpoint with primitive metadata and tensor state; no optimizer or training-resume state.",
        "graph_factors": "Explicitly persisted to remove reliance on reconstruction of the upstream unregistered tensors.",
    }
    exported = {
        "state_dict": state,
        "hyper_parameters": {
            "model_kwargs": plain_metadata(record["hyper_parameters"]["model_kwargs"]),
            "adj": torch.from_numpy(np.array(record["hyper_parameters"]["adj"], copy=True)),
        },
        "weight_space_graph_factors": [v.detach().cpu().clone() for v in runner.model.input_sfe.support[-1]],
        "weight_space_provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, args.output)
    torch.save(deltas, args.output.with_suffix(".delta.pt"))
    loaded = torch.load(args.output, map_location="cpu", weights_only=True)
    if any(not torch.equal(state[k], loaded["state_dict"][k]) for k in state):
        raise ValueError("Export round-trip changed tensor values")
    reloaded = Inference(args.output, original["replicate"])
    if any(not torch.equal(value, reloaded.model.state_dict()[key])
           for key, value in runner.model.state_dict().items()):
        raise ValueError("Export could not be reconstructed as the same model")
    data = load_data(original["dataset"])
    ids = data.indices("calibration", 1)
    batch = data.batch(ids)
    expected = runner.predict(batch, ids, 3, original["sigmas"], 73, 3)
    actual = reloaded.predict(batch, ids, 3, original["sigmas"], 73, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if sha256(source) != original["checkpoint_sha256"]:
        raise ValueError("The original checkpoint must remain unchanged")
    provenance["roundtrip_prediction_verified"] = True
    provenance["export_sha256"] = sha256(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(args.output), "sha256": provenance["export_sha256"],
                      "changed_values": provenance["changed_values"], "test_comparison": evidence}, indent=2))


if __name__ == "__main__":
    main()
