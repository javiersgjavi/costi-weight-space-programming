"""Run an already selected frozen candidate after the existing GPU queue.

The real pretest receipt is created only when the test is about to start.
This scheduler never selects weights, stops another job, or exports a model.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .after_plan import process_identity, verify_completed
from .checkpoints import sha256
from .evaluation_audit import audit_raw
from .finalize_clinical_candidate import verify_start_contract
from .summarize_test import summarize


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, record):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")


def start_record(schedule, plan, freeze):
    return {
        "time_unix": time.time(), "benchmark_reuse": True,
        "prior_clinical_test_candidate_recipes": plan["prior_clinical_test_candidate_recipes"],
        "candidate_freeze": schedule["freeze"], "candidate_freeze_sha256": schedule["freeze_sha256"],
        "test_plan": schedule["test_plan"], "test_plan_sha256": schedule["test_plan_sha256"],
        "frozen_recipe_sha256": freeze["recipe_sha256"],
        "selection_confirmation_report": schedule["confirmation_report"],
        "selection_confirmation_report_sha256": schedule["confirmation_report_sha256"],
        "selection_decision": schedule["selection_decision"],
        "selection_decision_sha256": schedule["selection_decision_sha256"],
        "native_reference_parity_audit": schedule["native_reference_parity_audit"],
        "native_reference_parity_audit_sha256": schedule["native_reference_parity_audit_sha256"],
        "retained_reference": read(schedule["retained_reference"]),
        "reason": (
            "Run the two prescribed full test seeds for the unchanged candidate selected after "
            "both full-confirmation seeds lowered MAE and MSE vs the verified retained checkpoint. "
            "Reuse the existing paired original and retained controls with their explicit runtime "
            "parity evidence. The benchmark has already been opened; no automatic export or promotion."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", type=Path)
    args = parser.parse_args()
    schedule = read(args.schedule)
    locked = {str(args.schedule): sha256(args.schedule), str(Path(__file__)): sha256(Path(__file__))}
    for key in ("wait_queue", "test_plan", "freeze", "confirmation_report", "selection_decision",
                "retained_reference", "native_reference_parity_audit"):
        assert sha256(schedule[key]) == schedule[key + "_sha256"]
        locked[schedule[key]] = schedule[key + "_sha256"]
    for path in ("weight_space/clinical_partition_report.py", "weight_space/evaluation_audit.py",
                 "weight_space/summarize_test.py", "weight_space/finalize_clinical_candidate.py",
                 "weight_space/batch.py", "weight_space/after_plan.py"):
        locked[path] = sha256(path)
    queue, plan, freeze = [read(schedule[key]) for key in ("wait_queue", "test_plan", "freeze")]
    for stage in queue["stages"]:
        for key in ("plan", "freeze"):
            assert sha256(stage[key]) == stage[key + "_sha256"]
            locked[stage[key]] = stage[key + "_sha256"]
    assert read(schedule["selection_decision"])["decision"] == "advance_unchanged_to_full_test"
    assert plan["sigmas"] == queue["sigmas"] == [80, 20.92]
    assert [job["seed"] for job in plan["jobs"]] == [20261103, 20261104]
    assert all(job["device"] == "cuda" and job["backend"] == "triton" for job in plan["jobs"])
    assert schedule["no_automatic_export_or_promotion"] is True
    for key in ("test_start", "original_report", "retained_report", "batch_log"):
        assert not Path(schedule[key]).exists()
    for job in plan["jobs"]:
        for extension in (".json", ".npz"):
            assert not Path(job["output"]).with_suffix(extension).exists()
    recipe = read(freeze["recipe"])
    assert sha256(freeze["recipe"]) == freeze["recipe_sha256"]
    assert sha256(freeze["source_checkpoint"]) == freeze["source_sha256"]
    locked[freeze["recipe"]] = freeze["recipe_sha256"]
    locked[freeze["source_checkpoint"]] = freeze["source_sha256"]
    for path, digest in freeze["source_sha256_files"].items():
        assert sha256(path) == digest
        locked[path] = digest
    # This validates the selected candidate and its actual retained reference
    # now, but deliberately does not create or pretend to create a test start.
    verify_start_contract(start_record(schedule, plan, freeze), freeze, Path(schedule["freeze"]), recipe)
    print(json.dumps({"prospective_pretest_contract_verified": True,
                      "actual_test_not_started": True, "locked_sha256": locked}), flush=True)
    pid, identity = schedule["wait_pid"], schedule["wait_process_identity"]
    start, last_notice = time.monotonic(), -60.
    while process_identity(pid) == identity:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            break
        if stat[0] == "Z":
            break
        elapsed = time.monotonic() - start
        if elapsed > 14400:
            raise TimeoutError("Existing confirmation queue still live; test not launched")
        if elapsed - last_notice >= 60:
            print(json.dumps({"waiting_for_queue_pid": pid, "elapsed_seconds": round(elapsed),
                              "test_plan": schedule["test_plan"]}), flush=True)
            last_notice = elapsed
        time.sleep(5)
    for path, digest in locked.items():
        assert sha256(path) == digest, path
    for stage in queue["stages"]:
        verify_completed(Path(stage["plan"]))
        for label in stage["references"]:
            path = Path(f"artifacts/clinical_{stage['candidate_stem']}_full_confirmation_{label}_report.json")
            report = read(path)
            assert report["full_partition_verified"] and report["patients_per_seed"] == 2016
            assert report["seeds"] == [20261103, 20261104]
    for job in plan["jobs"]:
        assert not Path(job["output"]).with_suffix(".json").exists()
        assert not Path(job["output"]).with_suffix(".npz").exists()
    started = start_record(schedule, plan, freeze)
    verify_start_contract(started, freeze, Path(schedule["freeze"]), recipe)
    write_new(schedule["test_start"], started)
    print(json.dumps({"actual_test_start": schedule["test_start"], "time_unix": started["time_unix"]}), flush=True)
    with Path(schedule["batch_log"]).open("x") as stream:
        subprocess.run([sys.executable, "-m", "weight_space.batch", schedule["test_plan"]],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    verify_completed(Path(schedule["test_plan"]))
    for path, digest in locked.items():
        assert sha256(path) == digest, path
    retained = read(started["retained_reference"]["test_report"])
    assert retained["seeds"] == [20261103, 20261104]
    candidates = [Path(job["output"]).with_suffix(".json") for job in plan["jobs"]]
    original_pairs = [(Path(row["original"]), candidate) for row, candidate in zip(retained["evidence"], candidates)]
    retained_pairs = [(Path(row["edited"]), candidate) for row, candidate in zip(retained["evidence"], candidates)]
    for pairs in (original_pairs, retained_pairs):
        for reference, candidate in pairs:
            assert read(candidate)["sigmas"] == [80, 20.92]
            a, b = [audit_raw(path, read(path)) for path in (reference, candidate)]
            for key in ("target", "mask", "bias", "scale", "window_ids"):
                np.testing.assert_array_equal(a[key], b[key])
    bridges = [(original, Path(f"artifacts/results/clinical_point_v0_original_test128x100_seed{i}_native_parity.json"))
               for i, (original, _) in enumerate(original_pairs, 1)]
    for key, pairs, evidence in (("original_report", original_pairs, bridges),
                                 ("retained_report", retained_pairs, None)):
        report = summarize(pairs, block_size=1, reference_equivalences=evidence)
        write_new(schedule[key], report)
        print(json.dumps({"report": schedule[key], "metrics": report["metrics"],
                          "balanced_improvement_established": report["balanced_improvement_established"]}), flush=True)
    print(json.dumps({"complete": True, "no_automatic_export_or_promotion": True}), flush=True)


if __name__ == "__main__":
    main()
