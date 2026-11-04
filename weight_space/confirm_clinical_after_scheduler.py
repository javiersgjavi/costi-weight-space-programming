"""Run a frozen confirmation plan after a known GPU scheduler finishes."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .after_plan import process_identity, verify_completed
from .checkpoints import sha256
from .clinical_partition_report import report_pairs


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", type=Path)
    args = parser.parse_args()
    schedule = read(args.schedule)
    locked = {str(args.schedule): sha256(args.schedule), str(Path(__file__)): sha256(Path(__file__))}
    for key in ("freeze", "plan", "selection_decision", "wait_schedule", "wait_plan"):
        assert sha256(schedule[key]) == schedule[key + "_sha256"], key
        locked[schedule[key]] = schedule[key + "_sha256"]
    freeze, plan = read(schedule["freeze"]), read(schedule["plan"])
    decision = read(schedule["selection_decision"])
    assert decision["decision"] == "advance_unchanged_to_full_confirmation"
    assert decision["calibration_reports"] == schedule["calibration_reports"]
    assert read(schedule["wait_schedule"])["test_plan"] == schedule["wait_plan"]
    assert plan["recipe_sha256"] == freeze["recipe_sha256"]
    assert plan["sigmas"] == [80, 20.92]
    assert [job["seed"] for job in plan["jobs"]] == [20261103, 20261104]
    for job in plan["jobs"]:
        assert job["dataset"] == "mimic-iii-challenge_point" and job["replicate"] == 0
        assert job["split"] == "confirmation" and "limit" not in job
        assert job["samples"] == job["chunk"] == 100 and job["batch"] == 1
        assert job["device"] == "cuda" and job["backend"] == "triton"
        assert job["edit"] == freeze["recipe"]
        for suffix in (".json", ".npz"):
            assert not Path(job["output"]).with_suffix(suffix).exists()
    for key in ("start_receipt", "batch_log"):
        assert not Path(schedule[key]).exists()
    for key in ("recipe", "source_checkpoint"):
        digest = freeze["recipe_sha256" if key == "recipe" else "source_sha256"]
        assert sha256(freeze[key]) == digest
        locked[freeze[key]] = digest
    for path, digest in freeze["source_sha256_files"].items():
        assert sha256(path) == digest
        locked[path] = digest
    for path in ("weight_space/after_plan.py", "weight_space/batch.py", "weight_space/compare.py",
                 "weight_space/clinical_partition_report.py", "weight_space/evaluation_audit.py",
                 "weight_space/summarize_test.py"):
        locked[path] = sha256(path)
    assert len(schedule["calibration_reports"]) == 2
    assert {r["reference_recipe"] for r in schedule["calibration_reports"]} == {
        "clinical_v0_gate50_midrange", "clinical_v0_gate212_shoulder"}
    for record in schedule["calibration_reports"]:
        path = Path(record["path"])
        assert sha256(path) == record["sha256"]
        locked[str(path)] = record["sha256"]
        report = read(path)
        assert report["split"] == "calibration" and report["patients_per_seed"] == 512
        assert report["samples"] == 100 and report["seeds"] == [20261103]
        assert report["candidate_recipe"] == freeze["candidate"]
        assert report["reference_recipe"] == record["reference_recipe"]
        assert report["both_mean_metrics_improve"] and report["both_metrics_improve_on_each_seed"]
        pairs = [(Path(r["reference"]), Path(r["candidate"])) for r in report["raw_audits"]]
        assert report_pairs(pairs) == report
        for raw in report["raw_audits"]:
            for role in ("reference", "candidate"):
                result = Path(raw[role])
                locked[str(result)] = raw[role + "_json_sha256"]
                locked[str(result.with_suffix(".npz"))] = raw[role + "_npz_sha256"]
    for record in schedule["full_references"]:
        assert not Path(record["output_report"]).exists()
        assert len(record["results"]) == 2
        for seed, path in zip((20261103, 20261104), record["results"]):
            result = read(path)
            assert result["split"] == "confirmation" and len(result["window_ids"]) == 2016
            assert result["samples"] == 100 and result["seed"] == seed
            assert result["sigmas"] == [80, 20.92]
            assert result["edit"]["name"] == record["recipe"]
            locked[path] = sha256(path)
            raw = str(Path(path).with_suffix(".npz"))
            locked[raw] = sha256(raw)

    def verify_locked():
        for path, digest in locked.items():
            assert sha256(path) == digest, path

    verify_locked()
    pid, identity = schedule["wait_pid"], schedule["wait_process_identity"]
    assert process_identity(pid) == identity
    command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    assert b"weight_space.test_clinical_after_queue" in command
    assert schedule["wait_schedule"].encode() in command
    print(json.dumps({"preflight_passed": True, "waiting_for_pid": pid,
                      "resource_dependency_only": True, "candidate_weights_already_frozen": True,
                      "locked_sha256": locked}), flush=True)
    started_wait, last_notice = time.monotonic(), -60.
    while process_identity(pid) == identity:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            break
        if state == "Z":
            break
        elapsed = time.monotonic() - started_wait
        if elapsed > 14400:
            raise TimeoutError("Prior GPU scheduler still live; confirmation not launched")
        if elapsed - last_notice >= 60:
            print(json.dumps({"waiting_for_pid": pid, "elapsed_seconds": round(elapsed),
                              "next_plan": schedule["plan"]}), flush=True)
            last_notice = elapsed
        time.sleep(5)
    verify_locked()
    verify_completed(Path(schedule["wait_plan"]))
    for job in read(schedule["wait_plan"])["jobs"]:
        previous = read(Path(job["output"]).with_suffix(".json"))
        assert previous["split"] == "test" and len(previous["window_ids"]) == 4033
        assert previous["samples"] == 100 and previous["sigmas"] == [80, 20.92]
    write_new(schedule["start_receipt"], {
        "time_unix": time.time(), "schedule": str(args.schedule), "locked_sha256": locked,
        "plan": schedule["plan"], "freeze": schedule["freeze"],
        "reason": "Run the unchanged candidate selected on calibration. Earlier test completion only releases the GPU slot; its metrics do not select or modify this candidate.",
        "no_automatic_test_export_or_promotion": True,
    })
    with Path(schedule["batch_log"]).open("x") as stream:
        subprocess.run([sys.executable, "-m", "weight_space.batch", schedule["plan"]],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    verify_locked()
    verify_completed(Path(schedule["plan"]))
    candidates = [Path(job["output"]).with_suffix(".json") for job in plan["jobs"]]
    for reference in schedule["full_references"]:
        pairs = list(zip(map(Path, reference["results"]), candidates))
        report = report_pairs(pairs, require_full=True)
        write_new(reference["output_report"], report)
        print(json.dumps({"report": reference["output_report"], "metrics": report["metrics"],
                          "both_metrics_improve_on_each_seed": report["both_metrics_improve_on_each_seed"]}), flush=True)
    print(json.dumps({"complete": True, "no_automatic_test_export_or_promotion": True}), flush=True)


if __name__ == "__main__":
    main()
