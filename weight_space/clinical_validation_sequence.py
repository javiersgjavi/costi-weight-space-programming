"""Execute the already frozen clinical validation plan, never change weights."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .checkpoints import sha256
from .compare import compare


def run_module(module, *arguments):
    command = [sys.executable, "-m", module, *map(str, arguments)]
    print(json.dumps({"execute": command}), flush=True)
    subprocess.run(command, check=True)


def main():
    # Let the in-flight measurement finish before retiring its stopped
    # coordinator. This never kills the evaluator or discards predictions.
    pending = Path("artifacts/results/clinical_point_v0_affine_calibration2017x100_seed1_full")
    coordinator_pid, evaluator_pid = 526581, 569165
    while not (pending.with_suffix(".json").exists() and pending.with_suffix(".npz").exists()):
        process = Path(f"/proc/{evaluator_pid}/stat")
        if not process.exists() or process.read_text().split(") ", 1)[1].startswith("Z"):
            raise RuntimeError("Pending calibration evaluator ended without complete evidence")
        time.sleep(5)
    while Path(f"/proc/{evaluator_pid}/stat").exists():
        status = Path(f"/proc/{evaluator_pid}/stat").read_text().split(") ", 1)[1]
        if status.startswith("Z"):
            break
        time.sleep(1)
    command_path = Path(f"/proc/{coordinator_pid}/cmdline")
    if command_path.exists():
        command = command_path.read_bytes()
        expected = b"weight_space.batch\x00artifacts/plans/clinical_point_full_validation_v1.json"
        if expected not in command:
            raise RuntimeError("Coordinator PID belongs to another process")
        os.kill(coordinator_pid, signal.SIGTERM)
        os.kill(coordinator_pid, signal.SIGCONT)
        print(json.dumps({"retired_completed_batch_coordinator": coordinator_pid}), flush=True)
    original = Path("artifacts/results/clinical_point_v0_base_calibration2017x100_seed1_full.json")
    report = compare(json.loads(original.read_text()), json.loads(pending.with_suffix(".json").read_text()))
    report_path = Path("artifacts/clinical_initial_affine_full_calibration_comparison.json")
    if not report_path.exists():
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"initial_affine_full_calibration": report["metrics"]}), flush=True)

    confirmation_plan = Path("artifacts/plans/clinical_point_full_confirmation_v1.json")
    run_module("weight_space.paired_batch", confirmation_plan)
    plan = json.loads(confirmation_plan.read_text())
    job = plan["jobs"][0]
    left_path = Path(job["output_original"]).with_suffix(".json")
    right_path = Path(job["output_edited"]).with_suffix(".json")
    left, right = [json.loads(p.read_text()) for p in (left_path, right_path)]
    if len(left["window_ids"]) != 2016 or left["samples"] != 100:
        raise RuntimeError("Expanded confirmation is incomplete")
    if not all(right["metrics"][key] < left["metrics"][key] for key in ("mae", "mse")):
        raise RuntimeError("Expanded confirmation did not improve both metrics; test remains unopened")
    test_plan = Path("artifacts/plans/clinical_point_full_test_v1.json")
    test_start = Path("artifacts/clinical_point_test_start_v1.json")
    if not test_start.exists():
        test_start.write_text(json.dumps({
            "time_unix": time.time(), "frozen_recipe_sha256": plan["recipe_sha256"],
            "confirmation_original": str(left_path), "confirmation_edited": str(right_path),
            "confirmation_original_sha256": sha256(left_path),
            "confirmation_edited_sha256": sha256(right_path),
            "test_plan_sha256": sha256(test_plan),
            "prior_clinical_test_candidate_evaluations": 0,
            "reason": "Frozen candidate lowered both metrics in expanded confirmation. Evaluate the same weights on both prespecified test seeds without further editing."
        }, indent=2) + "\n")
    run_module("weight_space.paired_batch", test_plan)
    tests = json.loads(test_plan.read_text())["jobs"]
    result = Path("artifacts/clinical_point_final_results.json")
    arguments = ["--block-size", "1", "--output", str(result)]
    for job in tests:
        arguments += ["--pair", str(Path(job["output_original"]).with_suffix(".json")),
                      str(Path(job["output_edited"]).with_suffix(".json"))]
    if not result.exists():
        run_module("weight_space.summarize_test", *arguments)
    summary = json.loads(result.read_text())
    if not summary["balanced_improvement_established"]:
        raise RuntimeError("Full test evidence does not support joint improvement; no export")
    export = Path("artifacts/edited_weights/clinical_v1/ctimba_mimic-iii-challenge_point/version_0/weights.ckpt")
    if not export.exists():
        run_module("weight_space.export", "--original-result", tests[0]["output_original"] + ".json",
                   "--edited-result", tests[0]["output_edited"] + ".json",
                   "--aggregate-report", result, "--output", export)
    print(json.dumps({"validation_and_export_complete": str(export),
                      "metrics": summary["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
