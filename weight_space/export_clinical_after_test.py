"""Export one preselected frozen candidate after its fixed test passes.

This schedules no accuracy evaluations and makes no new weight decisions.
It preserves earlier exports and leaves goal/progress/plot state unchanged.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .after_plan import process_identity, verify_completed
from .checkpoints import sha256
from .finalize_clinical_candidate import reproduce_report, verify_start


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def check_locks(plan):
    for path, digest in plan["locked_sha256"].items():
        assert sha256(path) == digest, path


def wait_for(plan):
    began, last = time.monotonic(), -60
    while True:
        active = [row for row in plan["wait_for"]
                  if process_identity(row["pid"]) == row["process_identity"]]
        if not active:
            return
        elapsed = time.monotonic() - began
        if elapsed > plan["max_wait_seconds"]:
            raise TimeoutError("Fixed test/report collection incomplete; no export attempted")
        if elapsed - last >= 60:
            print(json.dumps({"waiting_for": [row["label"] for row in active],
                              "elapsed_seconds": round(elapsed)}), flush=True)
            last = elapsed
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    plan_hash = sha256(args.plan)
    check_locks(plan)
    export = Path(plan["export"])
    audit_dir = Path(plan["audit_dir"])
    assert not export.parent.exists() and not audit_dir.exists()
    for row in plan["wait_for"]:
        current = process_identity(row["pid"])
        assert current is None or current == row["process_identity"], row
    print(json.dumps({"scheduled_fixed_candidate": plan["recipe_name"],
                      "plan_sha256": plan_hash, "no_accuracy_forward_scheduled": True}), flush=True)
    wait_for(plan)
    assert sha256(args.plan) == plan_hash
    check_locks(plan)
    verify_completed(Path(plan["test_plan"]))
    freeze = read(plan["freeze"])
    recipe = read(freeze["recipe"])
    assert recipe["name"] == plan["recipe_name"]
    original, pairs = reproduce_report(Path(plan["original_report"]))
    retained, retained_pairs = reproduce_report(Path(plan["retained_report"]))
    assert original["edit"] == retained["edit"] == recipe
    assert {b for a, b in pairs} == {b for a, b in retained_pairs}
    assert all(not read(a)["edit"]["operations"] for a, b in pairs)
    assert {read(a)["edit"]["name"] for a, b in retained_pairs} == {plan["retained_recipe_name"]}
    reuse = verify_start(read(plan["test_start"]), freeze, Path(plan["freeze"]), pairs, recipe)
    joint = all(report["metrics"][metric]["change_percent"] < 0
                for report in (original, retained) for metric in ("mae", "mse"))
    per_seed = all(seed["metrics"][metric]["change_percent"] < 0
                   for report in (original, retained) for seed in report["per_seed"]
                   for metric in ("mae", "mse"))
    passed = joint and per_seed and original["balanced_improvement_established"]
    decision = {"time_unix": time.time(), "candidate": plan["recipe_name"],
                "decision": "export_frozen_candidate" if passed else "keep_previous_checkpoint",
                "benchmark_reuse": reuse, "both_means_lower_vs_original_and_retained": joint,
                "both_metrics_lower_each_seed_vs_original_and_retained": per_seed,
                "metrics_vs_original": original["metrics"], "metrics_vs_retained": retained["metrics"],
                "plan": str(args.plan), "plan_sha256": plan_hash,
                "evidence_sha256": {path: sha256(path) for path in
                                    (plan["original_report"], plan["retained_report"])},
                "export_not_verified_yet": True}
    write_new(plan["decision"], decision)
    if not passed:
        print(json.dumps({"decision": decision["decision"], "no_export": True}), flush=True)
        return
    check_locks(plan)
    assert not export.parent.exists() and not audit_dir.exists()
    commands = [
        [sys.executable, "-m", "weight_space.export", "--original-result", pairs[0][0],
         "--edited-result", pairs[0][1], "--aggregate-report", plan["original_report"],
         "--output", str(export)],
        [sys.executable, "-m", "weight_space.finalize_clinical_candidate",
         "--report", plan["original_report"], "--retained-comparison", plan["retained_report"],
         "--freeze", plan["freeze"], "--test-start", plan["test_start"],
         "--checkpoint", str(export), "--output-dir", str(audit_dir)],
    ]
    for command, log_path in zip(commands, (plan["export_log"], plan["verification_log"])):
        with Path(log_path).open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        check_locks(plan)
    verified = read(audit_dir / "verification.json")
    assert verified["verification_passed"] and verified["source_checkpoint_unchanged"]
    assert verified["metrics_vs_original"] == original["metrics"]
    assert verified["metrics_vs_retained"] == retained["metrics"]
    receipt = {"time_unix": time.time(), "verification_passed": True,
               "checkpoint": str(export), "checkpoint_sha256": sha256(export),
               "verification": str(audit_dir / "verification.json"),
               "verification_sha256": sha256(audit_dir / "verification.json"),
               "decision": plan["decision"], "benchmark_reuse": reuse,
               "goal_progress_and_plots_not_changed": True}
    write_new(plan["receipt"], receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
