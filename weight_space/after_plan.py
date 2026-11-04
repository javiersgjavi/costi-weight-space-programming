"""Run a fixed inference plan after an identified earlier plan has finished.

This only schedules already specified forward passes. It does not select,
change, train, or promote weights, and it never terminates other processes.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .checkpoints import sha256


def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def verify_completed(plan_path):
    plan = json.loads(plan_path.read_text())
    for job in plan["jobs"]:
        prefix = Path(job["output"])
        result = json.loads(prefix.with_suffix(".json").read_text())
        for key in ("dataset", "replicate", "split", "samples", "seed"):
            assert result[key] == job[key], (prefix, key)
        assert result["edit"] == json.loads(Path(job["edit"]).read_text())
        with np.load(prefix.with_suffix(".npz")) as arrays:
            assert np.array_equal(arrays["window_ids"], result["window_ids"])
            assert np.isfinite(arrays["prediction"]).all()
            assert int(arrays["mask"].sum()) == result["metrics"]["n"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--wait-plan", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--max-wait-seconds", type=int, default=7200)
    args = parser.parse_args()
    locked_paths = [args.wait_plan, args.plan, args.freeze]
    hashes = {str(path): sha256(path) for path in locked_paths}
    identity = process_identity(args.wait_pid)
    if identity is not None:
        command = Path(f"/proc/{args.wait_pid}/cmdline").read_bytes().split(b"\0")
        assert b"weight_space.batch" in command and str(args.wait_plan).encode() in command
    start = time.monotonic()
    last_notice = -60.
    while identity is not None and process_identity(args.wait_pid) == identity:
        elapsed = time.monotonic() - start
        if elapsed > args.max_wait_seconds:
            raise TimeoutError("Prerequisite plan has not finished; no new inference was launched")
        if elapsed - last_notice >= 60:
            print(json.dumps({"waiting_for_pid": args.wait_pid, "elapsed_seconds": round(elapsed),
                              "next_plan": str(args.plan)}), flush=True)
            last_notice = elapsed
        time.sleep(5)
    for path in locked_paths:
        assert sha256(path) == hashes[str(path)], path
    verify_completed(args.wait_plan)
    freeze = json.loads(args.freeze.read_text())
    assert sha256(freeze["source_checkpoint"]) == freeze["source_sha256"]
    assert sha256(freeze["recipe"]) == freeze["recipe_sha256"]
    for path, digest in freeze["source_sha256_files"].items():
        assert sha256(path) == digest, path
    next_plan = json.loads(args.plan.read_text())
    assert next_plan["recipe_sha256"] == freeze["recipe_sha256"]
    for job in next_plan["jobs"]:
        assert job["edit"] == freeze["recipe"]
    print(json.dumps({"prerequisite_verified": str(args.wait_plan), "starting_fixed_plan": str(args.plan),
                      "locked_sha256": hashes}), flush=True)
    subprocess.run([sys.executable, "-m", "weight_space.batch", str(args.plan)], check=True)
    print(json.dumps({"complete": str(args.plan), "no_automatic_test_or_export": True}), flush=True)


if __name__ == "__main__":
    main()
