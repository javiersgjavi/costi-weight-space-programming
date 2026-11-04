"""Run a fixed list of paired forward evaluations, with immutable recipes."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .checkpoints import sha256
from .compare import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for job in plan["jobs"]:
        recipe = Path(job["edit"])
        if sha256(recipe) != plan["recipe_sha256"]:
            raise ValueError("The frozen recipe changed")
        for name, expected in plan.get("source_sha256", {}).items():
            if sha256(name) != expected:
                raise ValueError(f"Frozen inference implementation changed: {name}")
        outputs = [Path(job[key]) for key in ("output_original", "output_edited")]
        completed = [p.with_suffix(".json").exists() and p.with_suffix(".npz").exists()
                     for p in outputs]
        if any(completed) and not all(completed):
            raise ValueError("Incomplete pair: retain it for inspection and use new result paths")
        if not all(completed):
            command = [sys.executable, "-m", "weight_space.paired_head"]
            for key, value in job.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            logfile = Path("artifacts/logs") / (outputs[1].name + "_paired.log")
            logfile.parent.mkdir(parents=True, exist_ok=True)
            print(json.dumps({"start": str(outputs[1]), "command": command}), flush=True)
            with logfile.open("a") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        left, right = [json.loads(p.with_suffix(".json").read_text()) for p in outputs]
        for result in (left, right):
            for key in ("dataset", "replicate", "split", "samples", "seed"):
                if result[key] != job[key]:
                    raise ValueError(f"Completed result disagrees with the plan: {key}")
        if right["edit"] != json.loads(recipe.read_text()) or left["edit"]["operations"]:
            raise ValueError("Completed result used different weights")
        result = compare(left, right)
        comparison_path = Path(str(outputs[1]) + "_comparison.json")
        if comparison_path.exists():
            if json.loads(comparison_path.read_text()) != result:
                raise ValueError("A previous comparison no longer reproduces")
        else:
            comparison_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"complete": str(outputs[1]), "metrics": result["metrics"]}), flush=True)
        if plan.get("require_joint_improvement_before_next_job", False):
            if not all(right["metrics"][key] < left["metrics"][key] for key in ("mae", "mse")):
                raise ValueError("Both metrics must improve before any subsequent evaluation")


if __name__ == "__main__":
    main()
