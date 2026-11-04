"""Execute an explicit, reviewable list of inference jobs; never optimize weights."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("plan", type=Path)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    for job in plan["jobs"]:
        output = Path(job["output"])
        completed = output.with_suffix(".json")
        if completed.exists():
            result = json.loads(completed.read_text())
            for key in ("dataset", "replicate", "split", "samples", "seed"):
                if key in job and result[key] != job[key]:
                    raise ValueError(f"Existing result disagrees with plan: {output}: {key}")
            if job.get("edit"):
                if result["edit"] != json.loads(Path(job["edit"]).read_text()):
                    raise ValueError(f"Existing edit differs: {output}")
            elif result["edit"]["operations"]:
                raise ValueError(f"Expected an original checkpoint result: {output}")
            print(json.dumps({"already_complete": str(output)}), flush=True)
            continue
        command = [sys.executable, "-m", "weight_space.run"]
        for key, value in job.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        logfile = Path("artifacts/logs") / (output.name + ".log")
        logfile.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"start": str(output), "command": command}), flush=True)
        with logfile.open("a") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = json.loads(completed.read_text())
        print(json.dumps({"complete": str(output), "seconds": result["seconds"],
                          "metrics": result["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
