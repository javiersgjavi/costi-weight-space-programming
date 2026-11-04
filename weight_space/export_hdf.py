"""Export fixed-format Pandas HDF tables using h5py, without PyTables."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--tables", nargs="+", required=True)
    args = p.parse_args()
    arrays = {}
    with h5py.File(args.source, "r") as h:
        for name in args.tables:
            group = h[name]
            if "block1_values" in group:
                raise ValueError("Only homogeneous fixed-format frames supported")
            values = group["block0_values"][...]
            columns, index = group["axis0"][...], group["axis1"][...]
            if values.shape != (len(index), len(columns)):
                raise ValueError("Unexpected frame orientation")
            arrays[name] = values
            arrays[name + "_index"] = index
            arrays[name + "_columns"] = columns
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    manifest = {"source": str(args.source.resolve()),
                "sha256": hashlib.file_digest(args.source.open("rb"), "sha256").hexdigest(),
                "arrays": {k: list(v.shape) for k, v in arrays.items()}}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
