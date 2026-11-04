"""Check clinical forward inference against methods extracted from CoSTI.

Do not import its training engine or construct an optimizer. Only the
original forward/sampling methods are compiled, with explicit paired noise.
"""

import ast
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .checkpoints import sha256
from .data import load_data
from .run import Inference


def extract(path, name, namespace, class_name=None):
    tree = ast.parse(Path(path).read_text())
    nodes = tree.body
    if class_name is not None:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef)
                     and node.name == class_name).body
    node = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    # These whitelisted methods do not contain any training operation.
    assert name in {"forward_cm", "_impute_batch", "get_scalings", "rescale_vector",
                    "interpolate_sample", "create_interpolation", "redefine_eval_mask"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class ExplicitNoiseTorch:
    def __init__(self):
        self.noise = iter(())

    def __getattr__(self, name):
        return getattr(torch, name)

    def randn_like(self, value):
        noise = next(self.noise)
        assert noise.shape == value.shape
        return noise.clone()


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    output = Path("artifacts/audit/clinical_source_forward_semantics.json")
    if output.exists():
        raise FileExistsError(output)
    files = ["src/engines/ct_engine.py", "src/schedulers/karras.py",
             "src/nn/utils.py", "src/data/data_utils.py"]
    namespace = {"torch": torch, "np": np, "pd": pd}
    rescale = extract(files[2], "rescale_vector", namespace)
    namespace["rescale_vector"] = rescale
    get_scalings = extract(files[1], "get_scalings", namespace, "KarrasScheduler")
    extract(files[3], "interpolate_sample", namespace)
    create_interpolation = extract(files[3], "create_interpolation", namespace)
    redefine_eval_mask = extract(files[3], "redefine_eval_mask", namespace)
    facade = ExplicitNoiseTorch()
    sampler_namespace = {"torch": facade, "rescale_vector": rescale}
    forward_cm = extract(files[0], "forward_cm", sampler_namespace, "CT_Engine")
    impute = extract(files[0], "_impute_batch", sampler_namespace, "CT_Engine")
    data = load_data("mimic-iii-challenge_point")
    ids = data.indices("calibration", 2)
    batch = data.batch(ids)
    # CoSTI's original mask subtraction uses a numeric 0/1 mask. uint8 is
    # accepted by TSL and retains exactly the same condition values; modern
    # Torch rejects int-minus-bool in that original preprocessing function.
    source_batch = dict(batch, mask=batch["mask"].to(torch.uint8),
                        og_mask=batch["mask"] | batch["eval_mask"])
    source_batch = redefine_eval_mask(create_interpolation(source_batch))
    torch.testing.assert_close(source_batch["x_interpolated"], batch["x_interpolated"], rtol=0, atol=0)
    torch.testing.assert_close(source_batch["eval_mask"].bool(), batch["eval_mask"], rtol=0, atol=0)
    checkpoint = next(Path("../weights/ctimba_mimic-iii-challenge_point/version_0").glob("*.ckpt"))
    recipe = json.loads(Path("weight_space/edits/clinical_v0_affine_tail_readout.json").read_text())
    n, sigmas, seed = 3, [80, 20.92], 73
    noise = torch.stack([torch.randn((n, len(sigmas), *batch["x"].shape[1:]),
                         generator=torch.Generator().manual_seed(seed + int(i) * 1009)) for i in ids])
    rows = []
    for label, edit in [("original", None), ("selected_edit", recipe)]:
        runner = Inference(checkpoint, 0, edit)
        scheduler = types.SimpleNamespace(sigma_min=runner.sigma_min, sigma_data=runner.sigma_data)
        scheduler.get_scalings = types.MethodType(get_scalings, scheduler)
        engine = types.SimpleNamespace(model=runner.model, scheduler=scheduler, test_sigmas=sigmas)
        engine.forward_cm = types.MethodType(forward_cm, engine)
        expected = []
        for draw in range(n):
            facade.noise = iter(noise[:, draw].unbind(1))
            expected.append(impute(engine, source_batch))
            assert next(facade.noise, None) is None
        expected = torch.stack(expected, 1).median(1).values
        actual = runner.predict(batch, ids, n, sigmas, seed, chunk=1)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-5)
        assert all(not p.requires_grad and p.grad is None for p in runner.model.parameters())
        rows.append({"model": label, "maximum_normalized_prediction_difference":
                     float((actual - expected).abs().max())})
    result = {"patients": 2, "imputations": n, "sigmas": sigmas, "seed": seed,
              "source_method_sha256": {name: sha256(name) for name in files},
              "interpolation_exact": True, "scoring_mask_exact": True,
              "source_mask_storage": "uint8 with identical 0/1 values, for original arithmetic on modern Torch",
              "rows": rows, "gradients_disabled": not torch.is_grad_enabled(),
              "note": "Original forward sampling methods, identical supplied noise, independent scalar/tensor scheduler arithmetic. Training engine is not imported or instantiated. Small floating-point differences are covered by the stated 2e-5 relative/1e-5 absolute tolerance."}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
