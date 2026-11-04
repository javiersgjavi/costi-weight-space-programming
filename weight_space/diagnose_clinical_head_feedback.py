"""Trace a frozen head edit through both steps on named calibration patients.

Step-only interventions are diagnostic counterfactuals, never benchmark results
or new checkpoint candidates. No hidden targets are scored by this script.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoints import sha256
from .data import load_data
from .finalize_clinical import state_hash
from .run import Inference


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--parent-result", type=Path, required=True)
    p.add_argument("--candidate-result", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists() and not args.output.with_suffix(".npz").exists()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    freeze = json.loads(args.freeze.read_text())
    source = Path(freeze["source_checkpoint"])
    parent_path, candidate_path = map(Path, (freeze["reference_recipe"], freeze["recipe"]))
    assert sha256(source) == freeze["source_sha256"]
    assert sha256(parent_path) == freeze["reference_recipe_sha256"]
    assert sha256(candidate_path) == freeze["recipe_sha256"]
    for path, digest in freeze["source_sha256_files"].items():
        assert sha256(path) == digest
    selection = json.loads(args.selection.read_text())
    ids = selection["patient_ids"]
    assert selection["split"] == "calibration" and len(set(ids)) == len(ids)
    assert selection["candidate_recipe_sha256"] == freeze["recipe_sha256"]
    assert sha256(args.parent_result.with_suffix(".npz")) == selection["parent_raw_sha256"]
    assert sha256(args.candidate_result.with_suffix(".npz")) == selection["candidate_raw_sha256"]
    metadata = [json.loads(path.read_text()) for path in (args.parent_result, args.candidate_result)]
    for record, recipe_path in zip(metadata, (parent_path, candidate_path)):
        assert record["edit"] == json.loads(recipe_path.read_text())
        assert record["split"] == "calibration" and record["samples"] == 100
        assert record["seed"] == 20261103 and record["sigmas"] == [80, 20.92]
        assert record["checkpoint_sha256"] == freeze["source_sha256"]
        assert set(ids).issubset(record["window_ids"])
    assert metadata[0]["window_ids"] == metadata[1]["window_ids"]
    reference_arrays = []
    for path in (args.parent_result, args.candidate_result):
        with np.load(path.with_suffix(".npz")) as arrays:
            # Targets are intentionally not accessed: only predictions and
            # identities/masks/scalers are needed for mechanism parity.
            reference_arrays.append({k: arrays[k] for k in
                                     ("prediction", "window_ids", "mask", "bias", "scale")})
    for key in ("window_ids", "mask", "bias", "scale"):
        assert np.array_equal(reference_arrays[0][key], reference_arrays[1][key])
    parent = Inference(source, 0, json.loads(parent_path.read_text()), device="cuda", backend="triton")
    child = Inference(source, 0, json.loads(candidate_path.read_text()), device="cuda", backend="triton")
    assert state_hash(child.model) == freeze["edited_state_sha256"]
    for name, value in parent.model.state_dict().items():
        if not name.startswith("out_proj."):
            assert torch.equal(value, child.model.state_dict()[name]), name
    for a, b in zip(parent.model.input_sfe.support[-1], child.model.input_sfe.support[-1]):
        assert torch.equal(a, b)
    child_head = copy.deepcopy(child.model.out_proj)
    audit = json.loads(Path(freeze["construction_audit"]).read_text())
    assert sha256(freeze["construction_audit"]) == freeze["construction_audit_sha256"]
    unit = audit["construction"]["source_unit"]
    for key in ("out_proj.0.mlp.0.weight", "out_proj.0.mlp.0.bias"):
        assert torch.equal(parent.model.state_dict()[key][unit], child.model.state_dict()[key][unit])
    names = json.loads(Path("artifacts/audit/clinical_variable_names.json").read_text())
    variables = selection.get("variables", ["AST"])
    assert variables and len(variables) == len(set(variables))
    trace_nodes = {name: names.index(name) for name in variables}
    dataset = load_data("mimic-iii-challenge_point")
    assert set(ids).issubset(dataset.indices("calibration").tolist())
    variants = [("parent", parent, None), ("candidate_both_steps", child, None),
                ("candidate_step1_only", parent, 0), ("candidate_step2_only", parent, 1)]
    predictions = {name: [] for name, _, _ in variants}
    patient_records = []
    masks = []
    for patient in ids:
        batch = dataset.batch([patient])
        mask = batch["eval_mask"][0, ..., 0]
        mask_gpu = mask.cuda()
        reference_index = metadata[0]["window_ids"].index(patient)
        assert np.array_equal(batch["eval_mask"].numpy()[0], reference_arrays[0]["mask"][reference_index])
        masks.append(batch["eval_mask"].numpy()[0])
        locations = torch.nonzero(mask, as_tuple=False)
        details = {}
        for name, runner, intervention_step in variants:
            captures, capture_calls, head_calls = [[], []], 0, 0

            def capture(module, inputs, output):
                nonlocal capture_calls
                captures[capture_calls % 2].append(output[..., unit][:, mask_gpu].cpu())
                capture_calls += 1

            def replace_head(module, inputs, output):
                nonlocal head_calls
                current = head_calls % 2
                head_calls += 1
                return child_head(inputs[0]) if current == intervention_step else output

            capture_handle = runner.model.out_proj[0].mlp[0].register_forward_hook(capture)
            intervention_handle = (runner.model.out_proj.register_forward_hook(replace_head)
                                   if intervention_step is not None else None)
            try:
                normalized, draws = runner.predict(batch, [patient], 100, [80, 20.92], 20261103, 20,
                                                    return_samples=True)
            finally:
                capture_handle.remove()
                if intervention_handle is not None:
                    intervention_handle.remove()
            assert capture_calls == 10
            if intervention_step is not None:
                assert head_calls == 10
            physical = (normalized * dataset.scale + dataset.bias).numpy()[0]
            if name in ("parent", "candidate_both_steps"):
                which = 0 if name == "parent" else 1
                assert np.array_equal(physical, reference_arrays[which]["prediction"][reference_index]), name
            predictions[name].append(physical)
            selected = draws[0, ..., 0][:, mask].median(0).indices
            z = [torch.cat(captures[step])[selected, torch.arange(len(selected))] for step in (0, 1)]
            details[name] = {}
            for variable, node in trace_nodes.items():
                positions = torch.nonzero(locations[:, 1] == node).flatten().tolist()
                details[name][variable] = [
                    {"time_index": int(locations[i, 0]),
                     "prediction": float(physical[int(locations[i, 0]), node, 0]),
                     "median_selected_draw": int(selected[i]),
                     "source_unit_z_step1_for_selected_draw": float(z[0][i]),
                     "source_unit_z_step2_for_selected_draw": float(z[1][i])}
                    for i in positions]
        record = {"patient_id": patient, "variable_entries": {
            variable: {variant: entries[variable] for variant, entries in details.items()}
            for variable in variables}}
        if "AST" in variables:
            record["AST_entries"] = record["variable_entries"]["AST"]
        patient_records.append(record)
        print(json.dumps({"patient": patient, "variants_complete": len(variants),
                          "stored_parent_and_candidate_predictions_exact": True}), flush=True)
    arrays = {name: np.stack(values) for name, values in predictions.items()}
    arrays.update(mask=np.stack(masks), window_ids=np.asarray(ids),
                  bias=dataset.bias.numpy(), scale=dataset.scale.numpy())
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    report = {"freeze": str(args.freeze), "freeze_sha256": sha256(args.freeze),
              "recipe_sha256": freeze["recipe_sha256"], "parent_sha256": freeze["reference_recipe_sha256"],
              "script_sha256": sha256(Path(__file__)), "selection": selection,
              "selection_sha256": sha256(args.selection), "split": "calibration",
              "samples": 100, "seed": 20261103, "chunk": 20, "sigmas": [80, 20.92],
              "source_unit": unit, "traced_variables": variables,
              "parent_and_candidate_exact_stored_prediction_parity": True,
              "non_head_weights_and_graph_factors_identical": True,
              "patient_records": patient_records, "raw_sha256": sha256(args.output.with_suffix(".npz")),
              "hidden_targets_or_residuals_scored_by_probe": False, "gradients_or_fitting_used": False,
              "scope": "Retrospective mechanism study on named calibration cases. Step-only substitutions are diagnostic interventions with frozen head weights, not new candidates or benchmark scores. No checkpoint export, coefficient tuning, test access or promotion follows from this probe."}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "patients": ids,
                      "traced_variables": variables,
                      "ordinary_parent_and_candidate_predictions_exact": True}, indent=2))


if __name__ == "__main__":
    main()
