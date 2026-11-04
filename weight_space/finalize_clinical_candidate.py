"""Verify a frozen clinical candidate's completed test and exported checkpoint.

This audits existing measurements and performs a small CPU loading-parity
check. It never selects or changes weights, launches a test, or closes a goal.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_checkpoint, sha256
from .data import load_data
from .evaluation_audit import audit_raw
from .finalize_clinical import read, spanish, state_hash, variable_groups
from .run import Inference
from .summarize_test import summarize


def reproduce_report(path):
    report = read(path)
    pairs = [(row["original"], row["edited"]) for row in report["evidence"]]
    for row in report["evidence"]:
        for key in ("original", "edited"):
            assert sha256(row[key]) == row[key + "_sha256"]
    equivalences = [(row["reference"], row["native_subset"])
                    for row in report.get("runtime_equivalence_evidence", [])]
    assert summarize(pairs, 1, equivalences) == report
    assert report["dataset"] == "mimic-iii-challenge_point" and report["replicate"] == 0
    assert report["split"] == "test" and report["windows_per_seed"] == 4033
    assert report["samples_per_seed"] == 100 and report["seeds"] == [20261103, 20261104]
    return report, pairs


def retained_reference_record(label, manifest_path, verification_path, report_path):
    """Build a reference binding for a prospective pre-test receipt, not a test."""
    paths = {"manifest": Path(manifest_path), "verification": Path(verification_path),
             "test_report": Path(report_path), "checkpoint": Path(manifest_path).with_suffix(".ckpt")}
    manifest = read(paths["manifest"])
    binding = {"label": label, "recipe_name": manifest["edit"]["name"]}
    for key, path in paths.items():
        binding[key] = str(path)
        binding[key + "_sha256"] = sha256(path)
    return binding


def verify_retained_reference(binding, source_sha256, before_time):
    """Bind an earlier, actually verified export to its exact paired test files."""
    paths = {key: Path(binding[key]) for key in ("manifest", "checkpoint", "verification", "test_report")}
    for key, path in paths.items():
        assert sha256(path) == binding[key + "_sha256"], key
    assert paths["manifest"].with_suffix(".ckpt").resolve() == paths["checkpoint"].resolve()
    manifest, verification, report = [read(paths[key]) for key in ("manifest", "verification", "test_report")]
    assert manifest["source_sha256"] == source_sha256
    assert manifest["export_sha256"] == binding["checkpoint_sha256"]
    assert manifest["edit"]["name"] == binding["recipe_name"]
    assert manifest["test_comparison"] == report and manifest["roundtrip_prediction_verified"]
    assert manifest["training_or_gradients_used"] is False
    assert verification["verification_passed"] and verification["source_checkpoint_unchanged"]
    assert verification["frozen_state_and_graph_factors_match_export_exactly"]
    assert verification["cli_loading_parity"]["all_prediction_target_mask_scaler_and_id_arrays_exactly_equal"]
    assert verification["time_unix"] < before_time
    assert verification["metrics_vs_original"] == report["metrics"]
    audited = {Path(path).resolve(): digest for path, digest in verification["artifact_sha256"].items()}
    for key in ("manifest", "checkpoint", "test_report"):
        assert audited[paths[key].resolve()] == binding[key + "_sha256"]
    assert report["dataset"] == "mimic-iii-challenge_point" and report["replicate"] == 0
    assert report["split"] == "test" and report["windows_per_seed"] == 4033
    assert report["samples_per_seed"] == 100 and report["seeds"] == [20261103, 20261104]
    assert report["edit"] == manifest["edit"] and report["balanced_improvement_established"]
    assert all(seed["metrics"][metric]["change_percent"] < 0
               for seed in report["per_seed"] for metric in ("mae", "mse"))
    references = set()
    raw_audits = {Path(row["result"]).resolve(): row for row in verification["full_test_raw_audits"]}
    for row in report["evidence"]:
        for key in ("original", "edited"):
            path = Path(row[key])
            assert sha256(path) == row[key + "_sha256"]
            raw = raw_audits[path.resolve()]
            assert raw["json_sha256"] == row[key + "_sha256"]
            assert sha256(path.with_suffix(".npz")) == raw["npz_sha256"]
        candidate = read(row["edited"])
        assert candidate["edit"] == manifest["edit"] and candidate["checkpoint_sha256"] == source_sha256
        assert candidate["sigmas"] == [80, 20.92]
        references.add(Path(row["edited"]).resolve())
    return manifest, references


def start_retained_reference(started, freeze):
    """Only the real V4 receipt may use the older implicit retained-V1 schema."""
    binding = started.get("retained_reference")
    if binding is None:
        assert freeze["recipe_sha256"] == "3415dba55842fb889b7e16021f2275f6bede2a85a2462448edc253ed9279341e"
        assert sha256("artifacts/clinical_relative_tail_test_start.json") == "1a06e6a61ed671b5103e917addeda382da069ef8f2af87a9ae0cf49a225000ca"
        assert started == read("artifacts/clinical_relative_tail_test_start.json")
        binding = retained_reference_record(
            "V1", "artifacts/edited_weights/clinical_v1/ctimba_mimic-iii-challenge_point/version_0/weights.json",
            "artifacts/audit/clinical_candidate_verifier_v1_recheck_01/verification.json",
            "artifacts/clinical_point_final_results.json")
    manifest, references = verify_retained_reference(binding, freeze["source_sha256"], started["time_unix"])
    assert binding["recipe_name"] in started["prior_clinical_test_candidate_recipes"]
    assert binding["recipe_name"] != read(freeze["recipe"])["name"]
    return binding, manifest, references


def verify_start_contract(started, freeze, freeze_path, recipe):
    """Verify the real pre-test contract without pretending outputs exist yet."""
    assert freeze["time_unix"] < started["time_unix"]
    assert freeze["recipe_sha256"] == started["frozen_recipe_sha256"]
    historical_v1 = started.get("prior_clinical_test_candidate_evaluations") == 0
    if historical_v1:
        # Only the real, immutable first V1 experiment uses this old schema.
        assert freeze["recipe_sha256"] == "e9d13ffb880a9ed02dbda070b7456f2c334d02a9b5bb7455c674c021d7e4b0fa"
        plan_path = Path("artifacts/plans/clinical_point_full_test_v1.json")
        for key in ("confirmation_original", "confirmation_edited"):
            assert sha256(started[key]) == started[key + "_sha256"]
        benchmark_reuse = False
    else:
        assert started["benchmark_reuse"] is True
        assert "clinical_v0_affine_tail_readout" in started["prior_clinical_test_candidate_recipes"]
        assert Path(started["candidate_freeze"]).resolve() == freeze_path.resolve()
        assert sha256(freeze_path) == started["candidate_freeze_sha256"]
        plan_path = Path(started["test_plan"])
        confirmation_path = Path(started["selection_confirmation_report"])
        assert sha256(confirmation_path) == started["selection_confirmation_report_sha256"]
        confirmation = read(confirmation_path)
        assert confirmation["split"] == "confirmation" and confirmation["full_partition_verified"]
        assert confirmation["patients_per_seed"] == 2016 and confirmation["samples"] == 100
        assert confirmation["seeds"] == [20261103, 20261104]
        assert confirmation["candidate_recipe"] == recipe["name"]
        assert confirmation["both_mean_metrics_improve"] and confirmation["both_metrics_improve_on_each_seed"]
        from .clinical_partition_report import report_pairs
        confirmation_pairs = [(Path(row["reference"]), Path(row["candidate"]))
                              for row in confirmation["raw_audits"]]
        equivalences = confirmation.get("runtime_equivalence_evidence", [])
        assert len(equivalences) <= 1
        bridge = Path(equivalences[0]["native_subset"]) if equivalences else None
        assert report_pairs(confirmation_pairs, require_full=True, native_reference_subset=bridge) == confirmation
        binding, retained_manifest, _ = start_retained_reference(started, freeze)
        assert confirmation["reference_recipe"] == binding["recipe_name"]
        assert all(read(a)["edit"] == retained_manifest["edit"] for a, b in confirmation_pairs)
        benchmark_reuse = True
    assert sha256(plan_path) == started["test_plan_sha256"]
    plan = read(plan_path)
    assert plan["recipe_sha256"] == freeze["recipe_sha256"]
    jobs = plan["jobs"]
    assert len(jobs) == 2 and {job["seed"] for job in jobs} == {20261103, 20261104}
    expected = set()
    for job in jobs:
        assert job["dataset"] == "mimic-iii-challenge_point" and job["replicate"] == 0
        assert job["split"] == "test" and job["samples"] == 100 and "limit" not in job
        assert job["chunk"] == 100 and job["batch"] == 1
        assert read(job["edit"]) == recipe
        expected.add(Path(job.get("output_edited", job.get("output"))).with_suffix(".json").resolve())
    return benchmark_reuse, expected


def verify_start(started, freeze, freeze_path, pairs, recipe):
    benchmark_reuse, expected = verify_start_contract(started, freeze, freeze_path, recipe)
    assert expected == {Path(b).resolve() for a, b in pairs}
    assert all(Path(b).stat().st_mtime >= started["time_unix"] for a, b in pairs)
    return benchmark_reuse


def verify_cli(source, export, recipe_path, output_dir):
    """Use the actual public CLI in two fresh CPU processes, without a GPU job."""
    outputs = []
    for label, checkpoint, edit in (("source_recipe", source, recipe_path), ("export", export, None)):
        assert list(checkpoint.parent.glob("*.ckpt")) == [checkpoint]
        prefix = output_dir / (label + "_cli")
        command = [sys.executable, "-m", "weight_space.run", "--dataset", "mimic-iii-challenge_point",
                   "--replicate", "0", "--split", "calibration", "--limit", "1", "--samples", "3",
                   "--sigmas", "80", "20.92", "--seed", "73", "--chunk", "3", "--batch", "1",
                   "--device", "cpu", "--backend", "reference", "--weights", str(checkpoint.parent.parent.parent),
                   "--output", str(prefix)]
        if edit is not None:
            command += ["--edit", str(edit)]
        with (output_dir / (label + "_cli.log")).open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = read(prefix.with_suffix(".json"))
        assert result["checkpoint_sha256"] == sha256(checkpoint)
        assert result["samples"] == 3 and result["split"] == "calibration" and len(result["window_ids"]) == 1
        outputs.append(prefix)
    with np.load(outputs[0].with_suffix(".npz")) as a, np.load(outputs[1].with_suffix(".npz")) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            assert np.array_equal(a[key], b[key]), key
    return {"source_result": str(outputs[0].with_suffix(".json")),
            "export_result": str(outputs[1].with_suffix(".json")),
            "all_prediction_target_mask_scaler_and_id_arrays_exactly_equal": True,
            "scope": "Actual CLI loading parity on one calibration patient, three draws and a separate seed. This checks serialization/loading; full benchmark accuracy comes exclusively from the separately audited complete tests."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--retained-comparison", type=Path,
                        help="Complete test comparison to the previous retained checkpoint, required after V1")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--test-start", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new audit directory; preserve all previous deliverables")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    freeze, started = read(args.freeze), read(args.test_start)
    source, export = Path(freeze["source_checkpoint"]), args.checkpoint
    recipe_path = Path(freeze["recipe"])
    recipe, manifest = read(recipe_path), read(export.with_suffix(".json"))
    assert sha256(source) == freeze["source_sha256"]
    assert sha256(recipe_path) == freeze["recipe_sha256"]
    assert sha256(export) == manifest["export_sha256"]
    summary, pairs = reproduce_report(args.report)
    assert summary == manifest["test_comparison"] and summary["edit"] == recipe == manifest["edit"]
    assert summary["balanced_improvement_established"]
    assert all(all(seed["metrics"][metric]["change_percent"] < 0 for metric in ("mae", "mse"))
               for seed in summary["per_seed"])
    benchmark_reuse = verify_start(started, freeze, args.freeze, pairs, recipe)
    retained, retained_binding = None, None
    if benchmark_reuse:
        assert args.retained_comparison is not None, "A new best checkpoint must also be compared to the retained version"
    if args.retained_comparison:
        retained, retained_pairs = reproduce_report(args.retained_comparison)
        assert retained["edit"] == recipe
        assert {str(Path(b).resolve()) for a, b in retained_pairs} == {str(Path(b).resolve()) for a, b in pairs}
        assert all(retained["metrics"][metric]["change_percent"] < 0 for metric in ("mae", "mse"))
        assert all(all(seed["metrics"][metric]["change_percent"] < 0 for metric in ("mae", "mse"))
                   for seed in retained["per_seed"])
        retained_binding, retained_manifest, expected_references = start_retained_reference(started, freeze)
        assert {Path(a).resolve() for a, b in retained_pairs} == expected_references
        assert all(read(a)["edit"] == retained_manifest["edit"] for a, b in retained_pairs)
    expected_ids = load_data("mimic-iii-challenge_point").indices("test")
    frozen_sources = {str(Path(k).resolve()): v for k, v in freeze["source_sha256_files"].items()}
    core_sources = {k: v for k, v in frozen_sources.items() if Path(k).name != "paired_head.py"}
    loaded, raw_audits = [], []
    invariants = None
    for a_path, b_path in pairs:
        reports = [read(a_path), read(b_path)]
        assert not reports[0]["edit"]["operations"], "The main report must compare to original CoSTI"
        assert reports[1]["edit"] == recipe
        for path, report in zip((Path(a_path), Path(b_path)), reports):
            assert report["sigmas"] == [80, 20.92] and report["checkpoint_sha256"] == freeze["source_sha256"]
            actual = {str(Path(k).resolve()): v for k, v in report["runtime"]["source_sha256"].items()}
            assert actual == (frozen_sources if report["runtime"]["backend"].endswith("shared-head") else core_sources)
            arrays = audit_raw(path, report)
            assert np.array_equal(arrays["window_ids"], expected_ids)
            fixed = {k: arrays[k] for k in ("window_ids", "target", "mask", "bias", "scale")}
            if invariants is not None:
                assert all(np.array_equal(v, invariants[k]) for k, v in fixed.items())
            invariants = fixed
            error = (arrays["prediction"] - arrays["target"]).astype(np.float64)
            mask = arrays["mask"]
            counts = mask.sum(axis=(0, 1, 3))
            assert np.array_equal(counts, [row["n"] for row in report["per_node"]])
            valid = counts > 0
            for metric, losses in (("mae", np.abs(error)), ("mse", error ** 2)):
                computed = np.where(mask, losses, 0).sum(axis=(0, 1, 3))[valid] / counts[valid]
                recorded = [row[metric] for row in report["per_node"] if row["n"]]
                np.testing.assert_allclose(computed, recorded, rtol=1e-12, atol=1e-12)
            raw_audits.append({"result": str(path), "json_sha256": sha256(path),
                               "npz_sha256": sha256(path.with_suffix(".npz")),
                               "global_patient_and_variable_metrics_recomputed": True})
        loaded.append(reports)
    for path, digest in freeze["source_sha256_files"].items():
        assert sha256(path) == digest
    rebuilt, restored = Inference(source, 0, recipe), Inference(export, 0)
    assert state_hash(rebuilt.model) == state_hash(restored.model) == freeze["edited_state_sha256"]
    assert all(torch.equal(a, b) for a, b in zip(rebuilt.model.input_sfe.support[-1], restored.model.input_sfe.support[-1]))
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in restored.model.parameters())
    assert manifest["roundtrip_prediction_verified"] and manifest["training_or_gradients_used"] is False
    original_state = load_checkpoint(source)["state_dict"]
    changes = {key: int((value.cpu() != original_state["model." + key]).sum())
               for key, value in rebuilt.model.state_dict().items() if not torch.equal(value.cpu(), original_state["model." + key])}
    assert len(changes) == manifest["changed_tensors"] and sum(changes.values()) == manifest["changed_values"]
    names = read("artifacts/audit/clinical_variable_names.json")
    groups = variable_groups(loaded, names)
    args.output_dir.mkdir(parents=True)
    cli = verify_cli(source, export, recipe_path, args.output_dir)
    groups_path = args.output_dir / "variable_groups.json"
    groups_path.write_text(json.dumps(groups, indent=2) + "\n")
    variables = []
    for index, name in enumerate(names):
        n = loaded[0][0]["per_node"][index]["n"]
        row = {"variable": name, "scored_values_per_seed": n}
        for metric in ("mae", "mse"):
            a, b = [float(np.mean([pair[side]["per_node"][index][metric] for pair in loaded])) if n else None for side in (0, 1)]
            row.update({metric + "_original": a, metric + "_edited": b,
                        metric + "_change_percent": 100 * (b / a - 1) if a else None})
        variables.append(row)
    variables_path = args.output_dir / "per_variable.csv"
    with variables_path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(variables[0])); writer.writeheader(); writer.writerows(variables)
    lines = ["CoSTI · PhysioNet Challenge 2019 · point missing · version_0", "",
             "Esta auditoría verifica evaluaciones ya guardadas; no ejecuta un test nuevo.",
             "Test completo: 4033 pacientes, 100 imputaciones, semillas 20261103 y 20261104.",
             "Media de dos semillas del mismo checkpoint; no es una media de cinco checkpoints."]
    if benchmark_reuse:
        lines += ["El benchmark ya se había evaluado con V1. Esta evaluación reutiliza el benchmark; no representa un test intacto nuevo."]
    retained_label = retained_binding["label"] if retained_binding else "anterior"
    for label, report in (("Frente a CoSTI original", summary), (f"Frente al checkpoint retenido {retained_label}", retained)):
        if report is None:
            continue
        lines += ["", label + ":"]
        for metric in ("mae", "mse"):
            row = report["metrics"][metric]
            lines += [f"{metric.upper()}: {spanish(row['original'])} -> {spanish(row['edited'])}; cambio {spanish(row['change_percent'])}%.",
                      f"IC95% de la diferencia por paciente: {row['delta_ci95_patient_bootstrap']}."]
    lines += ["", f"Valores modificados: {sum(changes.values())}, en {len(changes)} tensores.",
              "El estado completo y los factores del grafo exportados coinciden exactamente con el candidato congelado.",
              "La carga real desde la CLI reproduce todas las predicciones de la comprobación en CPU.",
              "No se usaron entrenamiento, optimizadores, backpropagation ni información de gradientes.",
              "Las métricas por variable y grupo son descriptivas. La mejora agregada no implica mejora en todas las variables.",
              "El bootstrap por paciente promedia primero las semillas y no cubre toda la variación entre semillas.",
              "Checkpoint: " + str(export.resolve()), "SHA256: " + sha256(export)]
    text_path = args.output_dir / "resultado.txt"
    text_path.write_text("\n".join(lines) + "\n")
    assert sha256(source) == freeze["source_sha256"]
    artifact_paths = [args.report, args.freeze, args.test_start, export, export.with_suffix(".json"),
                      groups_path, variables_path, text_path]
    if args.retained_comparison:
        artifact_paths.append(args.retained_comparison)
    audit = {"time_unix": time.time(), "verification_passed": True, "goal_status_not_changed_by_this_script": True,
             "audits_existing_test_measurements_only": True,
             "benchmark_reuse": benchmark_reuse, "source_checkpoint_unchanged": True,
             "frozen_state_and_graph_factors_match_export_exactly": True,
             "changed_values": sum(changes.values()), "changes_by_tensor": changes,
             "full_test_raw_audits": raw_audits, "cli_loading_parity": cli,
             "metrics_vs_original": summary["metrics"], "metrics_vs_retained": retained["metrics"] if retained else None,
             "retained_reference": retained_binding,
             "verifier_sha256": sha256(Path(__file__)),
             "artifact_sha256": {str(path): sha256(path) for path in artifact_paths}}
    audit_path = args.output_dir / "verification.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"audit": str(audit_path), "report": str(text_path),
                      "changed_values": sum(changes.values()), "metrics": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
