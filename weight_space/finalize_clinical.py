"""Audit completed clinical evaluations and write the final local deliverables.

This performs no weight selection or optimization. It only verifies the
previously frozen candidate, already completed measurements, and its export.
"""

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoints import sha256
from .run import Inference
from .summarize_test import summarize


def read(path):
    return json.loads(Path(path).read_text())


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def spanish(value):
    return f"{value:.6f}".replace(".", ",")


def variable_groups(loaded, names):
    """Describe the frozen result; these post-test groups do not select weights."""
    assert len(names) == 40 and names[34:] == ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
    counts = np.array([row["n"] for row in loaded[0][0]["per_node"]])
    for pair in loaded:
        for report in pair:
            assert np.array_equal(counts, [row["n"] for row in report["per_node"]])
            assert counts.sum() == report["metrics"]["n"]
            for metric in ("mae", "mse"):
                total = sum(row["n"] * row[metric] for row in report["per_node"] if row["n"])
                assert np.isclose(total / counts.sum(), report["metrics"][metric], rtol=1e-12, atol=1e-12)
    groups = {}
    for label, ids in (("measured_physiological_variables", range(34)),
                       ("static_administrative_and_time_variables", range(34, 40))):
        n = int(counts[list(ids)].sum())
        result = {"variables": [names[i] for i in ids], "scored_values_per_seed": n}
        for metric in ("mae", "mse"):
            a, b = [float(np.mean([
                sum(pair[side]["per_node"][i]["n"] * pair[side]["per_node"][i][metric]
                    for i in ids if counts[i]) / n for pair in loaded])) for side in (0, 1)]
            overall_reduction = float(np.mean([
                pair[0]["metrics"][metric] - pair[1]["metrics"][metric] for pair in loaded])) * counts.sum()
            result[metric] = {"original": a, "edited": b, "change_percent": 100 * (b / a - 1),
                              "share_of_total_error_reduction_percent": 100 * (a - b) * n / overall_reduction}
        groups[label] = result
    return {"split": "test", "patients": len(loaded[0][0]["window_ids"]),
            "seeds": [pair[0]["seed"] for pair in loaded], "used_for_weight_selection": False,
            "uncertainty": "Descriptive point estimates; no group-level confidence claim.", "groups": groups}


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    final_path = Path("artifacts/clinical_point_final_results.json")
    freeze_path = Path("artifacts/clinical_point_candidate_freeze_v1.json")
    test_start_path = Path("artifacts/clinical_point_test_start_v1.json")
    export = Path("artifacts/edited_weights/clinical_v1/ctimba_mimic-iii-challenge_point/version_0/weights.ckpt")
    audit_path = Path("artifacts/clinical_completion_audit.json")
    text_path = Path("artifacts/clinical_point_resultado.txt")
    if audit_path.exists() or text_path.exists():
        raise FileExistsError("Final deliverables are immutable")
    summary, freeze, started = map(read, (final_path, freeze_path, test_start_path))
    manifest = read(export.with_suffix(".json"))
    assert sha256(export) == manifest["export_sha256"]
    assert sha256(freeze["source_checkpoint"]) == freeze["source_sha256"]
    assert sha256(freeze["recipe"]) == freeze["recipe_sha256"] == started["frozen_recipe_sha256"]
    assert freeze["time_unix"] < started["time_unix"]
    assert started["prior_clinical_test_candidate_evaluations"] == 0
    assert sha256("artifacts/plans/clinical_point_full_test_v1.json") == started["test_plan_sha256"]
    for label in ("original", "edited"):
        assert sha256(started["confirmation_" + label]) == started["confirmation_" + label + "_sha256"]
    pairs = [(r["original"], r["edited"]) for r in summary["evidence"]]
    assert len(pairs) == 2 and summary["seeds"] == [20261103, 20261104]
    assert summary["block_size_windows"] == 1
    assert summarize(pairs, 1) == summary == manifest["test_comparison"]
    assert summary["balanced_improvement_established"]
    assert all(summary["metrics"][m]["edited"] < summary["metrics"][m]["original"] for m in ("mae", "mse"))
    expected_sources = {str(Path(k).resolve()): v for k, v in freeze["source_sha256_files"].items()}
    loaded, raw_checks = [], []
    reference_mask, reference_target = None, None
    for original_path, edited_path in pairs:
        reports = [read(original_path), read(edited_path)]
        assert reports[1]["edit"] == read(freeze["recipe"])
        with np.load(Path(original_path).with_suffix(".npz")) as a, np.load(Path(edited_path).with_suffix(".npz")) as b:
            for key in ("target", "mask", "window_ids", "bias", "scale"):
                assert np.array_equal(a[key], b[key]), key
            assert np.array_equal(a["window_ids"], np.arange(36303, 40336))
            if reference_mask is None:
                reference_mask, reference_target = a["mask"], a["target"]
            else:
                assert np.array_equal(a["mask"], reference_mask)
                assert np.array_equal(a["target"], reference_target)
            for report, array in zip(reports, (a, b)):
                assert report["samples"] == 100 and report["split"] == "test"
                assert report["sigmas"] == [80, 20.92]
                assert report["checkpoint_sha256"] == freeze["source_sha256"]
                actual_sources = {str(Path(k).resolve()): v for k, v in report["runtime"]["source_sha256"].items()}
                assert actual_sources == expected_sources
                mask = array["mask"]
                full_error = (array["prediction"] - array["target"]).astype(np.float64)
                error = full_error[mask]
                assert np.isfinite(error).all() and len(error) == report["metrics"]["n"]
                computed = {"mae": float(np.abs(error).mean()), "mse": float(np.square(error).mean())}
                for key, value in computed.items():
                    assert np.isclose(value, report["metrics"][key], rtol=1e-12, atol=1e-12)
                for rows, axes in ((report["per_window"], (1, 2, 3)),
                                   (report["per_node"], (0, 1, 3))):
                    counts = mask.sum(axis=axes)
                    assert np.array_equal(counts, [row["n"] for row in rows])
                    valid = counts > 0
                    for metric, losses in (("mae", np.abs(full_error)), ("mse", np.square(full_error))):
                        recomputed = np.where(mask, losses, 0).sum(axis=axes)[valid] / counts[valid]
                        recorded = [row[metric] for row in rows if row["n"]]
                        np.testing.assert_allclose(recomputed, recorded, rtol=1e-12, atol=1e-12)
                raw_checks.append({"seed": report["seed"], "edit": report["edit"]["name"], **computed})
        loaded.append(reports)
    for name, expected in freeze["source_sha256_files"].items():
        assert sha256(name) == expected
    rebuilt = Inference(freeze["source_checkpoint"], 0, read(freeze["recipe"]))
    restored = Inference(export, 0)
    assert state_hash(rebuilt.model) == state_hash(restored.model) == freeze["edited_state_sha256"]
    assert all(torch.equal(a, b) for a, b in zip(rebuilt.model.input_sfe.support[-1], restored.model.input_sfe.support[-1]))
    assert all(not p.requires_grad and p.grad is None for p in restored.model.parameters())
    assert manifest["roundtrip_prediction_verified"] and manifest["training_or_gradients_used"] is False
    assert manifest["changed_tensors"] == 3 and manifest["changed_values"] == 193
    names = read("artifacts/audit/clinical_variable_names.json")
    variables = []
    for index, name in enumerate(names):
        row = {"variable": name, "scored_values_per_seed": loaded[0][0]["per_node"][index]["n"]}
        for metric in ("mae", "mse"):
            if not row["scored_values_per_seed"]:
                row.update({metric + "_original": None, metric + "_edited": None,
                            metric + "_change_percent": None})
                continue
            left = float(np.mean([a["per_node"][index][metric] for a, _ in loaded]))
            right = float(np.mean([b["per_node"][index][metric] for _, b in loaded]))
            row.update({metric + "_original": left, metric + "_edited": right,
                        metric + "_change_percent": 100 * (right / left - 1) if left else None})
        variables.append(row)
    variable_path = Path("artifacts/clinical_point_test_per_variable.csv")
    with variable_path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(variables[0]))
        writer.writeheader(); writer.writerows(variables)
    groups = variable_groups(loaded, names)
    group_path = Path("artifacts/clinical_point_test_variable_groups.json")
    group_path.write_text(json.dumps(groups, indent=2) + "\n")
    ledger = []
    test_candidates = set()
    for path in sorted(Path("artifacts/results").glob("*.json")):
        result = read(path)
        if not isinstance(result, dict) or result.get("dataset") != "mimic-iii-challenge_point" or result.get("replicate") != 0:
            continue
        if "checkpoint_sha256" not in result or "runtime" not in result:
            continue
        ledger.append({"result": str(path), "edit": result["edit"]["name"], "split": result["split"],
                       "patients": len(result["window_ids"]), "samples": result["samples"], "seed": result["seed"],
                       "mae": result["metrics"]["mae"], "mse": result["metrics"]["mse"],
                       "runtime": result["runtime"]["backend"]})
        if result["split"] == "test" and result["edit"]["operations"]:
            test_candidates.add(result["edit"]["name"])
    assert test_candidates == {summary["edit"]["name"]}
    with Path("artifacts/clinical_point_trial_ledger.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ledger[0]))
        writer.writeheader(); writer.writerows(ledger)
    negative = []
    for metric in ("mae", "mse"):
        if summary["metrics"][metric]["delta_ci95_patient_bootstrap"][1] >= 0:
            negative.append(metric.upper())
    lines = ["CoSTI PhysioNet Challenge 2019, point missing, version_0", "",
             "Comparación final: 4033 pacientes de test, 100 imputaciones y semillas 20261103 y 20261104.",
             "Los valores siguientes son la media de las dos evaluaciones emparejadas del mismo checkpoint.",
             "No son la media de cinco checkpoints publicada en el artículo.", ""]
    scored_patients = sum(row["n"] > 0 for row in loaded[0][0]["per_window"])
    lines += [f"Hay {summary['scored_entries_per_seed']} valores evaluables por semilla, distribuidos en {scored_patients} pacientes.",
              f"Los {summary['windows_per_seed'] - scored_patients} pacientes restantes se incluyen en la pasada completa, pero la máscara no contiene valores evaluables y no aportan error.", ""]
    for metric in ("mae", "mse"):
        row = summary["metrics"][metric]
        lines += [f"{metric.upper()}: {spanish(row['original'])} -> {spanish(row['edited'])}; reducción {spanish(-row['change_percent'])}%.",
                  f"IC95% de la diferencia editado-original, bootstrap por paciente: {row['delta_ci95_patient_bootstrap']}."]
    lines += ["", "Resultados por semilla:"]
    for a, b in loaded:
        lines.append(f"Semilla {a['seed']}: MAE {spanish(a['metrics']['mae'])} -> {spanish(b['metrics']['mae'])}; MSE {spanish(a['metrics']['mse'])} -> {spanish(b['metrics']['mse'])}.")
    lines += ["", "Desglose descriptivo por tipo de variable; no se utilizó para seleccionar pesos:"]
    for key, label in (("measured_physiological_variables", "Mediciones fisiológicas"),
                       ("static_administrative_and_time_variables", "Variables estáticas, administrativas y temporales")):
        row = groups["groups"][key]
        lines.append(label + ": " + "; ".join(
            f"{m.upper()} {spanish(row[m]['original'])} -> {spanish(row[m]['edited'])} ({spanish(row[m]['change_percent'])}%)"
            for m in ("mae", "mse")) + ".")
    administrative_share = groups["groups"]["static_administrative_and_time_variables"]["mae"]["share_of_total_error_reduction_percent"]
    lines += [f"El {spanish(administrative_share)}% de la reducción total del error absoluto procede de variables estáticas, administrativas y temporales.",
              "Estas cifras por grupo son descriptivas y no incluyen un intervalo de confianza propio."]
    lines += ["", "Cambio: 193 valores en tres tensores de la capa de salida, de un total de 2.406.697 valores.",
              "Corrección de ganancia y sesgo, seguida de un refuerzo del 1% de las vías positivas de las neuronas 38 y 198.",
              "No se usaron entrenamiento, optimizadores, backpropagation ni información de gradientes.",
              "El checkpoint original conserva su SHA256. Los pesos exportados reproducen exactamente el estado fijado antes del test.",
              "Una corrección adicional de la neurona 20 empeoró MAE y MSE en ambas semillas de confirmación y se descartó antes del test.",
              "Otra corrección localizada de la neurona 198 mejoró la calibración, pero empeoró MAE en ambas semillas de confirmación. Aunque su MSE medio bajó, no cumplió el criterio previo de mejorar ambas métricas frente al candidato seleccionado; también se descartó.",
              "Sólo este candidato se evaluó sobre el test clínico; todos los intentos anteriores se conservan en el registro CSV.",
              "La mejora es agregada: algunas variables pueden empeorar. El CSV por variable permite revisar esas diferencias.",
              "El bootstrap agrupa cada paciente y promedia las semillas; no cubre toda la posible variación entre semillas."]
    if negative:
        lines.append("La reducción numérica de " + ", ".join(negative) + " no queda establecida por su intervalo de confianza: incluye cero.")
    lines += ["", "Checkpoint: " + str(export.resolve()), "SHA256: " + manifest["export_sha256"],
              "Resultados y evidencia: " + str(final_path.resolve()),
              "Detalle por variable: " + str(variable_path.resolve()), "",
              "Ejemplo de carga y evaluación local, desde <HOME>/Projects/CoSTI-weight-space:",
              ".venv/bin/python -m weight_space.run --dataset mimic-iii-challenge_point --replicate 0 --split calibration --limit 1 --samples 3 --seed 73 --chunk 3 --batch 1 --device cpu --weights artifacts/edited_weights/clinical_v1 --output artifacts/reproduction/clinical_export_smoke", ""]
    text_path.write_text("\n".join(lines))
    audit = {"time_unix": time.time(), "verification_passed": True,
             "goal_status_not_changed_by_this_script": True, "full_test_metrics_recomputed_from_arrays": raw_checks,
             "two_complete_seeds": True, "same_masks_and_targets_across_models_and_seeds": True,
             "same_frozen_inference_sources": True,
             "per_patient_and_variable_metrics_verified_against_arrays": True,
             "source_checkpoint_unchanged": True, "export_matches_frozen_state_exactly": True,
             "persisted_graph_factors_match": True, "export_sha256": sha256(export),
             "changed_values": 193, "changed_tensors": 3, "clinical_test_candidates": sorted(test_candidates),
             "joint_improvement_on_each_seed": all(all(b["metrics"][m] < a["metrics"][m] for m in ("mae", "mse")) for a, b in loaded),
             "metrics": summary["metrics"], "uncertain_metric_intervals": negative,
             "completed_evaluations_retained": len(ledger),
             "descriptive_variable_groups": str(group_path),
             "deliverable_sha256": {str(p): sha256(p) for p in (export, final_path, freeze_path, test_start_path, text_path, variable_path, group_path)}}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"audit": str(audit_path), "report": str(text_path), "metrics": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
