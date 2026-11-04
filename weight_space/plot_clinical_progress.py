"""Plot comparable clinical weight-edit trials and the current validation stage."""

import argparse
import csv
import hashlib
import html
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path("artifacts/results")


def refresh_latest(prefix):
    """Keep one stable local link while preserving every dated snapshot."""
    aliases = []
    for extension in (".png", ".svg", ".pdf", ".html", ".json", ".csv"):
        target = prefix.with_suffix(extension)
        alias = prefix.with_name("clinical_evolution_latest").with_suffix(extension)
        if alias == target or (alias.exists() and not alias.is_symlink()):
            raise FileExistsError(alias)
        temporary = alias.with_name(alias.name + ".next")
        if temporary.exists() or temporary.is_symlink():
            raise FileExistsError(temporary)
        temporary.symlink_to(target.name)
        temporary.replace(alias)
        aliases.append(str(alias))
    return aliases


def load(stem):
    return json.loads((ROOT / (stem + ".json")).read_text())


def paired(stems):
    records = [(load(a), load(b)) for a, b in stems]
    for a, b in records:
        for field in ("dataset", "replicate", "split", "window_ids", "samples", "sigmas", "seed", "checkpoint_sha256"):
            assert a[field] == b[field], field
        assert a["samples"] == 100
    metrics = {}
    for metric in ("mae", "mse"):
        a = float(np.mean([a["metrics"][metric] for a, _ in records]))
        b = float(np.mean([b["metrics"][metric] for _, b in records]))
        metrics[metric] = {"original": a, "edited": b, "improvement_percent": 100 * (1 - b / a)}
    return {"patients": len(records[0][0]["window_ids"]), "seeds": [a["seed"] for a, _ in records],
            "metrics": metrics, "source_pairs": stems}


def es(value, digits=6):
    return f"{value:.{digits}f}".replace(".", ",")


def progress():
    path = Path("artifacts/logs/clinical_point_v0_targeted_test_full_seed2_paired_paired.log")
    latest = {"done": 0, "total": 4033}
    if path.exists():
        for line in reversed(path.read_text().splitlines()):
            if line.startswith('{"done"'):
                latest = json.loads(line)
                break
    complete = (ROOT / "clinical_point_v0_targeted_test_full_seed2_paired.json").exists()
    verified = Path("artifacts/audit/clinical_export_cli_verification.json").exists()
    if complete:
        latest["done"] = latest["total"]
    if verified:
        label = "HITO V1 VALIDADO · Dos semillas completas y checkpoint verificado; continúa la experimentación"
    elif complete:
        label = "TEST COMPLETO · Pendiente el cierre de la exportación y su verificación"
    else:
        label = f"EN CURSO · Segunda semilla: {latest['done']}/{latest['total']} pacientes ({es(100 * latest['done'] / latest['total'], 1)}%)"
    return {"done": latest["done"], "total": latest["total"], "complete": complete,
            "export_verified": verified, "label": label}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--extra-trials", type=Path, help="Additional measured hypotheses after the original V1 series")
    p.add_argument("--full-candidate-report", type=Path, action="append", default=[],
                   help="Audited full-confirmation comparisons, displayed separately from the retained test result")
    p.add_argument("--retained-test-report", type=Path,
                   help="Complete two-seed test report for a newly retained and exported candidate")
    p.add_argument("--retained-manifest", type=Path, help="Matching exported checkpoint .json manifest")
    p.add_argument("--retained-audit", type=Path, help="Completed export verification; also records benchmark reuse")
    p.add_argument("--retained-label", help="Human-readable label such as V2 or V4")
    p.add_argument("--activity-note", help="Explicit current activity, such as a user-requested pause; does not alter any metric")
    p.add_argument("--wide-snapshot", type=Path,
                   help="Separate audited512-patient calibration snapshot to show in the HTML, never in128-patient curves")
    args = p.parse_args()
    if bool(args.retained_test_report) != bool(args.retained_manifest):
        p.error("--retained-test-report and --retained-manifest must be supplied together")
    if args.retained_label and not args.retained_test_report:
        p.error("--retained-label requires an explicit retained test report")
    if args.retained_audit and not args.retained_test_report:
        p.error("--retained-audit requires an explicit retained test report")
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if any(prefix.with_suffix(ext).exists() for ext in (".png", ".svg", ".pdf", ".html", ".json", ".csv")):
        raise FileExistsError("Use a new prefix to preserve the previous snapshot")
    base_cal = "mimic-iii-challenge_point_v0_base_cal128x100_fused"
    evaluation_sigmas = load(base_cal)["sigmas"]
    base_conf = ["mimic-iii-challenge_point_v0_base_confirm128x100_fused",
                 "clinical_point_v0_base_confirmation128x100_seed2"]
    recipes = [
        ("Original", "original", base_cal, base_conf),
        ("Grafo ×0,5", "descartado", "mimic-iii-challenge_point_v0_graphhalf_cal128x100_gpu", None),
        ("Ganancia y sesgo", "avance", "mimic-iii-challenge_point_v0_affine_cal128x100_fused",
         ["mimic-iii-challenge_point_v0_affine_confirm128x100_fused", "clinical_point_v0_affine_confirmation128x100_seed2"]),
        ("Dos vías positivas", "retenido", "clinical_point_v0_affine_tail_cal128x100_seed1",
         [f"clinical_point_v0_affine_tail_confirmation128x100_seed{i}" for i in (1, 2)]),
        ("Cola negativa", "descartado", "clinical_point_v0_bounded_negative_tail_cal128x100_seed1",
         [f"clinical_point_v0_bounded_negative_tail_confirmation128x100_seed{i}" for i in (1, 2)]),
        ("Tramo intermedio", "descartado", "clinical_point_v0_local_positive_repair_cal128x100_seed1_chunk20",
         [f"clinical_point_v0_local_positive_repair_confirmation128x100_seed{i}_chunk20" for i in (1, 2)]),
    ]
    extra = json.loads(args.extra_trials.read_text()) if args.extra_trials else []
    extra_controls = {}
    for trial in extra:
        extra_controls[len(recipes)] = trial.get("original", "clinical_point_v0_original_cal128x100_seed1_chunk20")
        assert trial.get("calibration") or trial.get("confirmation"), "A plotted trial needs an actual measurement"
        recipes.append((trial["label"], trial["decision"], trial.get("calibration"), trial.get("confirmation")))
    trials = []
    for i, (label, decision, cal, conf) in enumerate(recipes):
        control = extra_controls.get(i) or ("mimic-iii-challenge_point_v0_base_cal128x100_gpu" if i == 1 else
                   "clinical_point_v0_original_cal128x100_seed1_chunk20" if i == 5 else base_cal)
        trials.append({"order": i, "label": label, "decision": decision,
                       "calibration": paired([(control, cal)]) if cal else None,
                       "confirmation": paired(list(zip(base_conf, conf))) if conf else None})
    selected_index, changed_values, retained_label = 3, 193, "V1"
    retained_evidence = None
    benchmark_reuse = False
    selected = trials[selected_index]
    full_confirm = paired([("clinical_point_v0_original_confirmation_full_seed1_paired",
                            "clinical_point_v0_targeted_confirmation_full_seed1_paired")])
    stage = progress()
    test_seeds = (1, 2) if stage["complete"] else (1,)
    test = paired([(f"clinical_point_v0_original_test_full_seed{i}_paired",
                    f"clinical_point_v0_targeted_test_full_seed{i}_paired") for i in test_seeds])
    if args.retained_test_report:
        report = json.loads(args.retained_test_report.read_text())
        manifest = json.loads(args.retained_manifest.read_text())
        checkpoint = args.retained_manifest.with_suffix(".ckpt")
        assert report["dataset"] == "mimic-iii-challenge_point" and report["replicate"] == 0
        assert report["split"] == "test" and report["samples_per_seed"] == 100
        assert report["windows_per_seed"] == 4033 and len(set(report["seeds"])) >= 2
        assert report["balanced_improvement_established"]
        assert manifest["test_comparison"] == report and manifest["edit"] == report["edit"]
        assert manifest["roundtrip_prediction_verified"]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == manifest["export_sha256"]
        for evidence in report["evidence"]:
            for key in ("original", "edited"):
                assert hashlib.sha256(Path(evidence[key]).read_bytes()).hexdigest() == evidence[key + "_sha256"]
            original_result = json.loads(Path(evidence["original"]).read_text())
            assert not original_result["edit"]["operations"], "Table requires the original CoSTI reference"
            assert original_result["sigmas"] == evaluation_sigmas
        matches = [i for i, (_, _, calibration, confirmation) in enumerate(recipes)
                   if load(calibration or confirmation[0])["edit"] == report["edit"]]
        assert len(matches) == 1, "Retained recipe must appear once in the measured trial series"
        selected_index = matches[0]
        selected = trials[selected_index]
        assert selected["confirmation"] is not None
        changed_values = manifest["changed_values"]
        retained_label = args.retained_label or report["edit"]["name"]
        for trial in trials:
            if trial["decision"] == "retenido":
                trial["decision"] = "validado_anterior"
        selected["decision"] = "retenido"
        test = {"patients": report["windows_per_seed"], "seeds": report["seeds"],
                "metrics": {metric: {"original": values["original"], "edited": values["edited"],
                                     "improvement_percent": -values["change_percent"]}
                            for metric, values in report["metrics"].items() if metric in ("mae", "mse")},
                "source_pairs": [(row["original"], row["edited"]) for row in report["evidence"]]}
        stage = {"done": report["windows_per_seed"], "total": report["windows_per_seed"],
                 "complete": True, "export_verified": True,
                 "label": f"HITO {retained_label} VALIDADO · Test completo con {len(report['seeds'])} semillas y checkpoint verificado; continúa la experimentación"}
        retained_evidence = {"report": str(args.retained_test_report),
                             "report_sha256": hashlib.sha256(args.retained_test_report.read_bytes()).hexdigest(),
                             "manifest": str(args.retained_manifest),
                             "manifest_sha256": hashlib.sha256(args.retained_manifest.read_bytes()).hexdigest(),
                             "checkpoint": str(checkpoint), "checkpoint_sha256": manifest["export_sha256"]}
        if args.retained_audit:
            audit = json.loads(args.retained_audit.read_text())
            assert audit["verification_passed"] and audit["source_checkpoint_unchanged"]
            assert audit["metrics_vs_original"] == report["metrics"]
            audited_files = {Path(key).resolve(): value for key, value in audit["artifact_sha256"].items()}
            for path in (args.retained_test_report, args.retained_manifest, checkpoint):
                assert audited_files[path.resolve()] == hashlib.sha256(path.read_bytes()).hexdigest()
            benchmark_reuse = audit["benchmark_reuse"]
            retained_evidence.update(audit=str(args.retained_audit),
                                     audit_sha256=hashlib.sha256(args.retained_audit.read_bytes()).hexdigest(),
                                     benchmark_reuse=benchmark_reuse)
        if selected_index != 3:
            # The historical full-confirmation row belongs to V1. Do not display
            # it as a measurement of a newly retained checkpoint.
            full_confirm = None
    stamp = datetime.now(ZoneInfo("Europe/Madrid")).strftime("%d/%m/%Y · %H:%M %Z")
    validations = []
    if selected["calibration"] is not None:
        validations.append(("Calibración · 128\n1 semilla", selected["calibration"]))
    validations.append(("Confirmación · 128\nmedia de 2 semillas", selected["confirmation"]))
    if full_confirm is not None:
        validations.append(("Confirmación · 2016\n1 semilla", full_confirm))
    validations.append((f"Test · 4033\n{'media de ' + str(len(test['seeds'])) + ' semillas' if stage['complete'] else 'primera semilla'}", test))
    note = ("Original = este checkpoint sin editar, no la media del paper. "
            "Comparaciones con los mismos pacientes y ruido; cohortes distintas no se comparan entre sí.")
    if benchmark_reuse:
        note += " Benchmark reutilizado: V1 ya se había evaluado en este test."
    if args.activity_note:
        stage["label"] = stage["label"].split(";", 1)[0] + "; " + args.activity_note
        stage["activity_note"] = args.activity_note
    full_candidates = []
    labels_by_recipe = {"clinical_v0_affine_tail_readout": "V1",
                        "clinical_v0_midrange_gate47": "Combinación V2",
                        "clinical_v0_composed_gate255_translate": "Combinación + umbral 255",
                        "clinical_v0_relative_tail": "Corrección de valores extremos",
                        "clinical_v0_gate212_shoulder": "Corrección 212 (V7)",
                        "clinical_v0_gate50_midrange": "Rango medio 50 (V9)",
                        "clinical_v0_gate145_inactive_tail": "Cola inactiva 145 (V10)"}
    full_candidate_rows = []
    for path in args.full_candidate_report:
        report = json.loads(path.read_text())
        assert report["split"] == "confirmation" and report["full_partition_verified"]
        assert report["samples"] == 100 and len(report["seeds"]) >= 2
        full_candidates.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "report": report})
        label = (labels_by_recipe.get(report["reference_recipe"], report["reference_recipe"]) + " → "
                 + labels_by_recipe.get(report["candidate_recipe"], report["candidate_recipe"]))
        row = [f"{label}\n{report['patients_per_seed']} pacientes · {len(report['seeds'])} semillas"]
        for metric in ("mae", "mse"):
            values = report["metrics"][metric]
            row += [es(values["reference"]), es(values["candidate"]), es(-values["change_percent"], 4) + "%"]
        full_candidate_rows.append(row)
    exploratory = paired([("mimic-iii-challenge_point_v0_base_cal128x100_gpu",
                           "mimic-iii-challenge_point_v0_graphhalf_cal128x100_gpu")])
    snapshot = {"checkpoint": "ctimba_mimic-iii-challenge_point/version_0", "timestamp": stamp,
                "sigmas": evaluation_sigmas, "inference_steps": len(evaluation_sigmas),
                "selected_changed_values": changed_values, "samples": 100, "trials": trials,
                "current_test": test, "full_confirmation": full_confirm, "stage": stage,
                "interpretation": note,
                "scope_note": "Series of comparable 128-patient, 100-imputation trials. Earlier 24-patient/10-imputation memory exploration and runtime-only repeats are retained in the complete trial ledger, not mixed into these curves.",
                "early_graph_trial": exploratory}
    snapshot["full_candidate_confirmations"] = full_candidates
    wide_links = None
    if args.wide_snapshot:
        wide = json.loads(args.wide_snapshot.read_text())
        assert wide["split"] == "calibration" and wide["patients"] == 512 and wide["samples"] == 100
        assert wide["reference"] == "clinical_v0_composed_gate255_translate"
        assert wide["retained_label"] == retained_label
        for metric in ("mae", "mse"):
            for key in ("original", "edited"):
                assert wide["retained_test_metrics"][metric][key] == test["metrics"][metric][key]
        for extension in (".png", ".html"):
            assert args.wide_snapshot.with_suffix(extension).exists()
        wide_links = {extension: html.escape(os.path.relpath(args.wide_snapshot.with_suffix(extension), prefix.parent), quote=True)
                      for extension in (".png", ".html")}
        snapshot["wide_calibration_snapshot"] = {"path": str(args.wide_snapshot),
                                                   "sha256": hashlib.sha256(args.wide_snapshot.read_bytes()).hexdigest(),
                                                   "scope": "Separate512-patient cohort, compared to V3; not added to128-patient curves."}
    if retained_evidence:
        snapshot["retained_checkpoint_evidence"] = retained_evidence
        snapshot["retained_label"] = retained_label
    prefix.with_suffix(".json").write_text(json.dumps(snapshot, indent=2) + "\n")
    rows = []
    for trial in trials:
        for split in ("calibration", "confirmation"):
            if trial[split] is None:
                continue
            for metric, result in trial[split]["metrics"].items():
                rows.append({"order": trial["order"], "trial": trial["label"], "decision": trial["decision"],
                             "split": split, "patients": trial[split]["patients"],
                             "seeds": ";".join(map(str, trial[split]["seeds"])), "metric": metric, **result})
    with prefix.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.edgecolor": "#cbd5e1",
                         "text.color": "#172b3a", "axes.labelcolor": "#334155",
                         "xtick.color": "#334155", "ytick.color": "#334155"})
    colors = {"calibration": "#6875c5", "confirmation": "#007f78"}
    labels = ["Original", "Grafo ×0,5", "Ganancia\ny sesgo", "Dos vías\npositivas", "Cola\nnegativa", "Tramo\nintermedio"]
    labels += [trial.get("plot_label", trial["label"].replace(" ", "\n", 1)) for trial in extra]
    fig = plt.figure(figsize=(max(16, 1.9 * len(trials)), 10), facecolor="white")
    fig.text(.055, .953, "CoSTI · evolución de la mejora", fontsize=24, weight="bold")
    fig.text(.055, .915, f"PhysioNet Challenge 2019 · point missing · version_0 · {len(evaluation_sigmas)} pasos · edición de pesos, sin entrenamiento", fontsize=12)
    fig.text(.945, .953, stamp, ha="right", fontsize=10, color="#64748b")
    handles = [Line2D([], [], color=colors["calibration"], linestyle="--", marker="o", label="Calibración: 128 pacientes, 1 semilla"),
               Line2D([], [], color=colors["confirmation"], marker="s", label="Confirmación: 128 pacientes, media de 2 semillas"),
               Line2D([], [], color="#bb4845", marker="x", linestyle="None", markersize=9, markeredgewidth=2, label="Propuesta descartada")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.05, .886), ncol=3, frameon=False, fontsize=10)
    for j, metric in enumerate(("mae", "mse")):
        ax = fig.add_axes([.065 + j * .49, .535, .395, .285])
        ax.set_title(f"{metric.upper()} · reducción frente al original", loc="left", fontsize=15, weight="bold", pad=12)
        ax.axvspan(selected_index - .28, selected_index + .28, color="#dcefe8", alpha=.85, zorder=0)
        ax.axhline(0, color="#94a3b8", linewidth=1)
        for split in ("calibration", "confirmation"):
            x = np.array([t["order"] for t in trials if t[split] is not None])
            y = np.array([t[split]["metrics"][metric]["improvement_percent"] for t in trials if t[split] is not None])
            ax.plot(x, y, color=colors[split], linestyle="--" if split == "calibration" else "-",
                    marker="o" if split == "calibration" else "s", markersize=5, linewidth=1.9)
            rejected = [k for k, order in enumerate(x) if trials[order]["decision"] == "descartado"]
            ax.scatter(x[rejected], y[rejected], marker="x", color="#bb4845", s=70, linewidths=1.8, zorder=5)
        retained = selected["confirmation"]["metrics"][metric]["improvement_percent"]
        ax.plot([selected_index, len(trials) - .82], [retained, retained], color="#172b3a", linewidth=1, linestyle=":", alpha=.65)
        ax.scatter([selected_index], [retained], s=165, facecolors="none", edgecolors="#007f78", linewidths=2.1, zorder=7)
        annotation_y = (2.85 if metric == "mae" else 1.13) if selected_index == 3 else retained + (.25 if metric == "mae" else .15)
        ax.annotate(f"Retenido: {es(retained, 3)}%", (selected_index, retained), xytext=(selected_index - .75, annotation_y),
                    fontsize=10, color="#006b63", weight="bold", arrowprops={"arrowstyle": "-", "color": "#007f78", "lw": 1})
        if metric == "mse":
            ax.annotate("Mejor MSE,\npero peor MAE", (5, trials[5]["confirmation"]["metrics"][metric]["improvement_percent"]),
                        xytext=(4.15, .50), fontsize=9, color="#a43c39", arrowprops={"arrowstyle": "-", "color": "#a43c39", "lw": .9})
        ax.set_xticks(range(len(trials)), labels, fontsize=9 if len(trials) <= 6 else 8)
        ax.tick_params(axis="both", length=0, pad=7)
        ax.set_xlim(-.2, len(trials) - .75)
        bounds = (-1.0, 3.15) if metric == "mae" else (-.17, 1.25)
        values = [t[s]["metrics"][metric]["improvement_percent"] for t in trials
                  for s in ("calibration", "confirmation") if t[s] is not None]
        ax.set_ylim(min(bounds[0], min(values) - .15), max(bounds[1], max(values) + .2, annotation_y + .12))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: es(value, 1) + "%"))
        ax.grid(axis="y", color="#e2e8f0", linewidth=.8)
        ax.set_axisbelow(True)
    fig.text(.055, .458, "Subir = mejorar. Se retiene el avance que reduce MAE y MSE; no basta con mejorar la calibración.", fontsize=11, color="#334155")
    fig.text(.055, .405, f"Modelo retenido · {changed_values} valores modificados", fontsize=16, weight="bold")
    ax_table = fig.add_axes([.052, .172, .897, .211]); ax_table.axis("off")
    table_rows = []
    for label, result in validations:
        row = [label]
        for metric in ("mae", "mse"):
            values = result["metrics"][metric]
            row += [es(values["original"]), es(values["edited"]), es(values["improvement_percent"], 3) + "%"]
        table_rows.append(row)
    table = ax_table.table(cellText=table_rows,
                           colLabels=["Evaluación", "MAE original", "MAE actual", "Reducción", "MSE original", "MSE actual", "Reducción"],
                           colWidths=[.24, .135, .135, .09, .155, .155, .09], cellLoc="right", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(.6); cell.set_edgecolor("#e2e8f0"); cell.PAD = .075
        if row == 0:
            cell.set_facecolor("#eef2f6"); cell.set_text_props(weight="bold", color="#334155")
        elif row == len(validations):
            cell.set_facecolor("#e5f3ef"); cell.set_text_props(weight="bold", color="#006b63")
        else:
            cell.set_facecolor("white" if row % 2 else "#f8fafc")
        if col == 0:
            cell.set_text_props(ha="left")
    fig.text(.055, .132, stage["label"], fontsize=12, weight="bold", color="#006b63")
    bar = fig.add_axes([.055, .105, .89, .011]); bar.set_xlim(0, 1); bar.set_ylim(0, 1); bar.axis("off")
    bar.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0", fc="#e2e8f0", ec="none"))
    bar.add_patch(FancyBboxPatch((0, 0), stage["done"] / stage["total"], 1, boxstyle="round,pad=0", fc="#007f78", ec="none"))
    if full_candidates:
        report = full_candidates[-1]["report"]
        mae, mse = report["metrics"]["mae"], report["metrics"]["mse"]
        selected_recipe = load(recipes[selected_index][2] or recipes[selected_index][3][0])["edit"]["name"]
        test_status = "Test verificado." if retained_evidence and report["candidate_recipe"] == selected_recipe else "Sin test de este candidato."
        line = (f"Confirmación completa · {full_candidate_rows[-1][0].replace(chr(10), ' · ')}: "
                f"MAE {es(mae['reference'])} → {es(mae['candidate'])} ({es(mae['change_percent'], 4)}%); "
                f"MSE {es(mse['reference'])} → {es(mse['candidate'])} ({es(mse['change_percent'], 4)}%). {test_status}")
        fig.text(.055, .087, line, fontsize=8.5, color="#006b63")
    fig.text(.055, .067, note, fontsize=9, color="#64748b")
    fig.text(.055, .044, "Serie principal: 100 imputaciones. Ensayos iniciales con menos pacientes/imputaciones y pruebas de ejecución se conservan en el registro completo.", fontsize=8.5, color="#64748b")
    for extension in (".png", ".svg", ".pdf"):
        fig.savefig(prefix.with_suffix(extension), dpi=190, facecolor="white")
    plt.close(fig)

    interactive = make_subplots(rows=1, cols=2, subplot_titles=["MAE: reducción frente al original", "MSE: reducción frente al original"])
    for col, metric in enumerate(("mae", "mse"), 1):
        for split, legend in (("calibration", "Calibración · 128 · 1 semilla"), ("confirmation", "Confirmación · 128 · 2 semillas")):
            points = [t for t in trials if t[split] is not None]
            interactive.add_trace(go.Scatter(
                x=[t["order"] for t in points], y=[t[split]["metrics"][metric]["improvement_percent"] for t in points],
                name=legend, legendgroup=split, showlegend=col == 1, mode="lines+markers",
                line={"color": colors[split], "dash": "dash" if split == "calibration" else "solid"},
                marker={"size": 10, "symbol": ["x" if t["decision"] == "descartado" else "diamond" if t["decision"] == "retenido" else "circle" for t in points]},
                customdata=[[t["label"], t[split]["metrics"][metric]["original"], t[split]["metrics"][metric]["edited"], t["decision"]] for t in points],
                hovertemplate="%{customdata[0]}<br>Original: %{customdata[1]:.6f}<br>Editado: %{customdata[2]:.6f}<br>Reducción: %{y:.4f}%<br>Estado: %{customdata[3]}<extra>%{fullData.name}</extra>"), row=1, col=col)
        interactive.add_vrect(x0=selected_index - .28, x1=selected_index + .28, fillcolor="#dcefe8", opacity=.6, line_width=0, row=1, col=col, layer="below")
        interactive.add_hline(y=0, line_color="#94a3b8", line_width=1, row=1, col=col)
        interactive.update_xaxes(tickvals=list(range(len(trials))), ticktext=[label.replace("\n", "<br>") for label in labels], row=1, col=col)
        interactive.update_yaxes(ticksuffix="%", title_text="Reducción: mayor es mejor", row=1, col=col)
    interactive.update_layout(template="plotly_white", height=560, margin={"t": 105, "b": 90, "l": 65, "r": 35},
                              title={"text": f"CoSTI · evolución de la mejora<br><sup>PhysioNet · point missing · v0 · {len(evaluation_sigmas)} pasos · " + stamp + "</sup>", "x": .03},
                              legend={"orientation": "h", "y": 1.17, "x": 0})
    html_table = "<table><thead><tr>" + "".join(f"<th>{x}</th>" for x in ["Evaluación", "MAE original", "MAE actual", "Mejora", "MSE original", "MSE actual", "Mejora"]) + "</tr></thead><tbody>"
    html_table += "".join("<tr>" + "".join("<td>" + cell.replace("\n", "<br>") + "</td>" for cell in row) + "</tr>" for row in table_rows) + "</tbody></table>"
    document = "<!doctype html><html lang='es'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>CoSTI · evolución</title><style>body{font-family:system-ui,sans-serif;color:#172b3a;max-width:1400px;margin:25px auto;padding:0 22px;background:#fff}h1{font-size:28px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:12px;border-bottom:1px solid #e2e8f0;text-align:right}th:first-child,td:first-child{text-align:left}thead{background:#eef2f6}tbody tr:last-child{background:#e5f3ef}.status{padding:18px;background:#e5f3ef;border-radius:8px;color:#006b63}p{line-height:1.55}.note{color:#64748b;font-size:14px}</style><body>"
    document += interactive.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False, "responsive": True})
    document += f"<p>Las cruces señalan propuestas descartadas. La franja verde marca la referencia validada {retained_label}: <strong>{changed_values} pesos modificados</strong>. Cada hipótesis se mide en calibración y sólo avanza a confirmación si mejora ambas métricas frente a su referencia. Una calibración realizada en 512 pacientes aparece en su propio panel y deja sin punto la curva de 128. Los puntos sin confirmación no son mejoras validadas.</p><h2>Resultados de la referencia validada {retained_label}</h2>" + html_table
    if full_candidate_rows:
        document += "<h2>Candidatos en confirmación completa</h2><p>La referencia de cada comparación aparece en la primera columna. Los resultados de test del checkpoint retenido se muestran en la tabla anterior.</p>"
        document += "<table><thead><tr>" + "".join(f"<th>{name}</th>" for name in ["Comparación", "MAE referencia", "MAE candidato", "Reducción", "MSE referencia", "MSE candidato", "Reducción"]) + "</tr></thead><tbody>"
        document += "".join("<tr>" + "".join("<td>" + cell.replace("\n", "<br>") + "</td>" for cell in row) + "</tr>" for row in full_candidate_rows)
        document += "</tbody></table>"
    if wide_links:
        document += "<h2>Exploración en 512 pacientes</h2><p>Panel independiente, con V3 como referencia común. Sus valores absolutos no se comparan con los de 128 pacientes ni con el test completo.</p>"
        document += f"<a href='{wide_links['.html']}'><img src='{wide_links['.png']}' alt='Ensayos de calibración en 512 pacientes, con incertidumbre y propuestas descartadas' style='width:100%;height:auto'></a>"
        document += f"<p><a href='{wide_links['.html']}'>Abrir el panel interactivo de 512 pacientes</a></p>"
    document += "<p class='status'><strong>" + stage["label"] + "</strong><br>Corte: " + stamp + "</p><p class='note'>" + note + " Este archivo es una instantánea; no se actualiza automáticamente. No se han usado entrenamiento ni gradientes.</p></body></html>"
    prefix.with_suffix(".html").write_text(document)
    aliases = refresh_latest(prefix)
    print(json.dumps({"files": [str(prefix.with_suffix(ext)) for ext in (".png", ".svg", ".pdf", ".html", ".json", ".csv")],
                      "latest_links": aliases, "stage": stage, "current_test": test["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
