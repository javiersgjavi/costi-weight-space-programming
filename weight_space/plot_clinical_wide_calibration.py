"""Show the separate512-patient cohort without mixing it with128 or test data."""

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .checkpoints import sha256
from .clinical_partition_report import report_pairs


def read(path):
    return json.loads(Path(path).read_text())


def es(value, digits=6):
    return f"{value:.{digits}f}".replace(".", ",")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--retained-test-report", type=Path, default=Path("artifacts/clinical_point_final_results.json"))
    p.add_argument("--retained-manifest", type=Path,
                   default=Path("artifacts/edited_weights/clinical_v1/ctimba_mimic-iii-challenge_point/version_0/weights.json"))
    p.add_argument("--retained-label", default="V1")
    p.add_argument("--retained-audit", type=Path, help="Completed export verification with benchmark-reuse status")
    args = p.parse_args()
    prefix, trials = args.output_prefix, read(args.trials)
    extensions = (".png", ".svg", ".pdf", ".html", ".json", ".csv")
    assert not any(prefix.with_suffix(ext).exists() for ext in extensions)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    reference_path = Path("artifacts/results/clinical_point_v0_gate255_control_cal512x100_seed1_chunk20.json")
    reference = read(reference_path)
    assert reference["split"] == "calibration" and len(reference["window_ids"]) == 512
    assert reference["samples"] == 100 and reference["seed"] == 20261103
    assert reference["edit"]["name"] == "clinical_v0_composed_gate255_translate"
    assert len({trial["recipe"] for trial in trials}) == len(trials)
    entries, rows = [], []
    for order, trial in enumerate(trials, 1):
        entry = {**trial, "order": order, "metrics": None}
        if trial["report"]:
            path = Path(trial["report"])
            report = read(path)
            assert report["split"] == "calibration" and report["patients_per_seed"] == 512
            assert report["samples"] == 100 and report["seeds"] == [20261103]
            assert report["reference_recipe"] == reference["edit"]["name"]
            assert report["candidate_recipe"] == trial["recipe"]
            pairs = [(Path(row["reference"]), Path(row["candidate"])) for row in report["raw_audits"]]
            assert len(pairs) == 1 and pairs[0][0].resolve() == reference_path.resolve()
            assert report_pairs(pairs) == report
            entry.update(metrics=report["metrics"], report_sha256=sha256(path))
            for metric, values in report["metrics"].items():
                rows.append({"order": order, "trial": trial["label"], "status": trial["status"],
                             "reference": report["reference_recipe"], "split": "calibration", "patients": 512,
                             "samples": 100, "seed": 20261103, "metric": metric,
                             "reference_value": values["reference"], "candidate_value": values["candidate"],
                             "reduction_percent": -values["change_percent"],
                             "delta_ci95_low": values["delta_ci95_patient_bootstrap"][0],
                             "delta_ci95_high": values["delta_ci95_patient_bootstrap"][1]})
        else:
            freeze = read(trial["freeze"])
            assert freeze["candidate"] == trial["recipe"] and sha256(freeze["recipe"]) == freeze["recipe_sha256"]
            entry["freeze_sha256"] = sha256(trial["freeze"])
        if trial.get("decision"):
            decision = read(trial["decision"])
            if trial["rejected"]:
                assert decision["decision"] == "rejected"
            entry["decision_sha256"] = sha256(trial["decision"])
        entries.append(entry)
    retained, manifest = read(args.retained_test_report), read(args.retained_manifest)
    assert manifest["test_comparison"] == retained and manifest["roundtrip_prediction_verified"]
    assert sha256(args.retained_manifest.with_suffix(".ckpt")) == manifest["export_sha256"]
    assert retained["split"] == "test" and retained["windows_per_seed"] == 4033
    assert retained["samples_per_seed"] == 100 and len(retained["seeds"]) >= 2
    assert retained["balanced_improvement_established"]
    benchmark_reuse = False
    if args.retained_audit:
        audit = read(args.retained_audit)
        assert audit["verification_passed"] and audit["source_checkpoint_unchanged"]
        assert audit["metrics_vs_original"] == retained["metrics"]
        audited_files = {Path(key).resolve(): value for key, value in audit["artifact_sha256"].items()}
        for path in (args.retained_test_report, args.retained_manifest, args.retained_manifest.with_suffix(".ckpt")):
            assert audited_files[path.resolve()] == sha256(path)
        benchmark_reuse = audit["benchmark_reuse"]
    for row in retained["evidence"]:
        for key in ("original", "edited"):
            assert sha256(row[key]) == row[key + "_sha256"]
    stamp = datetime.now(ZoneInfo("Europe/Madrid")).strftime("%d/%m/%Y %H:%M %Z")
    snapshot = {"timestamp": stamp, "reference": reference["edit"]["name"],
                "reference_result": str(reference_path), "reference_result_sha256": sha256(reference_path),
                "reference_metrics": reference["metrics"], "split": "calibration", "patients": 512,
                "samples": 100, "seed": 20261103, "trials": entries,
                "trials_config": str(args.trials), "trials_config_sha256": sha256(args.trials),
                "retained_test_report": str(args.retained_test_report), "retained_test_report_sha256": sha256(args.retained_test_report),
                "retained_label": args.retained_label, "retained_test_metrics": retained["metrics"],
                "retained_audit": str(args.retained_audit) if args.retained_audit else None,
                "benchmark_reuse": benchmark_reuse,
                "interpretation": "Separate512-patient calibration cohort. Percentages compare every candidate to the same V3 parent. Pending evaluations have no numerical point. Intervals are paired patient bootstrap intervals, not evidence of test generalization. A calibration improvement can be rejected on confirmation."}
    prefix.with_suffix(".json").write_text(json.dumps(snapshot, indent=2) + "\n")
    with prefix.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    labels = ["Base\nV3"] + [trial.get("plot_label", trial["label"].replace(" ", "\n", 1))
                                      for trial in entries]
    colors = {False: "#167f79", True: "#b84945"}
    table_rows = []
    for trial in entries:
        row = [trial["label"], trial["status"]]
        for metric in ("mae", "mse"):
            if trial["metrics"] is None:
                row += ["—", "—"]
            else:
                value = trial["metrics"][metric]
                row += [es(value["candidate"]), es(-value["change_percent"], 4) + "%"]
        table_rows.append(row)
    test_line = "Test completo · referencia validada " + args.retained_label + ": "
    test_line += "; ".join(f"{metric.upper()} {es(retained['metrics'][metric]['original'])} → {es(retained['metrics'][metric]['edited'])}"
                           for metric in ("mae", "mse"))
    if benchmark_reuse:
        test_line += " · Benchmark reutilizado"
    draw_figures(prefix, labels, entries, table_rows, reference, test_line, stamp)
    for ext in extensions:
        alias = prefix.with_name("clinical_calibration512_latest").with_suffix(ext)
        assert not alias.exists() or alias.is_symlink()
        temporary = alias.with_name(alias.name + ".next")
        assert not temporary.exists() and not temporary.is_symlink()
        temporary.symlink_to(prefix.with_suffix(ext).name); temporary.replace(alias)
    print(json.dumps({"prefix": str(prefix), "completed_trials": sum(t["metrics"] is not None for t in entries),
                      "pending_trials": sum(t["metrics"] is None for t in entries)}, indent=2))


def draw_figures(prefix, labels, entries, table_rows, reference, test_line, stamp):
    """Render the same values in printable and interactive figures."""
    colors = {False: "#167f79", True: "#b84945"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(left=.07, right=.97, top=.80, bottom=.47, wspace=.22)
    fig.suptitle("CoSTI · exploración en 512 pacientes", x=.055, y=.96, ha="left", fontsize=23, weight="bold")
    fig.text(.055, .905, "Calibración · 100 imputaciones · una semilla · referencia común: V3", fontsize=12)
    fig.text(.97, .905, stamp, ha="right", fontsize=10, color="#64748b")
    interactive = make_subplots(rows=1, cols=2, subplot_titles=["MAE: reducción frente a V3", "MSE: reducción frente a V3"])
    for column, metric in enumerate(("mae", "mse"), 1):
        ax = axes[column - 1]
        ax.axhline(0, color="#94a3b8", linewidth=1)
        ax.scatter([0], [0], color="#334155", s=45)
        for trial in entries:
            x, values = trial["order"], trial["metrics"]
            if values is None:
                ax.axvspan(x - .35, x + .35, color="#f1f5f9", zorder=0)
                ax.text(x, .94, "Pendiente", ha="center", va="top", transform=ax.get_xaxis_transform(), color="#64748b", fontsize=9)
                continue
            value = values[metric]
            y = -value["change_percent"]
            low, high = [-100 * delta / value["reference"] for delta in reversed(value["delta_ci95_patient_bootstrap"])]
            assert low <= y <= high
            color, marker = colors[trial["rejected"]], "x" if trial["rejected"] else "o"
            ax.errorbar(x, y, yerr=[[y - low], [high - y]], fmt=marker, color=color, capsize=5, markersize=8, linewidth=1.7)
            interactive.add_trace(go.Scatter(x=[x], y=[y], name=trial["label"], legendgroup=trial["recipe"], showlegend=column == 1,
                                  mode="markers", marker={"color": color, "symbol": "x" if trial["rejected"] else "circle", "size": 11},
                                  error_y={"type": "data", "symmetric": False, "array": [high - y], "arrayminus": [y - low]},
                                  customdata=[[trial["label"], value["reference"], value["candidate"], trial["status"]]],
                                  hovertemplate="%{customdata[0]}<br>V3: %{customdata[1]:.6f}<br>Candidato: %{customdata[2]:.6f}<br>Reducción: %{y:.5f}%<br>%{customdata[3]}<extra></extra>"), row=1, col=column)
        ax.set_title(metric.upper() + " · reducción frente a V3", loc="left", fontsize=14)
        ax.set_xticks(range(len(labels)), labels, fontsize=9)
        ax.set_xlim(-.4, len(labels) - .6)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: es(value, 2) + "%"))
        ax.grid(axis="y", color="#e2e8f0"); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        interactive.add_hline(y=0, line_color="#94a3b8", row=1, col=column)
        interactive.update_xaxes(tickvals=list(range(len(labels))), ticktext=[label.replace("\n", "<br>") for label in labels], range=[-.4, len(labels) - .6], row=1, col=column)
        interactive.update_yaxes(ticksuffix="%", row=1, col=column)
    fig.text(.055, .385, "Subir = mejorar en calibración. Barras: IC95% por paciente. Cruz roja: propuesta descartada.", fontsize=11)
    ref_line = f"V3 en estos 512 pacientes: MAE {es(reference['metrics']['mae'])} · MSE {es(reference['metrics']['mse'])}"
    fig.text(.055, .35, ref_line, fontsize=11, weight="bold")
    table_ax = fig.add_axes([.055, .13, .915, .19]); table_ax.axis("off")
    headings = ["Hipótesis", "Estado", "MAE", "Reducción", "MSE", "Reducción"]
    table = table_ax.table(cellText=table_rows, colLabels=headings, cellLoc="right", colWidths=[.19, .37, .12, .09, .14, .09], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#e2e8f0"); cell.set_linewidth(.7)
        if row == 0:
            cell.set_facecolor("#eef2f6"); cell.set_text_props(weight="bold")
        if col < 2:
            cell.set_text_props(ha="left")
    fig.text(.055, .085, test_line, fontsize=11, color="#006b63", weight="bold")
    fig.text(.055, .048, "La calibración no sustituye al test. Cohortes distintas: sus valores absolutos no son comparables entre sí.", fontsize=10, color="#64748b")
    for ext in (".png", ".svg", ".pdf"):
        fig.savefig(prefix.with_suffix(ext), dpi=180, facecolor="white")
    plt.close(fig)
    interactive.update_layout(template="plotly_white", height=530, title="CoSTI · calibración de 512 pacientes · referencia V3", margin={"b": 100}, legend={"orientation": "h", "y": 1.15})
    table_html = "<table><thead><tr>" + "".join("<th>" + html.escape(h) + "</th>" for h in headings) + "</tr></thead><tbody>"
    table_html += "".join("<tr>" + "".join("<td>" + html.escape(c) + "</td>" for c in row) + "</tr>" for row in table_rows) + "</tbody></table>"
    document = "<!doctype html><html lang='es'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>CoSTI · calibración512</title><style>body{font-family:system-ui,sans-serif;max-width:1400px;margin:24px auto;padding:0 20px;color:#172b3a}p{line-height:1.5}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:12px;border-bottom:1px solid #e2e8f0;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}.note{color:#64748b}.retained{background:#e5f3ef;padding:18px;color:#006b63}</style><body>"
    document += interactive.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False, "responsive": True})
    document += "<p>Solo esta cohorte: 512 pacientes de calibración, 100 imputaciones y semilla 20261103. Cada punto compara con el mismo V3. Barras: intervalo del 95% por paciente. Los ensayos pendientes no tienen punto numérico.</p>"
    document += "<p>Las cruces rojas conservan los ensayos descartados; la tabla indica en qué fase fallaron. La cola de la unidad 167 mejoró la calibración pero empeoró MAE y MSE en confirmación.</p><p><strong>" + html.escape(ref_line) + "</strong></p>" + table_html
    document += "<p class='retained'>" + html.escape(test_line) + "</p><p class='note'>Las cohortes de 128 y 512 pacientes y el test completo son distintos. No se comparan sus valores absolutos. Instantánea: " + stamp + ". Sin entrenamiento ni gradientes.</p></body></html>"
    prefix.with_suffix(".html").write_text(document)


if __name__ == "__main__":
    main()
