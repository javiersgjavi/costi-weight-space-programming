"""Build every figure in results/figures/ from the CSV/JSON files in results/data/.

    python scripts/make_figures.py

Needs only numpy, pandas and matplotlib. No checkpoint, dataset or GPU: the one
input that comes from the weights (output_head_coefficients.csv) is exported
once by scripts/export_head_coefficients.py.

All social figures are 16:9 (2400 x 1350 px), the size X/Twitter and LinkedIn
show without cropping.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "data"
OUT = ROOT / "results" / "figures"
REPO = "github.com/javiersgjavi/costi-weight-space-programming"

# Palette (validated for colour-vision deficiency with the dataviz validator).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"    # edited / kept / V4
ORANGE = "#eb6834"  # reused neurons / pending
RED = "#e34948"     # worse
PALE = "#d9d8d2"    # idle neurons, baseline bars

DPI = 150
SIZE = (16, 9)


def pick_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Inter", "Ubuntu", "Liberation Sans", "DejaVu Sans"):
        if name in have:
            return name
    return "sans-serif"


plt.rcParams.update({
    "font.family": [pick_font(), "DejaVu Sans"],
    "font.size": 14,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 1,
    "lines.solid_capstyle": "round",
    "mathtext.fontset": "dejavusans",
})


# --------------------------------------------------------------------------- helpers

def gelu(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def canvas(kicker, title, subtitle=None, size=SIZE):
    fig = plt.figure(figsize=size)
    fig.text(0.05, 0.935, kicker.upper(), fontsize=13, color=BLUE, weight="bold")
    fig.text(0.95, 0.935, REPO, fontsize=12, color=MUTED, ha="right")
    fig.text(0.05, 0.865, title, fontsize=30, weight="bold", va="center")
    if subtitle:
        fig.text(0.05, 0.795, subtitle, fontsize=16, color=INK_2, va="center")
    return fig


def footer(fig, note):
    fig.text(0.05, 0.04, note, fontsize=11.5, color=MUTED)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print("wrote", OUT / name)


def style_axis(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=12.5)


# --------------------------------------------------------------------------- 1 · headline

def fig_headline():
    test = pd.read_csv(DATA / "test_summary.csv").set_index("model")
    counts = json.loads((ROOT / "results/evidence/parameter_counts.json").read_text())
    o, v4 = test.loc["original"], test.loc["V4"]

    fig = canvas("Weight-space programming · CoSTI · PhysioNet 2019",
                 "965 weights, edited by hand. No training.",
                 "An AI agent read the weights of a trained imputation model, found where its output "
                 "was off,\nand rewrote them directly. No gradients, no optimizer, no new data.")

    tiles = [
        ("Mean absolute error", f"−{abs(v4.mae_change_vs_original_pct):.2f}%",
         f"{o.mae:.3f}  →  {v4.mae:.3f}"),
        ("Mean squared error", f"−{abs(v4.mse_change_vs_original_pct):.2f}%",
         f"{o.mse:.2f}  →  {v4.mse:.2f}"),
        ("Weights changed", f"{counts['percent_registered_parameters_changed']:.2f}%",
         f"{counts['changed_parameters']:,} of {counts['registered_parameters']:,}"),
    ]
    for i, (label, value, detail) in enumerate(tiles):
        x = 0.05 + i * 0.31
        fig.patches.append(matplotlib.patches.FancyBboxPatch(
            (x, 0.30), 0.28, 0.36, boxstyle="round,pad=0,rounding_size=0.012",
            transform=fig.transFigure, facecolor="#f3f2ee", edgecolor="none"))
        fig.text(x + 0.02, 0.61, label, fontsize=17, color=INK_2, va="center")
        fig.text(x + 0.02, 0.49, value, fontsize=64, weight="bold",
                 color=BLUE if i < 2 else INK, va="center")
        fig.text(x + 0.02, 0.365, detail, fontsize=17, color=INK_2, va="center")

    fig.text(0.05, 0.20,
             "Test set: 4,033 ICU patients from PhysioNet Challenge 2019, values hidden at random and imputed "
             "(100 samples × 2 noise seeds).", fontsize=14, color=INK_2)
    fig.text(0.05, 0.155,
             "Baseline: our own run of the released CoSTI checkpoint under the same protocol. Lower is better.",
             fontsize=14, color=INK_2)
    footer(fig, "Agent: OpenAI Codex, directed by Javier Solís-García. Method after Takehi et al., "
                "“Sparks of Weight-Space Programming” (MIT CSAIL, 2026).")
    save(fig, "01_headline.png")


# --------------------------------------------------------------------------- 2 · read the weights

def fig_read_weights():
    head = pd.read_csv(DATA / "output_head_coefficients.csv")
    idle_mask = head.a_original.abs() < 0.01
    n_idle = int(idle_mask.sum())

    fig = canvas("Step 1 · Read the weights",
                 f"{n_idle} of the 256 output neurons contribute almost nothing",
                 "The last two layers of CoSTI's output head are linear, so they collapse into one number "
                 "per hidden neuron:\nits effective weight aⱼ on the prediction. Sorting the 256 values "
                 "exposes a long flat stretch of idle neurons.")

    # Left: the algebra, as a compact card.
    fig.text(0.05, 0.66, "The output head", fontsize=17, weight="bold")
    fig.text(0.05, 0.615, "h  →  Linear 64→256  →  GELU  →  Linear 256→64  →  Linear 64→1", fontsize=13.5,
             color=INK_2)
    fig.text(0.05, 0.545, r"$y \;=\; b \;+\; \sum_{j=1}^{256}\; a_j \cdot \mathrm{GELU}(z_j)$", fontsize=22)
    fig.text(0.05, 0.475, r"$z_j = w_j \cdot h + b_j$   is neuron $j$'s input", fontsize=14, color=INK_2)
    fig.text(0.05, 0.435, r"$a_j$   is how hard neuron $j$ pushes the output", fontsize=14, color=INK_2)
    legend = [
        (BLUE, "edited existing neuron"),
        (ORANGE, "idle neuron reused as a blank register"),
        (MUTED, "active, untouched"),
        (PALE, f"idle ({n_idle} neurons, |aⱼ| < 0.01)"),
    ]
    for k, (c, t) in enumerate(legend):
        y = 0.345 - k * 0.045
        fig.patches.append(matplotlib.patches.Circle((0.058, y), 0.006, transform=fig.transFigure,
                                                     facecolor=c, edgecolor="none"))
        fig.text(0.072, y, t, fontsize=13.5, va="center", color=INK_2)

    # Right: the sorted effective weights.
    ax = fig.add_axes([0.45, 0.12, 0.51, 0.56])
    order = head.sort_values("a_original", ascending=False).reset_index(drop=True)
    x = np.arange(len(order))
    edited = {38, 198, 47, 255}
    reused = {246, 226, 179, 181, 150}
    colors = []
    for _, r in order.iterrows():
        u = int(r.unit)
        colors.append(BLUE if u in edited else ORANGE if u in reused else PALE if abs(r.a_original) < 0.01 else MUTED)
    ax.vlines(x, 0, order.a_original, colors=colors, linewidth=2.2)
    big = [i for i, u in enumerate(order.unit) if u in edited | reused]
    ax.scatter(x[big], order.a_original[big], s=70, c=[colors[i] for i in big], zorder=3,
               edgecolors=SURFACE, linewidths=2)
    ax.axhline(0, color=AXIS, linewidth=1)
    style_axis(ax)
    ax.set_xlim(-4, len(order) + 3)
    ax.set_ylim(-3.0, 5.8)
    ax.set_xticks([])
    ax.set_ylabel("effective weight aⱼ (original model)", fontsize=13)
    ax.spines["bottom"].set_visible(False)

    label_pos = {198: (12, 3.6), 38: (30, 3.0), 255: (200, -1.75), 47: (214, -2.6)}
    for u, (tx, ty) in label_pos.items():
        i = int(np.where(order.unit == u)[0][0])
        ax.annotate(f"unit {u}", (i, order.a_original[i]), (tx, ty), fontsize=12.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=1), ha="center")
    idle_idx = np.where(order.a_original.abs().values < 0.01)[0]
    mid = (idle_idx.min() + idle_idx.max()) / 2
    ax.annotate("", (idle_idx.min(), 0.55), (idle_idx.max(), 0.55),
                arrowprops=dict(arrowstyle="<->", color=INK_2, lw=1))
    ax.text(mid, 0.8, f"{n_idle} idle neurons, including the 5 reused ones\n(246, 226, 179, 181, 150)",
            ha="center", fontsize=12.5, color=INK_2)
    ax.text(0, 5.5, "neurons sorted by aⱼ", fontsize=12, color=MUTED)
    footer(fig, "Computed directly from the released checkpoint (scripts/export_head_coefficients.py). "
                "Model: CoSTI, PhysioNet 2019, point missing, version_0. Output in normalized units.")
    save(fig, "02_read_the_weights.png")


# --------------------------------------------------------------------------- 3 · program new functions

def fig_program_functions():
    head = pd.read_csv(DATA / "output_head_coefficients.csv").set_index("unit")
    a = head["a_v4"]  # exact effective output weights of the reused neurons in V4

    fig = canvas("Step 2 · Write the weights",
                 "Building small functions out of idle neurons",
                 "A very steep GELU behaves like a ReLU. Copying an existing neuron's input weights ×16 or ×64 "
                 "into an idle neuron,\nwith a new bias and output weight, makes it switch on exactly where "
                 "the agent wants. Sums of those switches draw shapes.")

    gs = fig.add_gridspec(1, 3, left=0.05, right=0.97, bottom=0.14, top=0.66, wspace=0.28)

    # (a) triangular bump on unit 198 from three reused neurons.
    ax = fig.add_subplot(gs[0])
    z = np.linspace(-0.5, 3.0, 800)
    # Each reused neuron sees 16 * (z198 - t): a 16x copy of unit 198's input row with a shifted bias.
    parts = [(246, 0.0), (226, 1.5), (179, 2.0)]
    for u, t in parts:
        ax.plot(z, a[u] * gelu(16 * (z - t)), color=ORANGE, lw=1.2, alpha=0.45)
    bump = sum(a[u] * gelu(16 * (z - t)) for u, t in parts)
    ax.plot(z, bump, color=ORANGE, lw=3)
    ax.axhline(0, color=AXIS, lw=1)
    style_axis(ax)
    ax.set_ylim(-0.15, 0.16)
    ax.set_xlabel("input of unit 198,  z₁₉₈", fontsize=13)
    ax.set_ylabel("added to the output (normalized)", fontsize=13)
    ax.set_title("A bump on unit 198", loc="left", fontsize=17, weight="bold", pad=34)
    ax.text(0, 1.04, "units 246 + 226 + 179 → peak at 1.5, off after 2", transform=ax.transAxes,
            fontsize=12.5, color=INK_2)
    ax.annotate("sum", (1.5, bump[np.argmin(abs(z - 1.5))]), (0.6, 0.13), fontsize=12.5,
                arrowprops=dict(arrowstyle="-", color=AXIS))
    ax.text(-0.45, -0.135, "faint lines: the three building blocks", fontsize=11.5, color=MUTED)

    # (b) capped ramp on the contrast z38 - z198 from two reused neurons.
    ax = fig.add_subplot(gs[1])
    d = np.linspace(-0.6, 1.8, 800)
    # Units 181 and 150 see 64 * (z38 - z198) and 64 * (z38 - z198 - 1).
    on = a[181] * gelu(64 * d)
    off = a[150] * gelu(64 * (d - 1))
    A = 64 * a[181]
    ax.plot(d, on, color=ORANGE, lw=1.2, alpha=0.45)
    ax.plot(d, off, color=ORANGE, lw=1.2, alpha=0.45)
    ax.plot(d, on + off, color=ORANGE, lw=3)
    ax.axhline(0, color=AXIS, lw=1)
    style_axis(ax)
    ax.set_ylim(-0.12, 0.12)
    ax.set_xlabel("contrast  d = z₃₈ − z₁₉₈", fontsize=13)
    ax.set_title("A capped ramp on a contrast", loc="left", fontsize=17, weight="bold", pad=34)
    ax.text(0, 1.04, f"units 181 + 150 → {A:.3f} · clip(d, 0, 1)", transform=ax.transAxes,
            fontsize=12.5, color=INK_2)
    ax.text(-0.55, 0.085, "fires only in the rare case\nwhere unit 38 overtakes 198", fontsize=11.5,
            color=MUTED)

    # (c) the full V4 recipe, as a card.
    card = fig.add_subplot(gs[2])
    card.axis("off")
    card.set_title("The full V4 recipe", loc="left", fontsize=17, weight="bold", pad=34)
    card.text(0, 1.04, "965 values in 5 tensors of the output head", transform=card.transAxes,
              fontsize=12.5, color=INK_2)
    steps = [
        ("V1", "Output gain ×1.005, bias −0.012", "recalibrates level and spread"),
        ("", "Units 198 and 38: output weight +1%", "two strong positive paths"),
        ("V2", "Bump on 198 (units 246, 226, 179)", "left panel"),
        ("", "Unit 47: input ×1.01, output ÷1.01", "same slope, shallower negative dip"),
        ("V3", "Unit 255: bias −0.01", "slides its negative dip sideways"),
        ("V4", "Ramp on z₃₈ − z₁₉₈ (units 181, 150)", "middle panel"),
    ]
    y = 0.97
    for tag, what, why in steps:
        if tag:
            card.text(0.0, y, tag, fontsize=13.5, weight="bold", color=BLUE, transform=card.transAxes, va="top")
        card.text(0.12, y, what, fontsize=13.5, color=INK, transform=card.transAxes, va="top")
        card.text(0.12, y - 0.075, why, fontsize=12, color=MUTED, transform=card.transAxes, va="top")
        y -= 0.165

    footer(fig, "Curves are the exact functions stored in the V4 weights. Every coefficient was fixed before it "
                "was evaluated; none was fitted to data.")
    save(fig, "03_write_the_weights.png")


# --------------------------------------------------------------------------- 4 · search

def fig_search():
    h = pd.read_csv(DATA / "hypotheses_128.csv")
    h = h[h.status != "baseline"].reset_index(drop=True)
    n_kept = int((h.status == "kept").sum())
    n_rej = int((h.status == "rejected").sum())
    n_pen = int((h.status == "pending").sum())

    fig = canvas("Step 3 · Test every idea with forward passes",
                 f"{len(h)} hypotheses: {n_kept} kept, {n_rej} rejected, {n_pen} still pending",
                 "Improvement over the original model on 128 held-out validation patients (2 noise seeds). "
                 "Each edit was frozen\nbefore its first evaluation. Kept edits lowered both errors "
                 "on held-out patients and became the base for the next one.")

    gs = fig.add_gridspec(1, 2, left=0.33, right=0.97, bottom=0.12, top=0.70, wspace=0.08)
    rows = np.arange(len(h))[::-1]
    style = {"kept": dict(c=BLUE, marker="o", s=110),
             "rejected": dict(c=MUTED, marker="x", s=80),
             "pending": dict(facecolors="none", edgecolors=ORANGE, marker="o", s=110, linewidths=2.2)}

    for col, (metric, title) in enumerate([("mae", "MAE reduction"), ("mse", "MSE reduction")]):
        ax = fig.add_subplot(gs[col])
        vals = h[f"confirmation_{metric}_reduction_pct"]
        v4 = h.loc[h.version == "V4", f"confirmation_{metric}_reduction_pct"].item()
        ax.axvline(v4, color=BLUE, lw=1, alpha=0.5)
        for status, st in style.items():
            m = (h.status == status) & vals.notna()
            ax.scatter(vals[m], rows[m], zorder=3, **st)
        for i, r in h.iterrows():
            if pd.isna(vals[i]):
                ax.text(0.05, rows[i], "rejected on calibration, never confirmed", fontsize=11, color=MUTED,
                        va="center")
        style_axis(ax, "x")
        ax.set_xlim(0, 2.7 if metric == "mae" else 1.12)
        ax.set_ylim(-0.7, len(h) - 0.3)
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
        ax.set_title(title, loc="left", fontsize=16, weight="bold")
        ax.text(v4, len(h) - 0.25, " V4", color=INK_2, fontsize=11.5, va="bottom")
        if col == 0:
            names = [r.hypothesis + (f"  ({r.version})" if r.status == "kept" and isinstance(r.version, str) else "")
                     for _, r in h.iterrows()]
            ax.set_yticks(rows)
            ax.set_yticklabels(names, fontsize=12.5)
            for tick, (_, r) in zip(ax.get_yticklabels(), h.iterrows()):
                if r.status == "kept":
                    tick.set_fontweight("bold")
                    tick.set_color(INK)
            ax.tick_params(axis="y", pad=14)
        else:
            ax.set_yticks([])

    for k, (lab, st) in enumerate([("kept", style["kept"]), ("rejected", style["rejected"]),
                                   ("pending (promising, never tested)", style["pending"])]):
        x0 = 0.60 + [0, 0.07, 0.165][k]
        fig.text(x0 + 0.014, 0.745, lab, fontsize=12.5, color=INK_2, va="center")
        axl = fig.add_axes([x0 - 0.006, 0.735, 0.012, 0.02])
        axl.scatter([0], [0], **st)
        axl.axis("off")

    footer(fig, "Rejected edits are documented with their numbers in docs/experiment-log.md. Values are "
                "relative to the original model on the same patients and noise.")
    save(fig, "04_search.png")


# --------------------------------------------------------------------------- 5 · where the gain comes from

def fig_where():
    g = json.loads((DATA / "per_group_test.json").read_text())["groups"]
    test = pd.read_csv(DATA / "test_summary.csv").set_index("model")
    ctx = g["static_administrative_and_time_variables"]
    phy = g["measured_physiological_variables"]

    fig = canvas("Reading the result honestly",
                 "Most of the gain is recalibration, not better physiology",
                 "Context columns (age, gender, ICU unit, admission time, length of stay) improve a lot. "
                 "The 34 vitals and lab\nvalues barely move. And the first, simplest 193-value edit (V1) "
                 "delivers most of the test gain.")

    # Left: by variable group.
    ax = fig.add_axes([0.05, 0.14, 0.40, 0.52])
    groups = [("6 context variables", ctx), ("34 physiological variables", phy)]
    y = np.array([1.0, 0.0])
    hgt = 0.28
    for gi, (name, gr) in enumerate(groups):
        for mi, (metric, color) in enumerate([("mae", BLUE), ("mse", ORANGE)]):
            v = -gr[metric]["change_percent"]
            yy = y[gi] + (0.18 if mi == 0 else -0.18)
            ax.barh(yy, v, height=hgt, color=color)
            ax.text(v + 0.5, yy, f"−{v:.2f}%", va="center", fontsize=14, color=INK)
    for gi, (name, _) in enumerate(groups):
        ax.text(0.3, y[gi] + 0.45, name, fontsize=14.5, color=INK, weight="bold", va="bottom")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    style_axis(ax, "x")
    ax.set_xlim(0, 33)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title("Error reduction on the test set, by variable group", loc="left", fontsize=16,
                 weight="bold", pad=14)
    ax.text(32.5, 1.62, f"{ctx['mae']['share_of_total_error_reduction_percent']:.0f}% of the total\n"
                        "MAE reduction", fontsize=12.5, color=INK_2, ha="right")
    ax.set_ylim(-0.45, 1.85)
    _legend(fig, 0.80, 0.725, [("MAE", BLUE), ("MSE", ORANGE)])

    # Right: by version.
    ax = fig.add_axes([0.57, 0.14, 0.38, 0.52])
    names = ["V1\n193 values", "V4\n965 values"]
    xs = np.array([0, 1])
    w = 0.28
    for mi, (metric, color) in enumerate([("mae", BLUE), ("mse", ORANGE)]):
        vals = [-test.loc["V1", f"{metric}_change_vs_original_pct"], -test.loc["V4", f"{metric}_change_vs_original_pct"]]
        xx = xs + (-w / 2 - 0.02 if mi == 0 else w / 2 + 0.02)
        ax.bar(xx, vals, width=w, color=color)
        for x_, v in zip(xx, vals):
            ax.text(x_, v + 0.05, f"−{v:.2f}%", ha="center", fontsize=14, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=14, color=INK)
    style_axis(ax, "y")
    ax.set_ylim(0, 3.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.set_title("Error reduction on the test set, by version", loc="left", fontsize=16, weight="bold", pad=14)
    ax.text(0.5, 2.75, "V1 = output gain/bias + two boosted neurons.\n"
                       "V2–V4 programmed functions add the last 0.29%.", ha="center", fontsize=12.5,
            color=INK_2)

    footer(fig, "Test set reused: V1 was tested before V4 was built. 17 of 40 variables improve on both "
                "metrics; some sparse lab values get worse (see per-variable figure).")
    save(fig, "05_where_the_gain_comes_from.png")


def _legend(fig, x, y, items):
    for k, (lab, c) in enumerate(items):
        xx = x + k * 0.075
        fig.patches.append(matplotlib.patches.FancyBboxPatch(
            (xx, y - 0.008), 0.016, 0.016, boxstyle="round,pad=0,rounding_size=0.003",
            transform=fig.transFigure, facecolor=c, edgecolor="none"))
        fig.text(xx + 0.022, y, lab, fontsize=13, va="center", color=INK_2)


# --------------------------------------------------------------------------- 6 · per variable

CONTEXT = {"Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"}


def fig_per_variable():
    v = pd.read_csv(DATA / "per_variable_test.csv")
    v = v.sort_values("mae_change_percent", ascending=False).reset_index(drop=True)
    n = len(v)
    fig = plt.figure(figsize=(16, 15))
    fig.text(0.05, 0.965, "PER-VARIABLE RESULT · TEST SET", fontsize=13, color=BLUE, weight="bold")
    fig.text(0.05, 0.94, "Not every variable improves", fontsize=30, weight="bold", va="center")
    fig.text(0.95, 0.965, REPO, fontsize=12, color=MUTED, ha="right")
    fig.text(0.05, 0.905, "Change in error for each of the 40 PhysioNet columns, V4 vs original. Left of zero "
                          "(blue) = better, right (red) = worse.\nBold = context variable. The grey number is "
                          "how many values were scored per seed.", fontsize=14, color=INK_2, va="center")
    gs = fig.add_gridspec(1, 2, left=0.20, right=0.97, bottom=0.05, top=0.86, wspace=0.12)
    rows = np.arange(n)[::-1]
    clip = 40
    for col, metric in enumerate(["mae", "mse"]):
        ax = fig.add_subplot(gs[col])
        vals = v[f"{metric}_change_percent"].values
        shown = np.clip(vals, -clip, clip)
        ax.barh(rows, shown, height=0.62, color=np.where(vals < 0, BLUE, RED))
        for r, val, s in zip(rows, vals, shown):
            if abs(val) > clip:
                ax.text(s + (1 if s > 0 else -1), r, f"{val:+.0f}%", va="center", fontsize=11,
                        ha="left" if s > 0 else "right", color=INK)
        ax.axvline(0, color=INK_2, lw=1)
        style_axis(ax, "x")
        ax.set_xlim(-clip - 8, clip + 8)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:+.0f}%" if x else "0%"))
        ax.set_title(f"{metric.upper()} change", loc="left", fontsize=17, weight="bold")
    for r, (_, row) in zip(rows, v.iterrows()):
        y = 0.05 + (r + 0.7) / (n + 0.4) * 0.81
        fig.text(0.05, y, row.variable, fontsize=12.5, va="center",
                 weight="bold" if row.variable in CONTEXT else "normal", color=INK)
        fig.text(0.185, y, f"{int(row.scored_values_per_seed):,}", fontsize=11, va="center", ha="right",
                 color=MUTED)
    fig.text(0.05, 0.018, "Bars clipped at ±40%; the true value is printed next to clipped bars. Variables with "
                          "few scored values (pH, Chloride) swing the most. Test set, mean of 2 seeds.",
             fontsize=11.5, color=MUTED)
    save(fig, "06_per_variable.png")


if __name__ == "__main__":
    fig_headline()
    fig_read_weights()
    fig_program_functions()
    fig_search()
    fig_where()
    fig_per_variable()
