#!/usr/bin/env python3
"""
ECAF publication figure — case-study version using the real Adult dataset.

Same layout as plot_ecaf_figure.py but distributions come from real classifier
runs (6 classifiers × 50 holdout splits) instead of full confusion-matrix
enumeration.

Three conditions (Wong 2011 colorblind-safe palette):
  • Balanced reference : IR = 0.5, GR = 0.5  (solid blue)
  • IR skew            : IR = 0.1, GR = 0.5  (solid orange)
  • GR skew            : IR = 0.5, GR = 0.1  (dashed teal)

Run from repo root or this directory:
    python figures/ecaf_metric_comparison/plot_ecaf_figure_cs.py

Outputs (same directory):
    ecaf_ridgeline_metrics_cs.{png,pdf,svg}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from adult_case_study import (
    load_adult_dataset,
    sample_adult_subset,
    preprocess_adult,
    confusion_row_from_predictions,
    CLASSIFIERS,
)
from builtin_metrics import equal_opportunity_diff, equalized_odds_diff, statistical_parity_diff
from custom_metrics import conditional_y_association, fairness_phi, marginal_y_association
from sklearn.impute import KNNImputer
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ADULT_DATA_PATH = REPO_ROOT / "data" / "adult.data"
OUT_DIR         = Path(__file__).parent
FIGURE_STEM     = "ecaf_ridgeline_metrics_cs"

SAMPLE_SIZE    = 1100
HOLDOUT_SPLITS = 50
TEST_SIZE      = 0.33
RANDOM_STATE   = 2137

# Wong (2011) colorblind-safe palette
# (legend label, ir, gr, colour, linestyle)
CONDITIONS = [
    ("Balanced (IR=0.5, GR=0.5)",       0.5,  0.5,  "#0072B2", "-"),                   # blue
    ("IR skew (IR=0.1, GR=0.5)",        0.1,  0.5,  "#E69F00", "-"),                   # orange
    ("GR skew (IR=0.5, GR=0.1)",        0.5,  0.1,  "#009E73", (0, (6, 2))),           # teal
    ("Both skewed (IR=0.1, GR=0.1)",  0.1, 0.1, "#CC79A7", (0, (3, 1, 1, 1))),     # pink
]

# col 0/1: [-1, 1]   col 2: [0, 1]  (sharex='col')
STANDARD_PANELS = [
    ("Statistical Parity\nDifference",  statistical_parity_diff,   (-1, 1)),
    ("Equal Opportunity\nDifference",   equal_opportunity_diff,    (-1, 1)),
    ("Equalized Odds\nDifference",      equalized_odds_diff,       ( 0, 1)),
]
PROPOSED_PANELS = [
    ("Fairness Phi  (Φ)", fairness_phi,              (-1, 1)),
    ("Marginal Y",        marginal_y_association,    (-1, 1)),
    ("Conditional Y",     conditional_y_association, ( 0, 1)),
]


def _collect_confusion_matrices(
    adult_df: pd.DataFrame,
    ir: float,
    gr: float,
) -> pd.DataFrame:
    """Run all classifiers with holdout splits for a single (ir, gr) condition."""
    try:
        subset = sample_adult_subset(
            adult_df,
            sample_size=SAMPLE_SIZE,
            gr=gr,
            ir=ir,
            random_state=RANDOM_STATE,
        )
    except ValueError as exc:
        print(f"  [warn] IR={ir}, GR={gr}: {exc}")
        return pd.DataFrame()

    X_all, y_all, protected_values, _ = preprocess_adult(subset)
    holdout = ShuffleSplit(
        n_splits=HOLDOUT_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    rows: list[dict] = []
    for train_idx, test_idx in holdout.split(X_all):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        prot_test       = protected_values[test_idx]

        for clf_name, spec in CLASSIFIERS.items():
            try:
                pipe = make_pipeline(
                    KNNImputer(),
                    StandardScaler(),
                    spec.builder(RANDOM_STATE),
                )
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
            except Exception as exc:
                print(f"  [warn] {clf_name}: {exc}")
                continue
            conf = confusion_row_from_predictions(
                y_test, y_pred, prot_test,
                protected_value="Female",
                positive_class=1,
            )
            conf["clf"] = clf_name
            rows.append(conf)

    return pd.DataFrame(rows)


def _kde(vals: np.ndarray, x_grid: np.ndarray, min_bw: float = 0.015) -> np.ndarray:
    from scipy.stats import norm as _norm
    vals = vals[~np.isnan(vals)]
    if len(vals) < 5:
        return np.zeros_like(x_grid)
    std = vals.std(ddof=1)
    if std < min_bw:
        return _norm.pdf(x_grid, loc=float(vals.mean()), scale=min_bw)
    kde = gaussian_kde(vals, bw_method="scott")
    bw  = kde.factor * std
    if bw < min_bw:
        kde = gaussian_kde(vals, bw_method=min_bw / std)
    return kde(x_grid)


def _draw_panel(
    ax: plt.Axes,
    entries: list[tuple],
    x_grid: np.ndarray,
    x_range: tuple[float, float],
    title: str,
    *,
    show_xticks: bool,
    show_ylabel: bool,
    y_ceil: float,
) -> None:
    x_lo, x_hi = x_range

    for y, nan_frac, color, ls in entries:
        ax.plot(x_grid, y, color=color, lw=2.2, ls=ls, zorder=3)

    ann = [(nf, c) for _, nf, c, _ in entries if nf > 0.05]
    for k, (nf, c) in enumerate(ann):
        ax.text(0.97, 0.96 - k * 0.18, f"undefined {nf:.0%}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color=c, fontstyle="italic", fontweight="bold")

    ax.axvline(0, color="#555", lw=0.9, ls=":", zorder=4)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, y_ceil)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
    ax.tick_params(axis="both", labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    if not show_xticks:
        ax.tick_params(labelbottom=False)
    if not show_ylabel:
        ax.yaxis.set_ticklabels([])
        ax.tick_params(axis="y", length=0)


def main() -> None:
    print("Loading Adult dataset …")
    adult_df = load_adult_dataset(ADULT_DATA_PATH)
    print(f"  {len(adult_df):,} rows loaded")

    print("Collecting confusion matrices from real classifiers …")
    dfs: dict[tuple, pd.DataFrame] = {}
    for _lbl, ir, gr, _c, _ls in CONDITIONS:
        key = (ir, gr)
        if key not in dfs:
            print(f"  IR={ir}, GR={gr} …")
            dfs[key] = _collect_confusion_matrices(adult_df, ir=ir, gr=gr)
            n = len(dfs[key])
            print(f"    → {n:,} confusion-matrix rows  "
                  f"({len(CLASSIFIERS)} classifiers × {HOLDOUT_SPLITS} splits = "
                  f"{len(CLASSIFIERS) * HOLDOUT_SPLITS} expected)")

    all_panels = [STANDARD_PANELS, PROPOSED_PANELS]

    panel_data: list[list] = [[None] * 3, [None] * 3]
    for row, panels in enumerate(all_panels):
        for col, (_title, fn, x_range) in enumerate(panels):
            x_lo, x_hi = x_range
            x_grid = np.linspace(x_lo, x_hi, 600)
            entries = []
            for _lbl, ir, gr, color, ls in CONDITIONS:
                key      = (ir, gr)
                df       = dfs[key]
                if df.empty:
                    entries.append((np.zeros_like(x_grid), 1.0, color, ls))
                    continue
                raw      = np.asarray(fn(df), dtype=np.float64)
                nan_frac = np.isnan(raw).mean()
                y        = _kde(raw, x_grid)
                entries.append((y, nan_frac, color, ls))
            panel_data[row][col] = (x_grid, entries)

    y_ceils: list[list[float]] = [[0.0] * 3, [0.0] * 3]
    for row in range(2):
        for col in range(3):
            _, entries = panel_data[row][col]
            peaks = sorted(e[0].max() for e in entries)
            y_ceils[row][col] = peaks[-2] * 2.5 if len(peaks) >= 2 else peaks[-1] * 1.5

    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "font.size": 12,
    })

    fig, axes = plt.subplots(
        2, 3,
        figsize=(11, 6.5),
        sharex="col",
        gridspec_kw={"hspace": 0.55, "wspace": 0.18},
    )

    for row, panels in enumerate(all_panels):
        for col, (title, _fn, x_range) in enumerate(panels):
            x_grid, entries = panel_data[row][col]
            _draw_panel(
                axes[row, col], entries, x_grid, x_range, title,
                show_xticks=(row == 1),
                show_ylabel=(col == 0),
                y_ceil=y_ceils[row][col],
            )

    for row, letter in enumerate(["(a)", "(b)"]):
        axes[row, 0].text(
            -0.04, 1.10, letter,
            transform=axes[row, 0].transAxes,
            fontsize=15, fontweight="bold", va="bottom", ha="right",
        )

    fig.text(0.50, 0.115, "Metric value", ha="center", va="bottom", fontsize=13)
    fig.text(0.03, 0.50, "Density", ha="center", va="center",
             fontsize=13, rotation=90)

    handles = [
        mlines.Line2D([], [], color=c, lw=2.2, ls=ls, label=lbl)
        for lbl, _ir, _gr, c, ls in CONDITIONS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.50, 0.06),
    )

    fig.subplots_adjust(
        left=0.09, right=0.98,
        bottom=0.185, top=0.95,
        hspace=0.55, wspace=0.18,
    )

    for suffix in ("png", "pdf", "svg"):
        path = OUT_DIR / f"{FIGURE_STEM}.{suffix}"
        kw   = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kw)
        print(f"  {path}")

    print("Done.")


if __name__ == "__main__":
    main()