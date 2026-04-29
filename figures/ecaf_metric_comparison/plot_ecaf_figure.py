#!/usr/bin/env python3
"""
ECAF publication figure  —  (a) standard  vs  (b) proposed metrics.

Four conditions per panel (all at n = 30):
  • Balanced reference : IR = 0.5, GR = 0.5  (solid blue)
  • IR skew            : IR = 0.1, GR = 0.5  (solid red)
  • GR skew            : IR = 0.5, GR = 0.1  (dashed green)
  • Biased classifier  : SR_p = 0.1 filtered from balanced  (dotted purple)

Run from repo root or this directory:
    python figures/ecaf_metric_comparison/plot_ecaf_figure.py

Outputs (same directory):
    ecaf_ridgeline_metrics.{png,pdf,svg}
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

from builtin_metrics import equal_opportunity_diff, equalized_odds_diff, statistical_parity_diff
from custom_metrics import conditional_y_association, fairness_phi, marginal_y_association

OUT_DIR     = Path(__file__).parent
FIGURE_STEM = "ecaf_ridgeline_metrics"
N_TOTAL     = 30

# (legend label, ir, gr, colour, linestyle)
CONDITIONS = [
    ("Balanced (IR=0.5, GR=0.5)", 0.5, 0.5, "#1A5276", "-"),
    ("IR=0.1 (GR=0.5)",           0.1, 0.5, "#9B2335", "-"),
    ("GR=0.1 (IR=0.5)",           0.5, 0.1, "#2E7D32", (0, (5, 2))),
]

# col 0/1: [-1, 1]   col 2: [0, 1]  (sharex='col')
STANDARD_PANELS = [
    ("Statistical Parity\nDifference",  statistical_parity_diff,   (-1, 1)),
    ("Equal Opportunity\nDifference",   equal_opportunity_diff,    (-1, 1)),
    ("Equalized Odds\nDifference",      equalized_odds_diff,       ( 0, 1)),
]
PROPOSED_PANELS = [
    ("Fairness Phi  (Φ)",               fairness_phi,              (-1, 1)),
    ("Marginal Y\nAssociation",         marginal_y_association,    (-1, 1)),
    ("Conditional Y\nAssociation",      conditional_y_association, ( 0, 1)),
]


def _enumerate_matrices(n_total: int, ir: float, gr: float) -> pd.DataFrame:
    n_pos = round(ir * n_total)
    n_j   = round(gr * n_total)
    n_i   = n_total - n_j

    j_min = max(0, n_pos - n_i)
    j_max = min(n_pos, n_j)

    blocks: list[np.ndarray] = []
    for jp in range(j_min, j_max + 1):
        ip  = n_pos - jp
        jn  = n_j   - jp
        i_n = n_i   - ip
        if i_n < 0 or jn < 0:
            continue
        jt  = np.arange(jp  + 1, dtype=np.int16)
        jf  = np.arange(jn  + 1, dtype=np.int16)
        it  = np.arange(ip  + 1, dtype=np.int16)
        if_ = np.arange(i_n + 1, dtype=np.int16)
        JT, JF, IT, IF_ = np.meshgrid(jt, jf, it, if_, indexing="ij")
        nr = JT.size
        blocks.append(np.column_stack([
            IT.ravel(), IF_.ravel(),
            np.full(nr, i_n, np.int16) - IF_.ravel(),
            np.full(nr, ip,  np.int16) - IT.ravel(),
            JT.ravel(), JF.ravel(),
            np.full(nr, jn,  np.int16) - JF.ravel(),
            np.full(nr, jp,  np.int16) - JT.ravel(),
        ]))

    return pd.DataFrame(
        np.vstack(blocks).astype(np.int32),
        columns=["i_tp","i_fp","i_tn","i_fn","j_tp","j_fp","j_tn","j_fn"],
    )


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
    pad = (x_hi - x_lo) * 0.04

    for y, nan_frac, color, ls in entries:
        ax.plot(x_grid, y, color=color, lw=1.5, ls=ls, zorder=3)

    ann = [(nf, c) for _, nf, c, _ in entries if nf > 0.05]
    for k, (nf, c) in enumerate(ann):
        ax.text(0.97, 0.96 - k * 0.15, f"undef. {nf:.0%}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color=c, fontstyle="italic")

    ax.axvline(0, color="#666", lw=0.6, ls=":", zorder=4)
    ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.set_ylim(0, y_ceil)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=4)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)

    if not show_xticks:
        ax.tick_params(labelbottom=False)
    if not show_ylabel:
        ax.yaxis.set_ticklabels([])
        ax.tick_params(axis="y", length=0)


def main() -> None:
    print("Enumerating confusion matrices …")
    dfs: dict[tuple, pd.DataFrame] = {}
    for _lbl, ir, gr, _c, _ls in CONDITIONS:
        key = (ir, gr)
        if key not in dfs:
            dfs[key] = _enumerate_matrices(N_TOTAL, ir, gr)
            print(f"  IR={ir}, GR={gr}: {len(dfs[key]):,} rows")

    all_panels = [STANDARD_PANELS, PROPOSED_PANELS]

    panel_data: list[list] = [[None] * 3, [None] * 3]
    for row, panels in enumerate(all_panels):
        for col, (_title, fn, x_range) in enumerate(panels):
            x_lo, x_hi = x_range
            pad    = (x_hi - x_lo) * 0.04
            x_grid = np.linspace(x_lo - pad, x_hi + pad, 600)
            entries = []
            for _lbl, ir, gr, color, ls in CONDITIONS:
                key      = (ir, gr)
                raw      = np.asarray(fn(dfs[key]), dtype=np.float64)
                nan_frac = np.isnan(raw).mean()
                y        = _kde(raw, x_grid)
                entries.append((y, nan_frac, color, ls))
            panel_data[row][col] = (x_grid, entries)

    y_ceils: list[list[float]] = [[0.0] * 3, [0.0] * 3]
    for row in range(2):
        for col in range(3):
            _, entries = panel_data[row][col]
            peaks = sorted(e[0].max() for e in entries)
            y_ceils[row][col] = peaks[-2] * 2.2 if len(peaks) >= 2 else peaks[-1] * 1.5

    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    fig, axes = plt.subplots(
        2, 3,
        figsize=(8.5, 4.8),
        sharex="col",
        gridspec_kw={"hspace": 0.50, "wspace": 0.14},
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
            -0.04, 1.08, letter,
            transform=axes[row, 0].transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="right",
        )

    fig.text(0.50, 0.06, "Metric value", ha="center", va="bottom", fontsize=9.5)
    fig.text(0.03, 0.50, "Density", ha="center", va="center",
             fontsize=9.5, rotation=90)

    handles = [
        mlines.Line2D([], [], color=c, lw=1.5, ls=ls, label=lbl)
        for lbl, _ir, _gr, c, ls in CONDITIONS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=8.0,
        frameon=False,
        bbox_to_anchor=(0.50, 0.00),
    )

    fig.subplots_adjust(
        left=0.09, right=0.98,
        bottom=0.16, top=0.95,
        hspace=0.50, wspace=0.14,
    )

    for suffix in ("png", "pdf", "svg"):
        path = OUT_DIR / f"{FIGURE_STEM}.{suffix}"
        kw   = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kw)
        print(f"  {path}")


if __name__ == "__main__":
    main()
