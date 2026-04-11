"""Matplotlib plot builders for the synthetic and case-study workflows."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib.ticker import PercentFormatter

from synthetic_analysis import ensure_metric_column, ensure_metric_columns, resolve_ratio_column, value_grid_for_heatmap


plt.style.use("default")

COLOURS = ["#EE7733", "#33BBEE", "#EE3377", "#888888", "#009988", "#332288"]
X_LABELS = {
    "gr": "Group ratio (GR)",
    "ir": "Imbalance ratio (IR)",
    "sr": "Stereotypical ratio SR_p",
    "sr_n": "Stereotypical ratio SR_n",
    "sr_c": "Stereotypical ratio SR_c",
}

STYLE_OVERRIDES = {
    "Accuracy Equality Difference": {"color": "#6699CC", "marker": "*"},
    "Statistical Parity Difference": {"color": "#994455", "marker": "."},
    "Equal Opportunity Difference": {"color": "#004488", "marker": "v"},
    "Equalized Odds Difference": {"color": "#117733", "marker": "^"},
    "Predictive Equality Difference": {"color": "#997700", "marker": "x"},
    "Negative Predictive Parity Difference": {"color": "#EECC66", "marker": "+"},
    "Positive Predictive Parity Difference": {"color": "#EE99AA", "marker": "o"},
    "Conditional Q Association": {"color": "#AA3377", "marker": "D"},
    "Phi Fair": {"color": "#BBBBBB", "marker": "s"},
}


def ratio_label(value: float, *, max_denominator: int = 500) -> str:
    frac = Fraction(float(value)).limit_denominator(max_denominator)
    if abs(float(frac) - float(value)) < 1e-9 and frac.denominator != 1:
        return f"{frac.numerator}/{frac.denominator}"
    if abs(float(value) - round(float(value))) < 1e-9:
        return str(int(round(float(value))))
    return f"{float(value):.2f}"



def decimal_ratio_labels(values: Iterable[float]) -> list[str]:
    return [f"{float(value):.2f}".rstrip("0").rstrip(".") for value in values]



def _metric_style(label: str, idx: int) -> dict[str, object]:
    return STYLE_OVERRIDES.get(label, {"color": COLOURS[idx % len(COLOURS)], "marker": "o"})



def _axis_label(ratio_type: str, group_ratio_basis: str = "j") -> str:
    if ratio_type in ("ir", "sr", "sr_n", "sr_c"):
        return X_LABELS.get(ratio_type, ratio_type.upper())
    return f"{X_LABELS.get(ratio_type, ratio_type.upper())} ({group_ratio_basis}-basis)"



def plot_histogram_grid(
    df: pd.DataFrame,
    metric_key: str,
    metric_label: str,
    gr_values: list[float],
    ir_values: list[float],
    *,
    bins: int = 109,
    group_ratio_basis: str = "j",
    show_nan_bar: bool = True,
) -> plt.Figure:
    df = ensure_metric_column(df, metric_key)
    gr_col = resolve_ratio_column("gr", group_ratio_basis)
    ir_col = resolve_ratio_column("ir", group_ratio_basis)

    if show_nan_bar:
        mosaic = [[f"a{i}{g}{suffix}" for g in range(len(gr_values)) for suffix in ("", "n")] for i in range(len(ir_values))]
        fig, axs = plt.subplot_mosaic(
            mosaic,
            width_ratios=[50, 1] * len(gr_values),
            sharex=False,
            sharey=True,
            layout="constrained",
            figsize=(4.0 * len(gr_values), 2.8 * len(ir_values)),
            gridspec_kw={"wspace": 0.1, "hspace": 0.1},
        )
    else:
        fig, axs_grid = plt.subplots(
            len(ir_values),
            len(gr_values),
            sharex=False,
            sharey=True,
            layout="constrained",
            figsize=(4.0 * len(gr_values), 2.8 * len(ir_values)),
            gridspec_kw={"wspace": 0.1, "hspace": 0.1},
        )
        if len(ir_values) == 1 and len(gr_values) == 1:
            axs_grid = np.array([[axs_grid]])
        elif len(ir_values) == 1:
            axs_grid = np.array([axs_grid])
        elif len(gr_values) == 1:
            axs_grid = np.array([[ax] for ax in axs_grid])

    fig.suptitle(metric_label)

    for i, ir_value in enumerate(ir_values):
        for g, gr_value in enumerate(gr_values):
            mask = np.isclose(df[ir_col], ir_value, atol=1e-9, rtol=0.0) & np.isclose(
                df[gr_col], gr_value, atol=1e-9, rtol=0.0
            )
            subset = df.loc[mask]
            total = len(subset)
            if show_nan_bar:
                ax = axs[f"a{i}{g}"]
                ax_nan = axs[f"a{i}{g}n"]
            else:
                ax = axs_grid[i, g]
                ax_nan = None

            if total == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            else:
                values = subset[metric_key]
                finite_values = values[np.isfinite(values)]
                nan_prob = float(values.isna().mean())
                if len(finite_values) > 0:
                    binned, edges = np.histogram(finite_values, bins=bins)
                    binned = binned / total
                    ax.hist(edges[:-1], edges, weights=binned, fc="black", ec="black")
                else:
                    ax.text(0.5, 0.5, "All values undefined", ha="center", va="center", transform=ax.transAxes)
                if ax_nan is not None:
                    ax_nan.bar(0, nan_prob, fc="red", ec="red", width=0.1, lw=0)
                    ax_nan.spines[["top", "left"]].set_visible(False)
                    if i == len(ir_values) - 1:
                        ax_nan.set_xticks([0], ["Undef."])
                    else:
                        ax_nan.set_xticks([0], [""])

            ax.spines[["top", "right"]].set_visible(False)
            if g == 0:
                ax.set_ylabel(f"IR = {ratio_label(ir_value)}")
            if i == 0:
                ax.set_title(f"GR = {ratio_label(gr_value)}")
            if i != len(ir_values) - 1:
                ax.set_xticklabels([])

    return fig



def plot_probability_lines(
    df: pd.DataFrame,
    metric_keys: list[str],
    metric_label_map: dict[str, str],
    ratio_type: str,
    *,
    title: str,
    y_label: str,
    group_ratio_basis: str = "j",
    y_max: float | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 8))
    x = df[ratio_type].to_numpy(dtype=np.float64)
    for idx, key in enumerate(metric_keys):
        label = metric_label_map.get(key, key)
        style = _metric_style(label, idx)
        ax.plot(x, df[key].to_numpy(dtype=np.float64), label=label, alpha=0.6, **style)

    if y_max is not None:
        ax.set_ylim(0, y_max)
    ax.set_xlabel(_axis_label(ratio_type, group_ratio_basis))
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.tight_layout()
    return fig



def plot_metric_vs_performance_heatmap(
    df: pd.DataFrame,
    fairness_key: str,
    fairness_label: str,
    performance_key: str,
    performance_label: str,
    *,
    bins: int = 100,
) -> plt.Figure:
    heat, x_edges, y_edges = value_grid_for_heatmap(df, fairness_key, performance_key, bins=bins)
    fig, ax = plt.subplots(figsize=(9, 8))
    mesh = ax.pcolormesh(x_edges, y_edges, heat, shading="auto")
    ax.set_xlabel(fairness_label)
    ax.set_ylabel(performance_label)
    ax.set_title(f"{fairness_label} vs {performance_label}")
    fig.colorbar(mesh, ax=ax, label="Proportion of confusion matrices")
    fig.tight_layout()
    return fig



def _case_plot_stats(
    fairness_df: pd.DataFrame,
    metric_key: str,
    ratio_type: str,
    *,
    other_ratio_fixed: float = 0.5,
    absolute: bool = False,
) -> tuple[list[float], list[str], dict[tuple[float, str], float], dict[tuple[float, str], float], dict[tuple[float, str], float]]:
    ratios = sorted(pd.unique(fairness_df[ratio_type]))
    clfs = list(pd.unique(fairness_df["clf"]))
    other_ratio = "gr" if ratio_type == "ir" else "ir"
    mean: dict[tuple[float, str], float] = {}
    stdev: dict[tuple[float, str], float] = {}
    err: dict[tuple[float, str], float] = {}

    for ratio in ratios:
        for clf in clfs:
            subset = fairness_df[
                (fairness_df[ratio_type] == ratio)
                & (fairness_df["clf"] == clf)
                & (fairness_df[other_ratio] == other_ratio_fixed)
                & (fairness_df["metric"] == metric_key)
                & fairness_df["value"].notna()
            ]["value"]
            if absolute:
                subset = subset.abs()
            mean[(ratio, clf)] = float(subset.mean()) if len(subset) else np.nan
            stdev[(ratio, clf)] = float(subset.std()) if len(subset) else np.nan
            err[(ratio, clf)] = float(scipy.stats.sem(subset, nan_policy="omit")) if len(subset) > 1 else np.nan
    return ratios, clfs, mean, stdev, err



def plot_case_line(
    fairness_df: pd.DataFrame,
    metric_key: str,
    metric_label: str,
    ratio_type: str,
    *,
    fill: str = "std",
    ylim: tuple[float, float] | None = (-0.9, 0.9),
) -> plt.Figure:
    ratios, clfs, mean, stdev, err = _case_plot_stats(fairness_df, metric_key, ratio_type)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_ylabel(metric_label)
    ax.set_xlabel(ratio_type.upper())
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)

    for idx, clf in enumerate(clfs):
        y = [mean[(ratio, clf)] for ratio in ratios]
        ax.plot(ratios, y, label=clf, color=COLOURS[idx % len(COLOURS)], marker="o")
        if fill == "err":
            low = [mean[(ratio, clf)] - err[(ratio, clf)] for ratio in ratios]
            high = [mean[(ratio, clf)] + err[(ratio, clf)] for ratio in ratios]
        else:
            low = [mean[(ratio, clf)] - stdev[(ratio, clf)] for ratio in ratios]
            high = [mean[(ratio, clf)] + stdev[(ratio, clf)] for ratio in ratios]
        ax.fill_between(ratios, low, high, alpha=0.15, color=COLOURS[idx % len(COLOURS)])

    ax.legend(loc=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(ratios, decimal_ratio_labels(ratios), rotation=90)
    ax.set_xlim(0, 1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    return fig



def plot_case_line_abs(
    fairness_df: pd.DataFrame,
    metric_key: str,
    metric_label: str,
    ratio_type: str,
    *,
    fill: str = "std",
    ylim: tuple[float, float] | None = (0.0, 0.6),
) -> plt.Figure:
    ratios, clfs, mean, stdev, err = _case_plot_stats(fairness_df, metric_key, ratio_type, absolute=True)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_ylabel(f"|{metric_label}|")
    ax.set_xlabel(ratio_type.upper())

    for idx, clf in enumerate(clfs):
        y = [mean[(ratio, clf)] for ratio in ratios]
        ax.plot(ratios, y, label=clf, color=COLOURS[idx % len(COLOURS)], marker="o")
        if fill == "err":
            low = [mean[(ratio, clf)] - err[(ratio, clf)] for ratio in ratios]
            high = [mean[(ratio, clf)] + err[(ratio, clf)] for ratio in ratios]
        else:
            low = [mean[(ratio, clf)] - stdev[(ratio, clf)] for ratio in ratios]
            high = [mean[(ratio, clf)] + stdev[(ratio, clf)] for ratio in ratios]
        ax.fill_between(ratios, low, high, alpha=0.15, color=COLOURS[idx % len(COLOURS)])

    ax.legend(loc=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(ratios, decimal_ratio_labels(ratios), rotation=90)
    ax.set_xlim(0, 1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    return fig



def plot_case_nan(
    fairness_df: pd.DataFrame,
    metric_keys: list[str],
    metric_label_map: dict[str, str],
    ratio_type: str,
    *,
    ylim: tuple[float, float] | None = None,
) -> plt.Figure:
    clfs = list(pd.unique(fairness_df["clf"]))
    ratios = sorted(pd.unique(fairness_df[ratio_type]))
    other_ratio = "gr" if ratio_type == "ir" else "ir"
    n_cols = 2
    n_rows = int(np.ceil(len(metric_keys) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(12, 4.5 * n_rows))
    axs = np.atleast_2d(axs)

    for idx, metric_key in enumerate(metric_keys):
        ax = axs[idx // n_cols, idx % n_cols]
        ax.set_title(metric_label_map.get(metric_key, metric_key))
        ax.set_ylabel("NaN probability")
        ax.set_xlabel(ratio_type.upper())
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.spines[["top", "right"]].set_visible(False)
        for clf_idx, clf in enumerate(clfs):
            subset = fairness_df[
                (fairness_df["clf"] == clf)
                & (fairness_df[other_ratio] == 0.5)
                & (fairness_df["metric"] == metric_key)
            ]
            counts = subset.groupby(ratio_type)["value"].apply(lambda x: x.isna().sum() / x.shape[0])
            y = [float(counts.get(ratio, np.nan)) for ratio in ratios]
            ax.plot(ratios, y, label=clf, color=COLOURS[clf_idx % len(COLOURS)], marker="o", alpha=0.6)

    # hide unused axes
    for idx in range(len(metric_keys), n_rows * n_cols):
        axs[idx // n_cols, idx % n_cols].axis("off")

    if ylim is not None:
        axs[0, 0].set_ylim(*ylim)
    axs[0, 0].set_xlim(0, 1)
    axs[0, 0].legend(loc=0)
    fig.tight_layout()
    return fig



def plot_case_line_all(
    fairness_df: pd.DataFrame,
    metric_keys: list[str],
    metric_label_map: dict[str, str],
    ratio_type: str,
    *,
    fill: str = "std",
    ylim: tuple[float, float] | None = (-0.9, 0.9),
) -> plt.Figure:
    n_cols = 2
    n_rows = int(np.ceil(len(metric_keys) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(12, 4.5 * n_rows))
    axs = np.atleast_2d(axs)

    clfs = list(pd.unique(fairness_df["clf"]))
    ratios = sorted(pd.unique(fairness_df[ratio_type]))
    other_ratio = "gr" if ratio_type == "ir" else "ir"

    for idx, metric_key in enumerate(metric_keys):
        ax = axs[idx // n_cols, idx % n_cols]
        label = metric_label_map.get(metric_key, metric_key)
        ax.set_ylabel(label.replace("Difference", "").strip())
        ax.axhline(0, color="black", linestyle="--", alpha=0.9, lw=1)

        mean: dict[tuple[float, str], float] = {}
        stdev: dict[tuple[float, str], float] = {}
        err: dict[tuple[float, str], float] = {}
        for ratio in ratios:
            for clf in clfs:
                subset = fairness_df[
                    (fairness_df[ratio_type] == ratio)
                    & (fairness_df["clf"] == clf)
                    & (fairness_df[other_ratio] == 0.5)
                    & (fairness_df["metric"] == metric_key)
                    & fairness_df["value"].notna()
                ]["value"]
                mean[(ratio, clf)] = float(subset.mean()) if len(subset) else np.nan
                stdev[(ratio, clf)] = float(subset.std()) if len(subset) else np.nan
                err[(ratio, clf)] = float(scipy.stats.sem(subset, nan_policy="omit")) if len(subset) > 1 else np.nan

        for clf_idx, clf in enumerate(clfs):
            y = [mean[(ratio, clf)] for ratio in ratios]
            ax.plot(ratios, y, label=clf, color=COLOURS[clf_idx % len(COLOURS)], marker="o", lw=1, alpha=0.85)
            if fill == "err":
                low = [mean[(ratio, clf)] - err[(ratio, clf)] for ratio in ratios]
                high = [mean[(ratio, clf)] + err[(ratio, clf)] for ratio in ratios]
            else:
                low = [mean[(ratio, clf)] - stdev[(ratio, clf)] for ratio in ratios]
                high = [mean[(ratio, clf)] + stdev[(ratio, clf)] for ratio in ratios]
            ax.fill_between(ratios, low, high, alpha=0.15, color=COLOURS[clf_idx % len(COLOURS)])

        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(ratios, decimal_ratio_labels(ratios), rotation=90)
        ax.set_xlim(0, 1)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel(X_LABELS[ratio_type])
        if ylim is not None:
            ax.set_ylim(*ylim)

    for idx in range(len(metric_keys), n_rows * n_cols):
        axs[idx // n_cols, idx % n_cols].axis("off")

    axs[0, 0].legend(loc=1, ncols=min(3, len(clfs)))
    fig.tight_layout()
    return fig



def plot_case_grouped_bar_by_metric(
    fairness_map: dict[str, dict[str, float]],
    *,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 8))
    metric_labels = list(next(iter(fairness_map.values())).keys())
    xticks = np.arange(len(metric_labels))
    width = 1.0 / (len(fairness_map) + 2)

    for idx, (clf_name, metric_values) in enumerate(fairness_map.items()):
        ax.bar(xticks + idx * width, list(metric_values.values()), width, label=clf_name, color=COLOURS[idx % len(COLOURS)])

    ax.set_title(title)
    ax.set_ylabel("Fairness metric value")
    wrapped_labels = ["\n".join([" ".join(label.split(" ")[:2]), " ".join(label.split(" ")[2:])]).strip() for label in metric_labels]
    ax.set_xticks(xticks + width * len(fairness_map) / 2, wrapped_labels, rotation=45)
    ax.legend(ncols=1)
    fig.tight_layout()
    return fig



def plot_case_grouped_bar_by_classifier(
    fairness_map: dict[str, dict[str, float]],
    *,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 12))
    metric_labels = list(next(iter(fairness_map.values())).keys())
    clf_names = list(fairness_map.keys())
    xticks = np.arange(len(clf_names))
    width = 1.0 / (len(metric_labels) + 2)
    shift = np.arange(len(metric_labels)) * width

    for idx, (_, metric_values) in enumerate(fairness_map.items()):
        ax.bar(idx + shift, list(metric_values.values()), width, color=COLOURS[: len(metric_labels)])

    ax.set_title(title)
    ax.set_ylabel("Fairness metric value")
    ax.set_xticks(xticks + width * len(metric_labels) / 2, clf_names)
    ax.legend(handles=[mpatches.Patch(color=COLOURS[idx], label=label) for idx, label in enumerate(metric_labels)], ncol=1)
    fig.tight_layout()
    return fig




def plot_sr_sensitivity(
    sensitivity_df: pd.DataFrame,
    metric_label_map: dict[str, str],
    *,
    value_col: str = "spearman_r",
    title: str = "Metric sensitivity to SR (Spearman ρ)",
) -> plt.Figure:
    """Horizontal bar chart showing how strongly each metric correlates with SR."""
    df = sensitivity_df.dropna(subset=[value_col]).copy()
    df["label"] = df["metric"].map(lambda k: metric_label_map.get(k, k))
    df = df.sort_values(value_col, ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.55)))
    colors = ["#EE3377" if v < 0 else "#33BBEE" for v in df[value_col]]
    bars = ax.barh(df["label"], df[value_col], color=colors, alpha=0.8, edgecolor="none")

    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, df[value_col]):
        if not np.isfinite(val):
            continue
        offset = 0.02 if val >= 0 else -0.02
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha=ha, fontsize=8)

    fig.tight_layout()
    return fig


def plot_histogram_grid_sr(
    df: pd.DataFrame,
    metric_key: str,
    metric_label: str,
    sr_values: list[float],
    *,
    sr_col: str = "stereotypical_ratio",
    bins: int = 50,
    show_nan_bar: bool = True,
) -> plt.Figure:
    """Row of histograms of *metric_key* at each given SR value."""
    from synthetic_analysis import ensure_metric_column
    df = ensure_metric_column(df, metric_key)
    sr_values = [sv for sv in sr_values if np.isfinite(sv)]

    n = len(sr_values)
    width_ratios = ([30, 1] * n) if show_nan_bar else ([1] * n)
    ncols = n * 2 if show_nan_bar else n

    fig, axs = plt.subplots(
        1, ncols,
        sharey=True,
        figsize=(3.5 * n, 3.8),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.05},
    )
    if ncols == 1:
        axs = [axs]

    sr_display = {"stereotypical_ratio": "SR_p", "stereotypical_ratio_negative": "SR_n", "stereotypical_ratio_combined": "SR_c"}.get(sr_col, "SR")
    fig.suptitle(f"{metric_label}  [{sr_display}]", y=1.02)

    for g, sv in enumerate(sr_values):
        ax = axs[g * 2] if show_nan_bar else axs[g]
        ax_nan = axs[g * 2 + 1] if show_nan_bar else None

        mask = np.isclose(df[sr_col].to_numpy(np.float64), float(sv), atol=1e-9, rtol=0)
        subset = df.loc[mask]
        total = len(subset)

        ax.set_title(f"{sr_display} = {ratio_label(sv)}")
        if total == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        else:
            values = subset[metric_key]
            finite_vals = values[np.isfinite(values)]
            nan_prob = float(values.isna().mean())
            if len(finite_vals) > 0:
                binned, edges = np.histogram(finite_vals, bins=bins)
                binned = binned / total
                ax.hist(edges[:-1], edges, weights=binned, fc="black", ec="black")
            else:
                ax.text(0.5, 0.5, "All undefined", ha="center", va="center", transform=ax.transAxes)
            if ax_nan is not None:
                ax_nan.bar(0, nan_prob, fc="red", ec="red", width=0.1, lw=0)
                ax_nan.set_xticks([0], ["Undef."])
                ax_nan.spines[["top", "left"]].set_visible(False)

        ax.spines[["top", "right"]].set_visible(False)
        if g > 0:
            ax.set_yticklabels([])

    fig.tight_layout()
    return fig


def plot_discrimination_sweep(
    df: pd.DataFrame,
    metric_keys: list[str],
    label_map: dict[str, str],
    disc_type: str = "tpr_gap",
    *,
    absolute: bool = False,
) -> plt.Figure:
    """Line plot: metric mean (±std) vs true_delta, one line per metric."""
    disc_axis = {
        "tpr_gap": "True discrimination δ  (TPR gap: TPR_j − TPR_i)",
        "fpr_gap": "True discrimination δ  (FPR gap: FPR_j − FPR_i)",
        "both":    "True discrimination δ  (TPR gap = FPR gap = δ)",
    }
    fig, ax = plt.subplots(figsize=(9, 7))
    deltas = np.sort(df["true_delta"].unique())

    for idx, key in enumerate(metric_keys):
        if key not in df.columns:
            continue
        label = label_map.get(key, key)
        style = _metric_style(label, idx)
        means, stds = [], []
        for d in deltas:
            vals = df.loc[df["true_delta"] == d, key].dropna()
            if absolute:
                vals = vals.abs()
            means.append(float(vals.mean()) if len(vals) else np.nan)
            stds.append(float(vals.std()) if len(vals) > 1 else 0.0)
        means_arr = np.array(means, dtype=np.float64)
        stds_arr = np.array(stds, dtype=np.float64)
        ax.plot(deltas, means_arr, label=label, alpha=0.85, **style)
        ax.fill_between(deltas, means_arr - stds_arr, means_arr + stds_arr,
                        alpha=0.12, color=style["color"])

    ax.axhline(0, color="black", linestyle="--", alpha=0.3, lw=1)
    ax.axvline(0, color="black", linestyle="--", alpha=0.3, lw=1)
    ax.set_xlabel(disc_axis.get(disc_type, "True discrimination δ"))
    ax.set_ylabel("|Metric| mean ± std" if absolute else "Metric mean ± std")
    ax.set_title("Metric response to injected discrimination")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_detection_power_bars(
    bench_df: pd.DataFrame,
    label_map: dict[str, str],
) -> plt.Figure:
    """Horizontal grouped bar: detection power (blue) vs false alarm rate (red)."""
    df = bench_df.dropna(subset=["detection_power", "false_alarm_rate"]).copy()
    df["label"] = df["metric"].map(lambda k: label_map.get(k, k))
    df = df.sort_values("detection_power", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.75)))
    y = np.arange(len(df))
    h = 0.35
    ax.barh(y + h / 2, df["detection_power"], h, label="Detection power",
            color="#33BBEE", alpha=0.85, edgecolor="none")
    ax.barh(y - h / 2, df["false_alarm_rate"], h, label="False alarm rate",
            color="#EE3377", alpha=0.85, edgecolor="none")

    for i, (dp, fa) in enumerate(zip(df["detection_power"], df["false_alarm_rate"])):
        if np.isfinite(dp):
            ax.text(dp + 0.01, i + h / 2, f"{dp:.2f}", va="center", fontsize=7)
        if np.isfinite(fa):
            ax.text(fa + 0.01, i - h / 2, f"{fa:.2f}", va="center", fontsize=7)

    ax.set_yticks(y, df["label"])
    ax.set_xlabel("Probability")
    ax.set_xlim(0, 1.15)
    ax.set_title("Detection power vs false alarm rate")
    ax.axvline(0, color="black", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_roc_curves(
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    label_map: dict[str, str],
) -> plt.Figure:
    """ROC curves for discrimination detection, one line per metric."""
    fig, ax = plt.subplots(figsize=(9, 8))
    sorted_items = sorted(roc_data.items(), key=lambda kv: kv[1][2], reverse=True)
    for idx, (key, (fpr, tpr, auc_val)) in enumerate(sorted_items):
        label_text = label_map.get(key, key)
        style = _metric_style(label_text, idx)
        ax.plot(fpr, tpr, label=f"{label_text}  (AUC = {auc_val:.3f})", alpha=0.85, **style)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (detection rate)")
    ax.set_title("ROC curves — discrimination detection")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig


def plot_detection_vs_ir(
    ir_bench_df: pd.DataFrame,
    metric_keys: list[str],
    label_map: dict[str, str],
    metric_col: str = "detection_power",
) -> plt.Figure:
    """Line plot: detection metric vs IR, one line per fairness metric."""
    col_labels = {
        "detection_power": "Detection power",
        "false_alarm_rate": "False alarm rate",
        "auc": "ROC AUC",
        "spearman_r": "Spearman ρ",
    }
    ir_values = sorted(ir_bench_df["ir"].unique())
    fig, ax = plt.subplots(figsize=(9, 7))

    for idx, key in enumerate(metric_keys):
        subset = ir_bench_df[ir_bench_df["metric"] == key]
        if subset.empty:
            continue
        label = label_map.get(key, key)
        style = _metric_style(label, idx)
        y = []
        for ir in ir_values:
            row = subset[np.isclose(subset["ir"].to_numpy(np.float64), ir, atol=1e-9)]
            y.append(float(row[metric_col].iloc[0]) if not row.empty else np.nan)
        ax.plot(ir_values, y, label=label, alpha=0.85, **style)

    ax.axhline(0.5, color="black", linestyle="--", alpha=0.25, lw=1)
    ax.set_xlabel("Imbalance Ratio (IR)")
    ax.set_ylabel(col_labels.get(metric_col, metric_col))
    ax.set_title(f"{col_labels.get(metric_col, metric_col)} vs Imbalance Ratio")
    ax.set_xlim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_metric_vs_sr_by_ir(
    df: pd.DataFrame,
    metric_key: str,
    metric_label: str,
    ir_values: list[float],
    *,
    absolute: bool = False,
    show_bands: bool = True,
    gr_value: float | None = None,
    sr_label: str = "SR",
) -> plt.Figure:
    """Line plot of metric mean vs SR with one line per IR value."""
    fig, ax = plt.subplots(figsize=(9, 8))
    std_col = f"{metric_key}_std"

    for idx, ir_val in enumerate(ir_values):
        subset = df[np.isclose(df["ir_value"].to_numpy(np.float64), ir_val, atol=1e-9, rtol=0)]
        if subset.empty:
            continue
        x = subset["sr"].to_numpy(dtype=np.float64)
        y = subset[metric_key].to_numpy(dtype=np.float64)
        color = COLOURS[idx % len(COLOURS)]
        ax.plot(x, y, label=f"IR = {ratio_label(ir_val)}", color=color, marker="o", lw=1.5, alpha=0.85)
        if show_bands and std_col in subset.columns:
            std = subset[std_col].to_numpy(dtype=np.float64)
            ax.fill_between(x, y - std, y + std, alpha=0.12, color=color)

    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel(f"Stereotypical Ratio ({sr_label})")
    ax.set_ylabel(f"|{metric_label}|" if absolute else metric_label)
    title = metric_label
    if gr_value is not None:
        title += f"  [GR = {ratio_label(gr_value)}]"
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend(loc=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


