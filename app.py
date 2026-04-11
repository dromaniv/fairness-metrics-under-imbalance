from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image as _PIL_Image

_PIL_Image.MAX_IMAGE_PIXELS = None  # allow large composite figures

import builtin_metrics  # noqa: F401 - registers built-in metrics
try:
    import custom_metrics  # noqa: F401 - optional user-defined metrics
except Exception as exc:  # pragma: no cover - Streamlit feedback only
    CUSTOM_METRIC_IMPORT_ERROR = str(exc)
else:
    CUSTOM_METRIC_IMPORT_ERROR = None

from adult_case_study import (
    CLASSIFIERS,
    aggregate_case_results,
    collect_adult_confusion_matrices,
    evaluate_case_study,
    load_adult_dataset,
    paper_ratio_sweep,
)
from fairness_benchmark import benchmark_metrics, sweep_discrimination
from metric_registry import list_metrics, compute_metrics, COUNT_COLUMNS
from plots import (
    plot_case_grouped_bar_by_classifier,
    plot_case_grouped_bar_by_metric,
    plot_case_line,
    plot_case_line_all,
    plot_case_nan,
    plot_detection_power_bars,
    plot_discrimination_sweep,
    plot_histogram_grid,
    plot_histogram_grid_sr,
    plot_metric_vs_sr_by_ir,
    plot_metric_vs_performance_heatmap,
    plot_probability_lines,
    plot_sr_sensitivity,
    ratio_label,
)
from stereotypical_study import (
    compute_sr_sensitivity_stratified,
    metric_means_by_sr_multi_ir,
    SR_COLUMNS,
    SR_LABELS,
)
from synthetic_analysis import probability_of_nan, probability_of_perfect_fairness
from synthetic_data import (
    add_base_columns,
    count_confusion_matrices,
    dump_confusion_matrices_to_pickle,
    generate_exact_confusion_matrices,
    load_confusion_matrices_from_pickle,
    paper_ratio_defaults,
    ratio_values,
    sample_uniform_confusion_matrices,
)


@st.cache_data(show_spinner=False)
def _cached_sr_sensitivity_stratified(df: pd.DataFrame, metric_keys: tuple[str, ...], sr_col: str) -> pd.DataFrame:
    """Cache stratified sensitivity so switching unrelated widgets doesn't recompute it."""
    return compute_sr_sensitivity_stratified(df, list(metric_keys), sr_col=sr_col)


@st.cache_data(show_spinner=False)
def _cached_metric_means_by_sr_multi_ir(
    df: pd.DataFrame,
    metric_key: str,
    ir_values: tuple[float, ...],
    sr_col: str,
    gr_value: float,
    atol: float,
) -> pd.DataFrame:
    return metric_means_by_sr_multi_ir(df, metric_key, list(ir_values), sr_col=sr_col, gr_value=gr_value, atol=atol)


st.set_page_config(page_title="Fairness Measures Explorer", layout="wide", page_icon="⚖️")


def nearest_available_ratios(total: int, targets: list[float]) -> list[float]:
    available = ratio_values(total)
    chosen: list[float] = []
    for target in targets:
        best = min(available, key=lambda value: abs(value - target))
        if best not in chosen:
            chosen.append(best)
    return chosen


_SMOOTHABLE_METRICS = {"conditional_q_association", "conditional_y_association"}


def apply_smoothing_override(df: pd.DataFrame, metric_key: str, smoothing: bool) -> pd.DataFrame:
    if metric_key not in _SMOOTHABLE_METRICS:
        return df
    from custom_metrics import conditional_q_association, conditional_y_association
    fn = conditional_q_association if metric_key == "conditional_q_association" else conditional_y_association
    out = df.copy()
    out[metric_key] = fn(out, smoothing=smoothing)
    return out


def smoothing_toggle(metric_key: str, widget_key: str) -> bool:
    """Render a smoothing checkbox when metric_key is smoothable; return the current value."""
    if metric_key in _SMOOTHABLE_METRICS:
        return st.checkbox(
            "Haldane-Anscombe smoothing (+0.5)",
            value=True,
            key=widget_key,
            help="Adds 0.5 to each cell of the per-stratum 2×2 table before computing the odds ratio. "
                 "When enabled, CQA and CYA is always defined (even at IR=0 or IR=1).",
        )
    return True  # irrelevant for other metrics, value unused


# Mapping from raw metric key → its FRN-wrapped sibling.
_FRN_KEY_MAP: dict[str, str] = {
    "accuracy_equality_difference":          "frn_accd",
    "statistical_parity_difference":         "frn_spd",
    "equal_opportunity_difference":          "frn_eod",
    "predictive_equality_difference":        "frn_fprd",
    "positive_predictive_parity_difference": "frn_ppvd",
    "negative_predictive_parity_difference": "frn_npvd",
}


def resolve_frn_key(key: str, use_frn: bool) -> str:
    """Return the FRN variant of *key* when *use_frn* is True and a variant exists."""
    if use_frn and key in _FRN_KEY_MAP:
        return _FRN_KEY_MAP[key]
    return key


def frn_toggle(metric_key: str, widget_key: str) -> bool:
    """Render an FRN checkbox when *metric_key* has an FRN variant; return the current value."""
    if metric_key in _FRN_KEY_MAP:
        frn_key = _FRN_KEY_MAP[metric_key]
        try:
            frn_label = list_metrics("fairness_frn")
            frn_label = next(s.label for s in frn_label if s.key == frn_key)
        except StopIteration:
            frn_label = frn_key
        return st.checkbox(
            "Apply Feasible-Range Normalization",
            value=False,
            key=widget_key,
            help=(
                "Scales the metric by the maximum (or minimum) value it could reach given the "
                "current confusion-matrix marginals, so the result is always in [−1, 1].\n\n"
                f"Shows: **{frn_label}**"
            ),
        )
    return False


def apply_frn_to_keys(keys: list[str], use_frn: bool) -> list[str]:
    """Replace each key in *keys* with its FRN variant when *use_frn* is True."""
    return [resolve_frn_key(k, use_frn) for k in keys]


def _valid_fairness_keys(keys: list[str]) -> list[str]:
    """Filter out any keys no longer present in the registry (e.g. from stale session state).
    Accepts both 'fairness' and 'fairness_frn' category keys.
    """
    registered = {spec.key for spec in list_metrics("fairness")} | \
                 {spec.key for spec in list_metrics("fairness_frn")}
    return [k for k in keys if k in registered]


def dataframe_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def figure_png_bytes(fig) -> bytes:
    buffer = BytesIO()
    w, h = fig.get_size_inches()
    dpi = max(72, min(150, int((100_000_000 / max(w * h, 1)) ** 0.5)))
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    return buffer.read()


def fairness_metric_specs() -> list:
    return list_metrics("fairness")


def metric_selector(label: str, category: str, default_keys: list[str] | None = None):
    specs = list_metrics(category)
    options = [spec.key for spec in specs]
    label_map = {spec.key: spec.label for spec in specs}
    default = [k for k in (default_keys or options) if k in options]
    return st.multiselect(label, options=options, default=default, format_func=lambda key: label_map[key])


def _filter_degenerate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where either group has zero total observations."""
    i_total = df["i_tp"] + df["i_fp"] + df["i_tn"] + df["i_fn"]
    j_total = df["j_tp"] + df["j_fp"] + df["j_tn"] + df["j_fn"]
    return df[(i_total > 0) & (j_total > 0)].reset_index(drop=True)


def _render_data_table_tab(
    df: pd.DataFrame,
    extra_columns: list[str],
    label_map: dict[str, str],
    *,
    widget_prefix: str,
    show_sr: bool = False,
) -> None:
    """Shared data-table tab used by both the synthetic and stereotypical pages."""
    preview_keys = st.multiselect(
        "Additional metric columns",
        options=list(extra_columns),
        default=[],
        format_func=lambda k: label_map.get(k, k),
        key=f"{widget_prefix}_preview_metrics",
    )
    hide_degen = st.checkbox(
        "Hide rows where either group is empty",
        value=True,
        key=f"{widget_prefix}_hide_degen",
    )
    work = _filter_degenerate(df) if hide_degen else df.reset_index(drop=True)
    base = list(COUNT_COLUMNS)
    if show_sr:
        base += [c for c in [
            "stereotypical_ratio", "stereotypical_ratio_negative",
            "stereotypical_ratio_combined", "imbalance_ratio", "group_ratio_j",
        ] if c in work.columns]
    base_df = work[[c for c in base if c in work.columns]].reset_index(drop=True)
    if preview_keys:
        extra = compute_metrics(work, preview_keys).reset_index(drop=True)
        display_df = pd.concat([base_df, extra], axis=1)
    else:
        display_df = base_df
    st.caption(f"Showing {min(1000, len(display_df)):,} of {len(display_df):,} rows.")
    st.dataframe(display_df.head(1000), use_container_width=True)


def _build_synthetic_dataset(
    mode: str,
    total: int,
    max_rows: int,
    draws: int,
    seed: int,
    pickle_path: str,
) -> pd.DataFrame:
    """Build a synthetic confusion-matrix DataFrame from the given parameters."""
    if mode == "Exact enumeration":
        return add_base_columns(generate_exact_confusion_matrices(int(total), max_rows=int(max_rows)))
    elif mode == "Monte Carlo sample":
        return add_base_columns(sample_uniform_confusion_matrices(int(total), int(draws), seed=int(seed)))
    else:
        if not pickle_path.strip():
            raise ValueError("Enter a path to a pickle file first.")
        return add_base_columns(load_confusion_matrices_from_pickle(pickle_path.strip()))


def render_synthetic_page() -> None:
    st.header("Synthetic study")
    st.write(
        "Generate or sample the full space of 8-cell confusion matrices, compute fairness metrics, and reproduce the paper's synthetic plots."
    )

    with st.sidebar:
        st.subheader("Synthetic dataset")
        synth_mode = st.radio(
            "Dataset source",
            options=["Exact enumeration", "Monte Carlo sample", "Load pickle"],
            key="synthetic_mode",
        )
        total = st.number_input("n (total samples)", min_value=1, value=24, step=1, key="synthetic_total_input")
        estimated_rows = count_confusion_matrices(int(total))
        st.caption(f"All possible confusion matrices: {estimated_rows:,}")
        if estimated_rows > 20_000_000:
            st.warning("Exact generation at this n is likely too heavy for an interactive session. Use sampling or reduce n.")

        max_exact_rows = st.number_input(
            "Exact-generation row cap",
            min_value=1000,
            value=20_000_000,
            step=1000,
            key="synthetic_exact_cap",
        )
        monte_carlo_draws = st.number_input(
            "Monte Carlo draws",
            min_value=100,
            value=200_000,
            step=100,
            key="synthetic_draws",
        )
        seed = st.number_input("Random seed", min_value=0, value=2137, step=1, key="synthetic_seed")
        pickle_path = st.text_input(
            "Pickle path (Set(08,n).bin or saved by this app)",
            value="",
            key="synthetic_pickle_path",
        )

        if st.button("Build synthetic dataset", type="primary"):
            try:
                df = _build_synthetic_dataset(
                    synth_mode, int(total), int(max_exact_rows),
                    int(monte_carlo_draws), int(seed), pickle_path,
                )
                built_total = int(df[COUNT_COLUMNS].iloc[0].sum()) if synth_mode == "Load pickle" else int(total)
                st.session_state["synthetic_df"] = df
                st.session_state["synthetic_total"] = built_total
                st.success("Synthetic dataset ready.")
            except Exception as exc:
                st.error(str(exc))

    synthetic_df = st.session_state.get("synthetic_df")
    synthetic_total = int(st.session_state.get("synthetic_total", total))
    if synthetic_df is None:
        st.info("Build a synthetic dataset from the sidebar to begin.")
        return

    st.subheader("Dataset summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(synthetic_df):,}")
    c2.metric("n", synthetic_total)
    c3.metric("Unique IR values", synthetic_df["imbalance_ratio"].nunique())
    c4.metric("Unique GR values", synthetic_df["group_ratio_i"].nunique())

    with st.expander("Download current synthetic dataset"):
        st.download_button(
            "Download pickle",
            data=dump_confusion_matrices_to_pickle(synthetic_df),
            file_name=f"Set(08,{synthetic_total})_streamlit.pkl",
            mime="application/octet-stream",
        )
        st.download_button(
            "Download CSV",
            data=dataframe_csv_bytes(synthetic_df),
            file_name=f"synthetic_confusion_matrices_n{synthetic_total}.csv",
            mime="text/csv",
        )

    tabs = st.tabs(["Histogram grids", "Perfect fairness / NaN", "Fairness vs performance", "Data table"])
    fairness_specs = fairness_metric_specs()
    fairness_label_map = {spec.key: spec.label for spec in fairness_specs}

    with tabs[0]:
        st.subheader("Histogram grids")
        col1, col2, col3 = st.columns([1, 1, 1])
        metric_key = col1.selectbox(
            "Metric",
            options=[spec.key for spec in fairness_specs],
            format_func=lambda key: fairness_label_map[key],
            key="hist_metric_key",
        )
        group_ratio_basis = col2.radio(
            "Group ratio basis",
            options=["i", "j"],
            format_func=lambda value: f"{value}-group / total",
            horizontal=True,
            key="hist_gr_basis",
        )
        bins = int(col3.number_input("Histogram bins", min_value=5, value=109, step=1, key="hist_bins"))

        default_grid = nearest_available_ratios(synthetic_total, paper_ratio_defaults(synthetic_total))
        available_gr = sorted(ratio_values(synthetic_total))
        available_ir = sorted(ratio_values(synthetic_total))

        selected_gr = sorted(st.multiselect(
            "GR panel values",
            options=available_gr,
            default=default_grid,
            format_func=ratio_label,
            key="hist_selected_gr",
        ))
        selected_ir = sorted(st.multiselect(
            "IR panel values",
            options=available_ir,
            default=sorted(default_grid),
            format_func=ratio_label,
            key="hist_selected_ir",
        ))
        show_nan_bar = st.checkbox("Show separate undefined-value bar", value=True, key="hist_nan_bar")
        hist_smoothing = smoothing_toggle(metric_key, "hist_smoothing")
        hist_frn = frn_toggle(metric_key, "hist_frn")
        active_metric_key = resolve_frn_key(metric_key, hist_frn)

        if selected_gr and selected_ir:
            hist_df = apply_smoothing_override(synthetic_df, active_metric_key, hist_smoothing)
            fig = plot_histogram_grid(
                hist_df,
                active_metric_key,
                fairness_label_map.get(active_metric_key, fairness_label_map[metric_key]),
                selected_gr,
                selected_ir,
                bins=bins,
                group_ratio_basis=group_ratio_basis,
                show_nan_bar=show_nan_bar,
            )
            st.pyplot(fig, use_container_width=True)
            st.download_button(
                "Download histogram grid (PNG)",
                data=figure_png_bytes(fig),
                file_name=f"histogram_grid_{active_metric_key}.png",
                mime="image/png",
            )
        else:
            st.info("Pick at least one GR and one IR value.")

    with tabs[1]:
        st.subheader("Probability of perfect fairness and undefined values")
        col1, col2, col3, col4 = st.columns([1.4, 1, 1, 1])
        selected_metric_keys = col1.multiselect(
            "Metrics",
            options=[spec.key for spec in fairness_specs],
            default=[spec.key for spec in fairness_specs],
            format_func=lambda key: fairness_label_map[key],
            key="ppf_metric_keys",
        )
        ratio_type = col2.radio("Sweep", options=["ir", "gr"], horizontal=True, key="ppf_ratio_type")
        epsilon = float(col3.number_input("Epsilon for near-perfect fairness", min_value=0.0, value=0.0, step=0.001))
        ppf_basis = col4.radio(
            "GR basis",
            options=["i", "j"],
            horizontal=True,
            key="ppf_gr_basis",
        )
        ppf_smoothing = st.checkbox(
            "Haldane-Anscombe smoothing for Conditional Q Association",
            value=True,
            key="ppf_smoothing",
            help="Applies +0.5 smoothing when computing Conditional Q Association.",
        ) if any(k in _SMOOTHABLE_METRICS for k in selected_metric_keys) else True

        ppf_frn = st.checkbox(
            "Apply Feasible-Range Normalization to supported metrics",
            value=False,
            key="ppf_frn",
            help=(
                "Replaces each metric that has a feasible-range normalized variant with that version. "
                "Supported: " + ", ".join(
                    s.label for s in list_metrics("fairness") if s.key in _FRN_KEY_MAP
                )
            ),
        ) if any(k in _FRN_KEY_MAP for k in selected_metric_keys) else False

        active_ppf_keys = apply_frn_to_keys(selected_metric_keys, ppf_frn)
        # rebuild label map to include any FRN keys that replaced raw ones
        all_ppf_label_map = {**fairness_label_map, **{spec.key: spec.label for spec in list_metrics("fairness_frn")}}

        if selected_metric_keys:
            ppf_work_df = synthetic_df
            for k in active_ppf_keys:
                ppf_work_df = apply_smoothing_override(ppf_work_df, k, ppf_smoothing)
            ppf_df = probability_of_perfect_fairness(
                ppf_work_df,
                active_ppf_keys,
                ratio_type,
                epsilon=epsilon,
                group_ratio_basis=ppf_basis,
            )
            nan_df = probability_of_nan(
                ppf_work_df,
                active_ppf_keys,
                ratio_type,
                group_ratio_basis=ppf_basis,
            )
            fig1 = plot_probability_lines(
                ppf_df,
                active_ppf_keys,
                all_ppf_label_map,
                ratio_type,
                title="Probability of perfect fairness",
                y_label="Probability of perfect fairness",
                group_ratio_basis=ppf_basis,
                y_max=1.0 if ratio_type == "ir" else None,
            )
            fig2 = plot_probability_lines(
                nan_df,
                active_ppf_keys,
                all_ppf_label_map,
                ratio_type,
                title="Probability of undefined values",
                y_label="Probability of undefined metric value",
                group_ratio_basis=ppf_basis,
                y_max=1.0,
            )
            left, right = st.columns(2)
            left.pyplot(fig1, use_container_width=True)
            right.pyplot(fig2, use_container_width=True)
            st.download_button(
                "Download perfect-fairness CSV",
                data=dataframe_csv_bytes(ppf_df),
                file_name=f"perfect_fairness_{ratio_type}.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download undefined-value CSV",
                data=dataframe_csv_bytes(nan_df),
                file_name=f"undefined_probability_{ratio_type}.csv",
                mime="text/csv",
            )

    with tabs[2]:
        st.subheader("Fairness vs predictive performance")
        col1, col2, col3 = st.columns([1, 1, 1])
        fairness_key = col1.selectbox(
            "Fairness metric",
            options=[spec.key for spec in fairness_specs],
            format_func=lambda key: fairness_label_map[key],
            key="heatmap_fairness_key",
        )
        performance_specs = list_metrics("performance")
        performance_label_map = {spec.key: spec.label for spec in performance_specs}
        performance_key = col2.selectbox(
            "Performance measure",
            options=[spec.key for spec in performance_specs],
            format_func=lambda key: performance_label_map[key],
            key="heatmap_performance_key",
        )
        heat_bins = int(col3.number_input("Heatmap bins", min_value=10, value=100, step=10, key="heatmap_bins"))
        heat_smoothing = smoothing_toggle(fairness_key, "heatmap_smoothing")
        heat_frn = frn_toggle(fairness_key, "heatmap_frn")
        active_fairness_key = resolve_frn_key(fairness_key, heat_frn)
        heat_df = apply_smoothing_override(synthetic_df, active_fairness_key, heat_smoothing)
        fig = plot_metric_vs_performance_heatmap(
            heat_df,
            active_fairness_key,
            fairness_label_map.get(active_fairness_key, fairness_label_map[fairness_key]),
            performance_key,
            performance_label_map[performance_key],
            bins=heat_bins,
        )
        st.pyplot(fig, use_container_width=True)
        st.download_button(
            "Download heatmap (PNG)",
            data=figure_png_bytes(fig),
            file_name=f"heatmap_{active_fairness_key}_vs_{performance_key}.png",
            mime="image/png",
        )

    with tabs[3]:
        st.subheader("Data table")
        _render_data_table_tab(
            synthetic_df,
            [s.key for s in fairness_specs],
            fairness_label_map,
            widget_prefix="synthetic",
        )

def render_case_study_page() -> None:
    st.header("Adult case study")
    st.write(
        "Run the controlled Adult/Census Income experiment with varying imbalance ratio (IR) and group ratio (GR), mirroring the case study in the paper."
    )

    _prog_bar = st.empty()
    _prog_cap = st.empty()

    with st.sidebar:
        st.subheader("Adult data source")
        default_path = "data/adult.data"
        adult_source_mode = st.radio("Adult dataset source", options=["Local path", "Upload file"], key="adult_source_mode")
        uploaded = (
            st.file_uploader("adult.data", type=["data", "csv", "txt"], key="adult_upload")
            if adult_source_mode == "Upload file" else None
        )
        path_value = (
            st.text_input("Path to adult.data", value=default_path, key="adult_path")
            if adult_source_mode == "Local path" else default_path
        )

        st.subheader("Experiment controls")
        sweep_ratios = st.multiselect(
            "Ratios to sweep",
            options=paper_ratio_sweep(),
            default=paper_ratio_sweep(),
            format_func=lambda value: f"{value:.2f}",
            key="adult_ratio_values",
        )
        fixed_ratio = st.number_input("Fixed ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="adult_fixed_ratio")
        sample_size = st.number_input("Subset size", min_value=100, value=1100, step=100, key="adult_sample_size")
        holdout_splits = st.number_input("Holdout repetitions", min_value=1, value=50, step=1, key="adult_holdout_splits")
        test_size = st.slider("Test size", min_value=0.1, max_value=0.9, value=0.33, step=0.01, key="adult_test_size")
        random_state = st.number_input("Random seed", min_value=0, value=2137, step=1, key="adult_random_state")
        selected_classifiers = st.multiselect(
            "Classifiers",
            options=list(CLASSIFIERS.keys()),
            default=list(CLASSIFIERS.keys()),
            key="adult_classifiers",
        )
        selected_fairness_metrics = metric_selector(
            "Fairness metrics",
            "fairness",
            default_keys=[spec.key for spec in fairness_metric_specs()],
        )
        case_frn = st.checkbox(
            "Apply Feasible-Range Normalization to supported metrics",
            value=False,
            key="adult_frn",
            help=(
                "Replaces each selected metric that has a feasible-range normalized variant with its "
                "normalized version before running the experiment."
            ),
        )

        if st.button("Run Adult case study", type="primary"):
            try:
                if adult_source_mode == "Upload file":
                    if uploaded is None:
                        raise ValueError("Upload the Adult dataset first.")
                    adult_df = load_adult_dataset(uploaded.getvalue())
                else:
                    adult_df = load_adult_dataset(path_value)

                # Guard against stale widget state holding keys removed from the registry.
                validated_metrics = _valid_fairness_keys(
                    apply_frn_to_keys(selected_fairness_metrics, case_frn)
                )
                if not validated_metrics:
                    raise ValueError("No valid fairness metrics selected.")

                _prog_bar.progress(0.0, text="Starting…")

                def _progress(frac: float, msg: str) -> None:
                    _prog_bar.progress(min(frac, 1.0), text=msg)
                    _prog_cap.caption(msg)

                fairness_results, performance_results = evaluate_case_study(
                    adult_df,
                    ratio_values=sweep_ratios,
                    fixed_ratio=fixed_ratio,
                    sample_size=int(sample_size),
                    holdout_splits=int(holdout_splits),
                    test_size=float(test_size),
                    classifier_names=selected_classifiers,
                    fairness_metric_keys=validated_metrics,
                    random_state=int(random_state),
                    progress_callback=_progress,
                )
                _prog_bar.empty()
                _prog_cap.empty()
                st.session_state["adult_fairness_results"] = fairness_results
                st.session_state["adult_performance_results"] = performance_results
                st.success("Adult case study finished.")
            except Exception as exc:
                st.error(str(exc))

    fairness_results = st.session_state.get("adult_fairness_results")
    performance_results = st.session_state.get("adult_performance_results")
    if fairness_results is None or performance_results is None:
        st.info("Provide the Adult dataset in the sidebar and run the experiment.")
        return

    all_fairness_specs = list_metrics("fairness") + list_metrics("fairness_frn")
    fairness_label_map = {spec.key: spec.label for spec in all_fairness_specs}

    c1, c2, c3 = st.columns(3)
    c1.metric("Fairness rows", f"{len(fairness_results):,}")
    c2.metric("Performance rows", f"{len(performance_results):,}")
    c3.metric("Classifiers", len(pd.unique(fairness_results["clf"])))

    tabs = st.tabs(["Fairness lines", "NaN probability", "All metrics", "Tables", "Grouped bars", "Raw results"])

    with tabs[0]:
        col1, col2, col3 = st.columns([1, 1, 1])
        line_metric_key = col1.selectbox(
            "Metric",
            options=list(pd.unique(fairness_results["metric"])),
            format_func=lambda key: fairness_label_map.get(key, key),
            key="adult_line_metric",
        )
        ratio_type = col2.radio("Sweep", options=["ir", "gr"], horizontal=True, key="adult_line_ratio")
        fill = col3.radio("Band", options=["std", "err"], horizontal=True, key="adult_line_fill")
        fig1 = plot_case_line(
            fairness_results,
            line_metric_key,
            fairness_label_map.get(line_metric_key, line_metric_key),
            ratio_type,
            fill=fill,
        )
        fig2 = plot_case_line(
            fairness_results,
            line_metric_key,
            fairness_label_map.get(line_metric_key, line_metric_key),
            ratio_type,
            fill=fill,
            absolute=True,
        )
        left, right = st.columns(2)
        left.pyplot(fig1, use_container_width=True)
        right.pyplot(fig2, use_container_width=True)

    with tabs[1]:
        ratio_type = st.radio("NaN sweep", options=["ir", "gr"], horizontal=True, key="adult_nan_ratio")
        metric_keys = list(pd.unique(fairness_results["metric"]))
        fig = plot_case_nan(fairness_results, metric_keys, fairness_label_map, ratio_type)
        st.pyplot(fig, use_container_width=True)

    with tabs[2]:
        ratio_type = st.radio("Combined sweep", options=["ir", "gr"], horizontal=True, key="adult_all_ratio")
        fill = st.radio("Combined band", options=["std", "err"], horizontal=True, key="adult_all_fill")
        metric_keys = list(pd.unique(fairness_results["metric"]))
        fig = plot_case_line_all(fairness_results, metric_keys, fairness_label_map, ratio_type, fill=fill)
        st.pyplot(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("Aggregated tables")
        fairness_agg = aggregate_case_results(fairness_results)
        performance_agg = aggregate_case_results(performance_results)
        st.write("Fairness metrics")
        st.dataframe(fairness_agg, use_container_width=True, hide_index=True)
        st.write("Classifier performance")
        st.dataframe(performance_agg, use_container_width=True, hide_index=True)
        st.download_button(
            "Download fairness aggregation CSV",
            data=dataframe_csv_bytes(fairness_agg),
            file_name="adult_fairness_aggregation.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download performance aggregation CSV",
            data=dataframe_csv_bytes(performance_agg),
            file_name="adult_performance_aggregation.csv",
            mime="text/csv",
        )

    with tabs[4]:
        st.subheader("Single-slice grouped bar charts")
        available_gr = sorted(pd.unique(fairness_results["gr"]))
        available_ir = sorted(pd.unique(fairness_results["ir"]))
        col1, col2 = st.columns(2)
        selected_gr = col1.selectbox("GR", options=available_gr, format_func=lambda value: f"{value:.2f}", key="adult_bar_gr")
        selected_ir = col2.selectbox("IR", options=available_ir, format_func=lambda value: f"{value:.2f}", key="adult_bar_ir")
        subset = fairness_results[
            (fairness_results["gr"] == selected_gr) & (fairness_results["ir"] == selected_ir)
        ]
        if subset.empty:
            st.info("No rows for the selected GR/IR combination.")
        else:
            fairness_map: dict[str, dict[str, float]] = {}
            for clf_name, group in subset.groupby("clf"):
                fairness_map[clf_name] = {
                    fairness_label_map.get(mk, mk): float(group[group["metric"] == mk]["value"].mean())
                    for mk in pd.unique(subset["metric"])
                }
            title = f"Fairness metrics for classifiers; GR = {selected_gr:.2f}, IR = {selected_ir:.2f}"
            fig1 = plot_case_grouped_bar_by_metric(fairness_map, title=title)
            fig2 = plot_case_grouped_bar_by_classifier(fairness_map, title=title)
            left, right = st.columns(2)
            left.pyplot(fig1, use_container_width=True)
            right.pyplot(fig2, use_container_width=True)

    with tabs[5]:
        st.subheader("Raw results")
        st.write("Fairness results")
        st.dataframe(fairness_results.head(1000), use_container_width=True)
        st.write("Performance results")
        st.dataframe(performance_results.head(1000), use_container_width=True)
        st.download_button(
            "Download raw fairness CSV",
            data=dataframe_csv_bytes(fairness_results),
            file_name="adult_fairness_results.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download raw performance CSV",
            data=dataframe_csv_bytes(performance_results),
            file_name="adult_performance_results.csv",
            mime="text/csv",
        )

def render_stereotypical_page() -> None:
    st.header("Stereotypical bias study")
    st.write(
        "Explore how the Stereotypical Ratio correlates with fairness metrics. "
        "Three variants: **SR_p** (j's share of positive predictions), "
        "**SR_n** (j's share of negative predictions), "
        "**SR_c** = √(SR_p·SR_n) (combined). SR = GR_j is the proportional baseline."
    )

    _prog_bar = st.empty()
    _prog_cap = st.empty()

    # Map display name → (column, ratio_type string for analysis functions)
    _SR_VARIANTS: dict[str, tuple[str, str]] = {
        "SR_p": ("stereotypical_ratio", "sr"),
        "SR_n": ("stereotypical_ratio_negative", "sr_n"),
        "SR_c": ("stereotypical_ratio_combined", "sr_c"),
    }

    with st.sidebar:
        st.subheader("Stereotypical bias study")
        data_source = st.radio("Dataset source", ["Synthetic", "Adult"], key="stereo_source")

        if data_source == "Synthetic":
            synth_mode = st.radio(
                "Dataset mode",
                ["Exact enumeration", "Monte Carlo sample", "Load pickle"],
                key="stereo_synth_mode",
            )
            total = st.number_input("n (total samples)", min_value=1, value=24, step=1, key="stereo_n")
            est = count_confusion_matrices(int(total))
            st.caption(f"All possible matrices: {est:,}")
            if est > 20_000_000:
                st.warning("Exact generation at this n may be too heavy. Use Monte Carlo or reduce n.")
            draws = (
                int(st.number_input("Monte Carlo draws", min_value=100, value=200_000, step=100, key="stereo_draws"))
                if synth_mode == "Monte Carlo sample" else 200_000
            )
            max_rows = int(st.number_input("Exact row cap", min_value=1000, value=20_000_000, step=1000, key="stereo_cap"))
            pickle_path = st.text_input("Pickle path", value="", key="stereo_pickle") if synth_mode == "Load pickle" else ""
            seed = int(st.number_input("Random seed", min_value=0, value=2137, step=1, key="stereo_seed"))

            if st.button("Build synthetic dataset", type="primary", key="stereo_build"):
                try:
                    df = _build_synthetic_dataset(synth_mode, int(total), max_rows, draws, seed, pickle_path)
                    st.session_state["stereo_df"] = df
                    st.session_state["stereo_label"] = f"Synthetic (n={int(total)})"
                    st.success("Dataset ready.")
                except Exception as exc:
                    st.error(str(exc))

        else:  # Adult
            adult_mode = st.radio("File source", ["Local path", "Upload file"], key="stereo_adult_mode")
            stereo_upload = (
                st.file_uploader("adult.data", type=["data", "csv", "txt"], key="stereo_adult_upload")
                if adult_mode == "Upload file" else None
            )
            stereo_path = (
                st.text_input("Path to adult.data", value="data/adult.data", key="stereo_adult_path")
                if adult_mode == "Local path" else "data/adult.data"
            )
            sweep_ratios = st.multiselect(
                "Ratios to sweep", options=paper_ratio_sweep(), default=paper_ratio_sweep(),
                format_func=lambda v: f"{v:.2f}", key="stereo_sweep_ratios",
            )
            fixed_ratio = st.number_input("Fixed ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="stereo_fixed_ratio")
            sample_size = st.number_input("Subset size", min_value=100, value=1100, step=100, key="stereo_sample_size")
            holdout_splits = st.number_input("Holdout splits", min_value=1, value=20, step=1, key="stereo_splits")
            test_size = st.slider("Test size", min_value=0.1, max_value=0.9, value=0.33, step=0.01, key="stereo_test_size")
            seed = int(st.number_input("Random seed", min_value=0, value=2137, step=1, key="stereo_seed"))
            classifiers = st.multiselect(
                "Classifiers", options=list(CLASSIFIERS.keys()), default=list(CLASSIFIERS.keys()),
                key="stereo_classifiers",
            )

            if st.button("Collect confusion matrices", type="primary", key="stereo_run"):
                try:
                    if adult_mode == "Upload file":
                        if stereo_upload is None:
                            raise ValueError("Upload the Adult dataset first.")
                        adult_df = load_adult_dataset(stereo_upload.getvalue())
                    else:
                        adult_df = load_adult_dataset(stereo_path)

                    def _prog(frac: float, msg: str) -> None:
                        _prog_bar.progress(min(frac, 1.0), text=msg)
                        _prog_cap.caption(msg)

                    raw_df = collect_adult_confusion_matrices(
                        adult_df,
                        ratio_values=sweep_ratios,
                        fixed_ratio=float(fixed_ratio),
                        sample_size=int(sample_size),
                        holdout_splits=int(holdout_splits),
                        test_size=float(test_size),
                        classifier_names=classifiers,
                        random_state=seed,
                        progress_callback=_prog,
                    )
                    _prog_bar.empty()
                    _prog_cap.empty()
                    st.session_state["stereo_df"] = add_base_columns(raw_df)
                    st.session_state["stereo_label"] = "Adult"
                    st.success(f"Collected {len(raw_df):,} confusion matrices.")
                except Exception as exc:
                    st.error(str(exc))

    df: pd.DataFrame | None = st.session_state.get("stereo_df")
    if df is None:
        st.info("Build a dataset from the sidebar to begin.")
        return

    if "stereotypical_ratio" not in df.columns:
        df = add_base_columns(df.copy())
        st.session_state["stereo_df"] = df

    dataset_label: str = st.session_state.get("stereo_label", "Unknown")
    available_ir = sorted(pd.unique(df["imbalance_ratio"].dropna()))
    available_gr = sorted(pd.unique(df["group_ratio_j"].dropna()))
    # Pre-compute available values for each SR variant
    all_avail_sr: dict[str, list[float]] = {
        col: sorted(v for v in pd.unique(df[col]) if np.isfinite(v))
        for col in [c for _, (c, _) in _SR_VARIANTS.items() if c in df.columns]
    }
    available_sr_p = all_avail_sr.get("stereotypical_ratio", [])
    default_atol = 0.015 if len(available_ir) > 20 else 0.06

    fairness_specs = fairness_metric_specs()
    fairness_label_map = {spec.key: spec.label for spec in fairness_specs}
    all_label_map = {**fairness_label_map, **{s.key: s.label for s in list_metrics("fairness_frn")}}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Dataset", dataset_label)
    c3.metric("Unique SR_p values", len(available_sr_p))
    c4.metric("Unique IR values", len(available_ir))

    tabs = st.tabs(["Histogram grids", "Metric vs SR", "Perfect fairness / NaN", "Sensitivity ranking", "Data table"])

    with tabs[0]:
        st.subheader("Histogram grids")
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        hist_metric_key = col1.selectbox(
            "Metric", options=[s.key for s in fairness_specs],
            format_func=lambda k: fairness_label_map[k], key="stereo_hist_metric",
        )
        hist_sr_variant = col2.radio("SR variant", list(_SR_VARIANTS), horizontal=True, key="stereo_hist_sr_variant")
        bins = int(col3.number_input("Histogram bins", min_value=5, value=109, step=1, key="stereo_hist_bins"))
        show_nan_bar = col4.checkbox("Show undefined-value bar", value=True, key="stereo_hist_nan_bar")
        hist_smoothing = smoothing_toggle(hist_metric_key, "stereo_hist_smoothing")
        hist_frn = frn_toggle(hist_metric_key, "stereo_hist_frn")
        active_hist_key = resolve_frn_key(hist_metric_key, hist_frn)
        hist_sr_col, _ = _SR_VARIANTS[hist_sr_variant]
        avail_sr_hist = all_avail_sr.get(hist_sr_col, [])
        default_sr_panel = (
            list(dict.fromkeys(min(avail_sr_hist, key=lambda v, t=t: abs(v - t)) for t in [0.08, 0.25, 0.5, 0.75, 0.92]))
            if avail_sr_hist else []
        )
        selected_sr = sorted(st.multiselect(
            f"{hist_sr_variant} panel values", options=avail_sr_hist,
            default=[v for v in default_sr_panel if v in avail_sr_hist],
            format_func=ratio_label, key=f"stereo_hist_sr_{hist_sr_variant}",
        ))
        if selected_sr:
            hist_df = apply_smoothing_override(df, active_hist_key, hist_smoothing)
            fig = plot_histogram_grid_sr(
                hist_df, active_hist_key,
                all_label_map.get(active_hist_key, fairness_label_map[hist_metric_key]),
                selected_sr, sr_col=hist_sr_col, bins=bins, show_nan_bar=show_nan_bar,
            )
            st.pyplot(fig, use_container_width=True)
            st.download_button(
                "Download histogram grid (PNG)", data=figure_png_bytes(fig),
                file_name=f"histogram_{hist_sr_variant}_{active_hist_key}.png", mime="image/png",
            )
        else:
            st.info(f"Pick at least one {hist_sr_variant} value.")

    with tabs[1]:
        st.subheader("Metric vs SR")
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        line_metric_key = col1.selectbox(
            "Metric", options=[s.key for s in fairness_specs],
            format_func=lambda k: fairness_label_map[k], key="stereo_line_metric",
        )
        line_sr_variant = col2.radio("SR variant", list(_SR_VARIANTS), horizontal=True, key="stereo_line_sr_variant")
        fixed_gr = col3.selectbox(
            "Fixed GR", options=available_gr, index=len(available_gr) // 2,
            format_func=ratio_label, key="stereo_fixed_gr",
        )
        fill = col4.radio("Band", options=["std", "err"], horizontal=True, key="stereo_line_fill")
        line_smoothing = smoothing_toggle(line_metric_key, "stereo_line_smoothing")
        line_frn = frn_toggle(line_metric_key, "stereo_line_frn")
        active_line_key = resolve_frn_key(line_metric_key, line_frn)
        line_sr_col, _ = _SR_VARIANTS[line_sr_variant]
        default_ir_lines = available_ir[:: max(1, len(available_ir) // 5)][:5]
        sweep_ir_values = sorted(st.multiselect(
            "IR values to compare", options=available_ir,
            default=[v for v in default_ir_lines if v in available_ir],
            format_func=ratio_label, key="stereo_sweep_ir",
        ))
        if not sweep_ir_values:
            st.info("Select at least one IR value to compare.")
        else:
            line_df = apply_smoothing_override(df, active_line_key, line_smoothing)
            multi_df = _cached_metric_means_by_sr_multi_ir(
                line_df, active_line_key, tuple(sweep_ir_values),
                line_sr_col, float(fixed_gr), default_atol,
            )
            if multi_df.empty:
                st.info("No data for the current GR slice. Try adjusting Fixed GR.")
            else:
                fig1 = plot_metric_vs_sr_by_ir(
                    multi_df, active_line_key,
                    all_label_map.get(active_line_key, fairness_label_map[line_metric_key]),
                    sweep_ir_values, absolute=False, show_bands=(fill == "std"),
                    gr_value=float(fixed_gr), sr_label=line_sr_variant,
                )
                fig2 = plot_metric_vs_sr_by_ir(
                    multi_df, active_line_key,
                    all_label_map.get(active_line_key, fairness_label_map[line_metric_key]),
                    sweep_ir_values, absolute=True, show_bands=(fill == "std"),
                    gr_value=float(fixed_gr), sr_label=line_sr_variant,
                )
                left, right = st.columns(2)
                left.pyplot(fig1, use_container_width=True)
                right.pyplot(fig2, use_container_width=True)
                st.download_button(
                    "Download metric-vs-SR CSV", data=dataframe_csv_bytes(multi_df),
                    file_name=f"metric_vs_{line_sr_variant}_{active_line_key}.csv", mime="text/csv",
                )

    with tabs[2]:
        st.subheader("Probability of perfect fairness and undefined values")
        col1, col2, col3, col4 = st.columns([1.4, 1, 1, 1])
        ppf_metric_keys = col1.multiselect(
            "Metrics", options=[s.key for s in fairness_specs],
            default=[s.key for s in fairness_specs], format_func=lambda k: fairness_label_map[k],
            key="stereo_ppf_metrics",
        )
        ppf_sr_variant = col2.radio("SR variant", list(_SR_VARIANTS), horizontal=True, key="stereo_ppf_sr_variant")
        epsilon = float(col3.number_input("Epsilon", min_value=0.0, value=0.0, step=0.001, key="stereo_ppf_eps"))
        ppf_smoothing = (
            st.checkbox("Haldane-Anscombe smoothing", value=True, key="stereo_ppf_smoothing")
            if any(k in _SMOOTHABLE_METRICS for k in ppf_metric_keys) else True
        )
        ppf_frn = (
            st.checkbox("Apply FRN to supported metrics", value=False, key="stereo_ppf_frn")
            if any(k in _FRN_KEY_MAP for k in ppf_metric_keys) else False
        )
        active_ppf_keys = apply_frn_to_keys(ppf_metric_keys, ppf_frn)
        _, ppf_ratio_type = _SR_VARIANTS[ppf_sr_variant]
        if ppf_metric_keys:
            ppf_work = df
            for k in active_ppf_keys:
                ppf_work = apply_smoothing_override(ppf_work, k, ppf_smoothing)
            ppf_df = probability_of_perfect_fairness(ppf_work, active_ppf_keys, ppf_ratio_type, epsilon=epsilon)
            nan_df = probability_of_nan(ppf_work, active_ppf_keys, ppf_ratio_type)
            fig1 = plot_probability_lines(
                ppf_df, active_ppf_keys, all_label_map, ppf_ratio_type,
                title=f"Probability of perfect fairness vs {ppf_sr_variant}",
                y_label="Probability of perfect fairness", y_max=1.0,
            )
            fig2 = plot_probability_lines(
                nan_df, active_ppf_keys, all_label_map, ppf_ratio_type,
                title=f"Probability of undefined values vs {ppf_sr_variant}",
                y_label="Probability of undefined value", y_max=1.0,
            )
            left, right = st.columns(2)
            left.pyplot(fig1, use_container_width=True)
            right.pyplot(fig2, use_container_width=True)
            st.download_button(
                "Download perfect-fairness CSV", data=dataframe_csv_bytes(ppf_df),
                file_name=f"{ppf_sr_variant}_perfect_fairness.csv", mime="text/csv",
            )
            st.download_button(
                "Download undefined-value CSV", data=dataframe_csv_bytes(nan_df),
                file_name=f"{ppf_sr_variant}_undefined_probability.csv", mime="text/csv",
            )

    with tabs[3]:
        st.subheader("Sensitivity ranking")
        st.write(
            "**Stratified** Spearman ρ: correlation is computed within each (IR, GR) stratum, "
            "then combined via Fisher Z-weighting."
        )
        all_fairness_keys = [s.key for s in fairness_specs]
        sens_work = df.copy()
        for k in all_fairness_keys:
            if k in _SMOOTHABLE_METRICS:
                sens_work = apply_smoothing_override(sens_work, k, True)

        sens_dfs: dict[str, pd.DataFrame] = {
            variant: _cached_sr_sensitivity_stratified(sens_work, tuple(all_fairness_keys), col)
            for variant, (col, _) in _SR_VARIANTS.items()
            if col in sens_work.columns
        }

        any_data = any(not d.empty and not d["spearman_r"].isna().all() for d in sens_dfs.values())
        if not any_data:
            st.info("Not enough data to compute sensitivity statistics.")
        else:
            sens_cols = st.columns(len(sens_dfs))
            for idx, (variant, sens_df) in enumerate(sens_dfs.items()):
                with sens_cols[idx]:
                    st.markdown(f"**{variant}**")
                    if sens_df.empty or sens_df["spearman_r"].isna().all():
                        st.info("No data.")
                    else:
                        fig_rho = plot_sr_sensitivity(
                            sens_df, fairness_label_map,
                            value_col="spearman_r",
                            title=f"Stratified ρ vs {variant}",
                        )
                        st.pyplot(fig_rho, use_container_width=True)

            st.subheader("Combined |ρ| table — all SR variants")
            tbl_rows = []
            for key in all_fairness_keys:
                row: dict = {"Metric": fairness_label_map.get(key, key)}
                for variant, sens_df in sens_dfs.items():
                    match = sens_df[sens_df["metric"] == key]
                    rho = float(abs(match["spearman_r"].iloc[0])) if not match.empty and match["spearman_r"].notna().any() else np.nan
                    row[f"|ρ| {variant}"] = rho
                tbl_rows.append(row)
            combined_tbl = pd.DataFrame(tbl_rows)
            sort_col = next((f"|ρ| {v}" for v in _SR_VARIANTS if f"|ρ| {v}" in combined_tbl.columns), "Metric")
            combined_tbl = combined_tbl.sort_values(sort_col, ascending=True)
            st.dataframe(combined_tbl, use_container_width=True, hide_index=True)
            st.caption(
                "Most SR-resistant metric = lowest |ρ| across all three variants. "
                "A metric is robustly resistant only if it scores low on SR_p, SR_n **and** SR_c."
            )
            st.download_button(
                "Download sensitivity CSV", data=dataframe_csv_bytes(combined_tbl),
                file_name="sr_sensitivity_stratified.csv", mime="text/csv",
            )

    with tabs[4]:
        st.subheader("Data table")
        _render_data_table_tab(
            df,
            [s.key for s in fairness_specs],
            fairness_label_map,
            widget_prefix="stereo",
            show_sr=True,
        )

def render_fairness_benchmark_page() -> None:
    st.header("Fairness detection benchmark")
    st.write(
        "Pick a discrimination type, inject a controlled gap \u03b4 into synthetic confusion matrices, "
        "and see **exactly what score each metric produces** \u2014 from perfectly fair (\u03b4\u00a0=\u00a00) "
        "to strongly discriminating."
    )

    _prog_bar = st.empty()
    _prog_cap = st.empty()

    with st.sidebar:
        st.subheader("Benchmark setup")
        bench_n = int(st.number_input("n (total samples)", min_value=20, value=200, step=20, key="bench_n"))
        bench_ir = st.slider("Imbalance ratio (IR)", 0.05, 0.95, 0.5, 0.05, key="bench_ir",
                             help="Overall positive-class fraction. Both groups share this base rate.")
        bench_gr = st.slider("Group ratio (GR)", 0.05, 0.95, 0.5, 0.05, key="bench_gr",
                             help="j-group's fraction of total samples.")
        bench_disc_type = st.radio(
            "Discrimination type",
            options=["tpr_gap", "fpr_gap", "both"],
            format_func=lambda v: {
                "tpr_gap": "TPR gap  (j recall \u2212 i recall = \u03b4)",
                "fpr_gap": "FPR gap  (j false-alarm \u2212 i false-alarm = \u03b4)",
                "both":    "Both     (TPR gap = FPR gap = \u03b4)",
            }[v],
            key="bench_disc_type",
        )
        bench_max_delta = st.slider("Max |\u03b4| for inner steps", 0.1, 0.9, 0.8, 0.05, key="bench_max_delta",
                                    help="9 steps from \u2212max to +max, plus forced \u00b10.99 extremes (11 columns total).")
        bench_seed = int(st.number_input("Random seed", min_value=0, value=2137, step=1, key="bench_seed"))

        if st.button("Run benchmark", type="primary", key="bench_run"):
            try:
                _prog_bar.progress(0.05, text="Generating confusion matrices\u2026")
                _prog_cap.caption("Injecting discrimination and computing metrics\u2026")
                # 9 inner steps (nice round numbers) + forced ±0.99 extremes = 11 columns.
                inner = np.linspace(-bench_max_delta, bench_max_delta, 9).tolist()
                delta_values = sorted({round(d, 10) for d in inner + [-0.99, 0.99]})
                all_fairness_keys = [s.key for s in fairness_metric_specs()]
                df = sweep_discrimination(
                    bench_n, bench_ir, bench_gr,
                    delta_values, bench_disc_type,
                    400, all_fairness_keys, bench_seed,
                )
                _prog_bar.progress(1.0, text="Done.")
                _prog_bar.empty()
                _prog_cap.empty()
                if df.empty:
                    raise ValueError("No matrices could be generated for these parameters.")
                st.session_state["fairness_benchmark_df"] = df
                st.session_state["fairness_benchmark_params"] = {
                    "n": bench_n, "ir": bench_ir, "gr": bench_gr,
                    "disc_type": bench_disc_type, "max_delta": bench_max_delta,
                    "seed": bench_seed,
                }
                st.success(f"Ready \u2014 {len(df):,} rows across 9 \u03b4 values.")
            except Exception as exc:
                _prog_bar.empty()
                _prog_cap.empty()
                st.error(str(exc))

    df: pd.DataFrame | None = st.session_state.get("fairness_benchmark_df")
    if df is None:
        st.info("Configure the benchmark in the sidebar and click **Run benchmark**.")
        return

    params: dict = st.session_state.get("fairness_benchmark_params", {})
    disc_type_used: str = params.get("disc_type", "tpr_gap")

    fairness_specs = fairness_metric_specs()
    fairness_label_map = {s.key: s.label for s in fairness_specs}
    available_keys = [s.key for s in fairness_specs if s.key in df.columns]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n", params.get("n", "?"))
    c2.metric("IR", f"{params.get('ir', 0.5):.2f}")
    c3.metric("GR", f"{params.get('gr', 0.5):.2f}")
    c4.metric("Discrimination", {"tpr_gap": "TPR gap", "fpr_gap": "FPR gap", "both": "Both"}.get(disc_type_used, disc_type_used))

    tabs = st.tabs(["Scores table", "Response curves", "Detection power"])

    with tabs[0]:
        st.subheader("Real scores at each discrimination level")
        st.caption(
            "Mean metric value across 400 random base-rate draws per \u03b4. "
            "Each cell is literally **what the metric reads** when that much discrimination is injected. "
            "The centre column (\u03b4\u00a0=\u00a00.00) is the fair baseline \u2014 all metrics should read near 0 there."
        )

        col_left, col_right = st.columns([3, 1])
        table_metric_keys = col_left.multiselect(
            "Metrics in table",
            options=available_keys,
            default=available_keys,
            format_func=lambda k: fairness_label_map.get(k, k),
            key="bench_table_metrics",
        )
        table_abs = col_right.checkbox(
            "Show absolute values", value=False, key="bench_table_abs",
            help="Collapses direction \u2014 useful when you care about magnitude only.",
        )

        if table_metric_keys:
            deltas_sorted = sorted(df["true_delta"].unique())
            pivot_rows: dict[str, dict] = {}
            for key in table_metric_keys:
                label = fairness_label_map.get(key, key)
                row: dict = {}
                for d in deltas_sorted:
                    vals = df.loc[df["true_delta"] == d, key].dropna()
                    if table_abs:
                        vals = vals.abs()
                    row[f"{d:+.2f}"] = float(vals.mean()) if len(vals) else np.nan
                pivot_rows[label] = row

            pivot_df = pd.DataFrame(pivot_rows).T
            pivot_df.index.name = "Metric"

            vmax = float(max(0.01, pivot_df.abs().max(skipna=True).max(skipna=True)))
            cmap = "YlOrRd" if table_abs else "coolwarm"
            vmin = 0.0 if table_abs else -vmax
            styled = (
                pivot_df.style
                .background_gradient(cmap=cmap, vmin=vmin, vmax=vmax, axis=None)
                .format("{:.3f}")
                .set_properties(**{"text-align": "center", "font-size": "13px"})
            )
            st.dataframe(styled, use_container_width=True)
            st.caption(
                "\U0001f7e6 positive (j favoured) \u00b7 \u26aa near-zero (fair) \u00b7 \U0001f7e5 negative (j disadvantaged)  \n"
                "\u26a0\ufe0f **CQA / CYA** are unsigned (always \u2265\u00a00) and cap at **1/\u221a2 \u2248 0.707** "
                "under `TPR gap` or `FPR gap` mode because only one decision stratum is discriminated "
                "(the other has OR\u00a0=\u00a01 \u2192 Q\u00a0=\u00a00). "
                "Switch to **Both** to let both strata discriminate and reach 1.0.  \n"
                "\u26a0\ufe0f **MI / NMI** are also unsigned. **PED = 0** under TPR gap (FPR equalized by design). "
                "**DIR** is centred at 1\u00a0(fair), not 0 \u2014 use **log DIR** for a zero-centred reading."
            )
            st.download_button(
                "Download scores table CSV",
                data=pivot_df.to_csv().encode("utf-8"),
                file_name="benchmark_scores_table.csv",
                mime="text/csv",
            )

    with tabs[1]:
        st.subheader("Metric response curves")
        st.caption(
            "Mean \u00b1\u202fstd across base-rate draws at each \u03b4. "
            "A sensitive metric rises steeply from zero \u2014 "
            "a flat line means it misses the discrimination entirely."
        )
        col1, col2 = st.columns([3, 1])
        curve_metric_keys = col1.multiselect(
            "Metrics",
            options=available_keys,
            default=available_keys,
            format_func=lambda k: fairness_label_map.get(k, k),
            key="bench_curve_metrics",
        )
        curve_abs = col2.checkbox("Absolute values", value=False, key="bench_curve_abs")

        if curve_metric_keys:
            fig_signed = plot_discrimination_sweep(
                df, curve_metric_keys, fairness_label_map, disc_type_used, absolute=False,
            )
            fig_abs = plot_discrimination_sweep(
                df, curve_metric_keys, fairness_label_map, disc_type_used, absolute=True,
            )
            left, right = st.columns(2)
            left.pyplot(fig_signed, use_container_width=True)
            right.pyplot(fig_abs, use_container_width=True)
            st.download_button(
                "Download curves CSV",
                data=dataframe_csv_bytes(df[["true_delta"] + curve_metric_keys].dropna(how="all")),
                file_name="benchmark_curves.csv",
                mime="text/csv",
            )
        else:
            st.info("Select at least one metric.")

    with tabs[2]:
        st.subheader("Detection power & false alarm rate")
        st.caption(
            "**Detection power** = P(|metric|\u00a0>\u00a0\u03c4\u00a0|\u00a0discrimination injected). "
            "**False alarm rate** = P(|metric|\u00a0>\u00a0\u03c4\u00a0|\u00a0\u03b4\u00a0=\u00a00, no discrimination). "
            "A useful metric scores high on the first and near-zero on the second."
        )
        col1, col2, col3 = st.columns([1, 1, 2])
        det_threshold = col1.slider(
            "Detection threshold \u03c4", 0.0, 0.5, 0.05, 0.01, key="bench_det_threshold",
            help="Flag a case as discriminating when |metric| > \u03c4.",
        )
        det_null_eps = float(col2.number_input(
            "Fair zone |\u03b4| \u2264 \u03b5", min_value=0.0, max_value=0.2,
            value=0.01, step=0.005, format="%.3f", key="bench_det_null_eps",
            help="Rows with |true_delta| \u2264 \u03b5 count as the fair (null) cases.",
        ))
        det_metric_keys = col3.multiselect(
            "Metrics", options=available_keys, default=available_keys,
            format_func=lambda k: fairness_label_map.get(k, k), key="bench_det_metrics",
        )

        if det_metric_keys:
            bench_df = benchmark_metrics(df, det_metric_keys, float(det_threshold), det_null_eps)
            if bench_df.empty:
                st.info("No results \u2014 try adjusting the threshold or fair-zone \u03b5.")
            else:
                fig_bars = plot_detection_power_bars(bench_df, fairness_label_map)
                st.pyplot(fig_bars, use_container_width=True)

                bench_df["label"] = bench_df["metric"].map(lambda k: fairness_label_map.get(k, k))
                bench_df["net_gain"] = bench_df["detection_power"] - bench_df["false_alarm_rate"]
                display_bench = (
                    bench_df[["label", "detection_power", "false_alarm_rate", "net_gain", "spearman_r"]]
                    .rename(columns={
                        "label": "Metric",
                        "detection_power": "Detection power",
                        "false_alarm_rate": "False alarm rate",
                        "net_gain": "Net gain (power \u2212 alarm)",
                        "spearman_r": "Spearman \u03c1 vs \u03b4",
                    })
                    .sort_values("Net gain (power \u2212 alarm)", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(
                    display_bench.style.format({
                        "Detection power": "{:.3f}",
                        "False alarm rate": "{:.3f}",
                        "Net gain (power \u2212 alarm)": "{:.3f}",
                        "Spearman \u03c1 vs \u03b4": "{:.3f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download detection table CSV",
                    data=dataframe_csv_bytes(display_bench),
                    file_name="benchmark_detection_power.csv",
                    mime="text/csv",
                )
        else:
            st.info("Select at least one metric.")


def render_metric_registry_page() -> None:
    st.header("Metric registry")
    if CUSTOM_METRIC_IMPORT_ERROR:
        st.warning(f"custom_metrics.py failed to import: {CUSTOM_METRIC_IMPORT_ERROR}")

    category_labels = {
        "fairness":     "Fairness metrics",
        "fairness_frn": "Feasible-Range Normalized metrics",
        "performance":  "Performance metrics",
        "component":    "Component metrics",
        "ratio":        "Ratio metrics",
    }

    all_cats = list(dict.fromkeys(s.category for s in list_metrics()))
    for cat in all_cats:
        specs = list_metrics(cat)
        st.subheader(category_labels.get(cat, cat))
        for spec in specs:
            with st.expander(spec.label):
                st.latex(spec.formula)
                st.caption(spec.description)

st.title("Fairness Measures Explorer")
st.caption(
    "Interactive Streamlit recreation of the synthetic analyses and Adult case study from the provided fairness-measure repository."
)

page = st.radio(
    "Workflow",
    options=["Synthetic study", "Adult case study", "Stereotypical bias study", "Fairness benchmark", "Metric registry"],
    horizontal=True,
)

if page == "Synthetic study":
    render_synthetic_page()
elif page == "Adult case study":
    render_case_study_page()
elif page == "Stereotypical bias study":
    render_stereotypical_page()
elif page == "Fairness benchmark":
    render_fairness_benchmark_page()
else:
    render_metric_registry_page()
